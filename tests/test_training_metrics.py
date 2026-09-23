from __future__ import annotations

import numpy as np
import pytest

from cmdrisk.training.calibration import Calibration
from cmdrisk.training.metrics import (
    average_precision,
    binary_classification_metrics,
    detection_delay_metrics,
    expected_calibration_error,
    multilabel_metrics,
    promotion_report,
    roc_auc,
)
from cmdrisk.training.pipeline import _metrics


def test_binary_discrimination_and_calibration_metrics() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.float64)
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    metrics = binary_classification_metrics(labels, probabilities, threshold=0.5)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["fpr"] == 0.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["recall_ci_low"] <= metrics["recall"] <= metrics["recall_ci_high"] <= 1.0


def test_average_precision_roc_auc_and_ece_reference_values() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.float64)
    probabilities = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float64)
    assert average_precision(labels, probabilities) == pytest.approx(1.0)
    assert roc_auc(labels, probabilities) == pytest.approx(1.0)
    assert expected_calibration_error(labels, probabilities, n_bins=2) < 0.2


def test_detection_delay_uses_first_latched_alert() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1, 1, 1], dtype=np.float64)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    metrics = detection_delay_metrics(
        labels,
        probabilities,
        session_ids=["a", "a", "a", "a", "b", "b", "b", "b"],
        prefix_indices=[0, 1, 2, 3, 0, 1, 2, 3],
        threshold=0.5,
    )
    assert metrics["malicious_sessions"] == 2
    assert metrics["detected_sessions"] == 1
    assert metrics["missed_sessions"] == 1
    assert metrics["alert_command_numbers"] == [3]
    assert metrics["median_detection_delay_commands"] == 0.0


def test_multilabel_metrics_ignore_zero_support_heads() -> None:
    labels = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.float64)
    probabilities = np.asarray(
        [[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], dtype=np.float64
    )
    metrics = multilabel_metrics(labels, probabilities, thresholds=0.5)
    assert metrics["macro_f1_all"] == metrics["macro_f1"]
    assert metrics["macro_f1_all"] == pytest.approx(1.0 / 3.0)
    assert metrics["supported_macro_precision"] == pytest.approx(1.0)
    assert metrics["supported_macro_recall"] == pytest.approx(1.0)
    assert metrics["supported_macro_f1"] == pytest.approx(1.0)
    assert metrics["supported_heads"] == 1
    assert metrics["total_heads"] == 3
    assert metrics["unsupported_heads"] == 2
    assert metrics["macro_f1_support_ge_1"] == pytest.approx(1.0)
    assert metrics["heads_support_ge_1"] == 1
    assert metrics["macro_f1_support_ge_10"] is None
    assert metrics["macro_f1_support_ge_50"] is None
    assert metrics["heads_support_ge_50"] == 0


def test_multilabel_metrics_floor_fifty_ignores_low_support_heads() -> None:
    labels = np.zeros((51, 2), dtype=np.float64)
    labels[:, 0] = 1.0
    labels[0, 1] = 1.0
    probabilities = np.zeros((51, 2), dtype=np.float64)
    probabilities[:, 0] = 0.9
    probabilities[:, 1] = 0.1

    metrics = multilabel_metrics(labels, probabilities, thresholds=0.5)

    assert metrics["macro_f1_all"] == pytest.approx(0.5)
    assert metrics["supported_macro_f1"] == pytest.approx(0.5)
    assert metrics["macro_f1_support_ge_1"] == pytest.approx(0.5)
    assert metrics["macro_f1_support_ge_10"] == pytest.approx(1.0)
    assert metrics["macro_f1_support_ge_50"] == pytest.approx(1.0)
    assert metrics["heads_support_ge_1"] == 2
    assert metrics["heads_support_ge_50"] == 1
    assert metrics["macro_precision_support_ge_50"] == pytest.approx(1.0)
    assert metrics["macro_recall_support_ge_50"] == pytest.approx(1.0)


def _passing_metrics() -> dict[str, float]:
    return {
        "recall": 0.95,
        "fpr": 0.01,
        "pr_auc": 0.95,
        "roc_auc": 0.98,
        "brier": 0.05,
        "ece": 0.02,
    }


def _passing_delay() -> dict[str, float]:
    return {
        "detection_rate": 0.95,
        "median_censored_detection_delay_commands": 2.0,
    }


def _passing_audit() -> dict[str, object]:
    return {
        "eligible_for_promotion": True,
        "resolved_cases": 500,
        "unresolved": 0,
        "prediction_metrics": {"scored_cases": 500, "unpredicted_cases": 0},
        "promotion_blockers": [],
    }


def test_promotion_report_exposes_all_target_checks() -> None:
    report = promotion_report(
        _passing_metrics(),
        _passing_delay(),
        tactic_macro_f1=0.7,
        technique_macro_f1=0.66,
        tactic_macro_f1_all_heads=0.2,
        tactic_macro_f1_support_ge_1=0.8,
        tactic_macro_f1_support_ge_50=0.7,
        tactic_heads_support_ge_50=7,
        technique_macro_f1_all_heads=0.3,
        technique_macro_f1_support_ge_1=0.7,
        technique_macro_f1_support_ge_50=0.66,
        technique_heads_support_ge_50=5,
        proxy_audit_report=_passing_audit(),
    )
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["tactic_macro_f1_all_heads"] == pytest.approx(0.2)
    assert report["tactic_macro_f1_support_ge_1"] == pytest.approx(0.8)
    assert report["tactic_macro_f1_support_ge_50"] == pytest.approx(0.7)
    assert report["tactic_heads_support_ge_50"] == 7
    assert report["technique_macro_f1_all_heads"] == pytest.approx(0.3)
    assert report["technique_macro_f1_support_ge_1"] == pytest.approx(0.7)
    assert report["technique_macro_f1_support_ge_50"] == pytest.approx(0.66)
    assert report["technique_heads_support_ge_50"] == 5
    assert report["targets"]["recall"] == 0.90
    assert report["targets"]["fpr"] == 0.02
    assert report["targets"]["pr_auc"] == 0.90
    assert report["targets"]["roc_auc"] == 0.95
    assert report["targets"]["brier"] == 0.10
    assert report["targets"]["ece"] == 0.05
    assert report["targets"]["detection_delay_commands"] == 3.0


def test_promotion_report_fails_closed_on_missing_or_missed_metrics() -> None:
    missing = _passing_metrics()
    missing.pop("ece")
    report = promotion_report(
        missing,
        _passing_delay(),
        tactic_macro_f1=0.7,
        technique_macro_f1=0.7,
        proxy_audit_report=_passing_audit(),
    )
    assert report["passed"] is False
    assert report["checks"]["ece"] is False
    assert "ece" in report["missing_metrics"]

    missed = _passing_metrics()
    missed["brier"] = 0.11
    report = promotion_report(
        missed,
        _passing_delay(),
        tactic_macro_f1=0.7,
        technique_macro_f1=0.7,
        proxy_audit_report=_passing_audit(),
    )
    assert report["passed"] is False
    assert report["checks"]["brier"] is False
    assert "brier" in report["violations"]


def test_promotion_report_requires_complete_500_case_audit() -> None:
    report = promotion_report(
        _passing_metrics(),
        _passing_delay(),
        tactic_macro_f1=0.7,
        technique_macro_f1=0.7,
        proxy_audit_report=None,
    )
    assert report["passed"] is False
    assert report["checks"]["proxy_audit"] is False

    incomplete = _passing_audit()
    incomplete["resolved_cases"] = 499
    report = promotion_report(
        _passing_metrics(),
        _passing_delay(),
        tactic_macro_f1=0.7,
        technique_macro_f1=0.7,
        proxy_audit_report=incomplete,
    )
    assert report["passed"] is False
    assert report["proxy_audit"]["resolved_cases"] == 499


def test_tactic_label_coverage_gate_uses_floor_fifty_and_ten_heads() -> None:
    tactic_ids = [f"TA{index:04d}" for index in range(14)]

    def report_with_support(support: list[int]) -> dict[str, object]:
        return promotion_report(
            _passing_metrics(),
            _passing_delay(),
            tactic_macro_f1=0.7,
            technique_macro_f1=0.7,
            proxy_audit_report=_passing_audit(),
            tactic_label_support=support,
            tactic_ids=tactic_ids,
            require_tactic_label_coverage=True,
        )

    failing = report_with_support([50] * 7 + [49] * 7)
    assert failing["passed"] is False
    assert failing["checks"]["tactic_label_coverage"] is False
    assert failing["tactic_label_coverage"]["observed"] == 7
    assert failing["tactic_label_coverage"]["required"] == 10
    assert failing["tactic_label_coverage"]["support_floor"] == 50
    assert failing["tactic_label_coverage"]["covered_tactics"] == tactic_ids[:7]
    assert failing["tactic_label_coverage"]["uncovered_tactics"] == tactic_ids[7:]

    passing = report_with_support([50] * 10 + [49] * 4)
    assert passing["passed"] is True
    assert passing["checks"]["tactic_label_coverage"] is True
    assert passing["tactic_label_coverage"]["observed"] == 10


def _slice_predictions(required_slices: list[str]) -> dict[str, np.ndarray]:
    rows_per_slice = 2
    count = rows_per_slice * len(required_slices)
    session_ids: list[str] = []
    slices: list[str] = []
    for slice_index, slice_name in enumerate(required_slices):
        session_ids.extend([f"s{slice_index}"] * rows_per_slice)
        slices.extend([slice_name] * rows_per_slice)
    labels = np.tile(np.asarray([0.0, 1.0]), len(required_slices))
    logits = np.where(labels > 0, 2.0, 0.0)
    multilabel = labels.reshape(-1, 1)
    return {
        "malicious_logits": logits,
        "malicious_labels": labels,
        "tactic_logits": logits.reshape(-1, 1),
        "tactic_labels": multilabel,
        "technique_logits": logits.reshape(-1, 1),
        "technique_labels": multilabel,
        "session_ids": np.asarray(session_ids, dtype=object),
        "prefix_indices": np.tile(np.asarray([0, 1]), len(required_slices)),
        "command_counts": np.full(count, 2, dtype=np.int64),
        "source_kinds": np.asarray(slices, dtype=object),
        "evaluation_slices": np.asarray(slices, dtype=object),
    }


def _slice_calibration() -> Calibration:
    return Calibration(
        malicious_temperature=1.0,
        malicious_threshold=0.5,
        tactic_temperature=1.0,
        technique_temperature=1.0,
        tactic_thresholds={"TA0002": 0.5},
        technique_thresholds={"T1105": 0.5},
    )


def test_pipeline_fails_closed_on_missing_required_slice() -> None:
    metrics = _metrics(
        _slice_predictions(["real_only"]),
        _slice_calibration(),
        tactic_ids=["TA0002"],
        technique_ids=["T1105"],
    )
    assert metrics["promotion"]["checks"]["evaluation_slices_present"] is False
    assert metrics["promotion"]["missing_evaluation_slices"] == [
        "ood_2024",
        "real_plus_semi_real",
        "synthetic_stress",
    ]
    assert metrics["promotion"]["passed"] is False


def test_pipeline_applies_thresholds_to_every_required_slice() -> None:
    names = ["ood_2024", "real_only", "real_plus_semi_real", "synthetic_stress"]
    metrics = _metrics(
        _slice_predictions(names),
        _slice_calibration(),
        tactic_ids=["TA0002"],
        technique_ids=["T1105"],
    )
    assert metrics["promotion"]["checks"]["evaluation_slices_present"] is True
    assert metrics["promotion"]["checks"]["evaluation_slices_passed"] is False
    assert set(metrics["promotion"]["slice_checks"]) == set(names)
    assert all(
        item["checks"]["brier"] is False
        for item in metrics["promotion"]["slice_checks"].values()
    )


def test_pipeline_fails_closed_on_zero_positive_slice() -> None:
    names = ["ood_2024", "real_only", "real_plus_semi_real", "synthetic_stress"]
    predictions = _slice_predictions(names)
    predictions["malicious_labels"] = np.zeros_like(
        predictions["malicious_labels"], dtype=np.float64
    )
    predictions["malicious_logits"] = np.full_like(
        predictions["malicious_logits"], -10.0, dtype=np.float64
    )
    predictions["tactic_labels"] = np.zeros_like(
        predictions["tactic_labels"], dtype=np.float64
    )
    predictions["technique_labels"] = np.zeros_like(
        predictions["technique_labels"], dtype=np.float64
    )
    metrics = _metrics(
        predictions,
        _slice_calibration(),
        tactic_ids=["TA0002"],
        technique_ids=["T1105"],
    )
    assert metrics["tactic"]["heads_support_ge_50"] == 0
    assert metrics["tactic"]["macro_f1_support_ge_50"] is None
    for slice_check in metrics["promotion"]["slice_checks"].values():
        assert slice_check["passed"] is False
        assert "insufficient_positives" in slice_check["violations"]
        assert slice_check["class_balance"]["positive_count"] == 0
        assert slice_check["class_balance"]["deficient"] is True
        for metric_name in (
            "recall",
            "fpr",
            "precision",
            "pr_auc",
            "roc_auc",
            "ece",
        ):
            assert metric_name in slice_check["undefined_metrics"]
            assert slice_check["metric_values"][metric_name] is None
        assert "brier" in slice_check["defined_metrics"]
        assert "detection_rate" in slice_check["undefined_metrics"]
        assert "tactic_macro_f1" in slice_check["undefined_metrics"]
    assert metrics["promotion"]["checks"]["evaluation_slices_passed"] is False
    assert metrics["promotion"]["passed"] is False


def test_pipeline_fails_closed_on_single_class_malicious_slice() -> None:
    """An all-malicious slice cannot support fpr/roc_auc or a calibrated ECE.

    This is the ``ood_2024`` regression: with no benign sessions the promotion
    code used to treat fpr/fpr-based metrics as passing and reported an ECE of
    ``1 - mean(p)`` that is a confidence statistic, not a calibration error.
    """

    names = ["ood_2024", "real_only", "real_plus_semi_real", "synthetic_stress"]
    predictions = _slice_predictions(names)
    predictions["malicious_labels"] = np.ones_like(
        predictions["malicious_labels"], dtype=np.float64
    )
    predictions["malicious_logits"] = np.full_like(
        predictions["malicious_logits"], 2.0, dtype=np.float64
    )
    metrics = _metrics(
        predictions,
        _slice_calibration(),
        tactic_ids=["TA0002"],
        technique_ids=["T1105"],
    )
    for slice_check in metrics["promotion"]["slice_checks"].values():
        assert slice_check["passed"] is False
        assert "insufficient_negatives" in slice_check["violations"]
        assert slice_check["class_balance"]["negative_count"] == 0
        for metric_name in ("fpr", "precision", "pr_auc", "roc_auc", "ece"):
            assert metric_name in slice_check["undefined_metrics"]
            assert slice_check["metric_values"][metric_name] is None
        # Metrics that only need malicious sessions stay defined.
        assert "detection_rate" in slice_check["defined_metrics"]
        assert "detection_delay_commands" in slice_check["defined_metrics"]
    assert metrics["promotion"]["checks"]["evaluation_slices_passed"] is False
    assert metrics["promotion"]["passed"] is False


def test_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        binary_classification_metrics([0, 1], [0.1, 1.1], threshold=0.5)
