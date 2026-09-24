"""Evaluation helpers shared by training, export, and promotion gates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_ECE_BINS = 15
WILSON_Z_95 = 1.959963984540054
# Macro-F1 is only meaningful when each class has enough evaluation support.
# These floors are measurement-validity rules, not threshold relaxations.
# All-head and floor-1 values remain published for transparency.
TACTIC_SUPPORT_FLOOR = 50
TECHNIQUE_SUPPORT_FLOOR = 50
PROMOTION_TARGETS: dict[str, float] = {
    "recall": 0.90,
    "fpr": 0.02,
    "pr_auc": 0.90,
    "roc_auc": 0.95,
    "brier": 0.10,
    "ece": 0.05,
    "tactic_macro_f1": 0.65,
    "technique_macro_f1": 0.65,
    "tactic_label_coverage": 10.0,
    "detection_rate": 0.90,
    "detection_delay_commands": 3.0,
    "minimum_resolved_audit_cases": 500.0,
}

_LOWER_IS_BETTER = {"fpr", "brier", "ece", "detection_delay_commands"}
_METRIC_NAMES = (
    "recall",
    "fpr",
    "pr_auc",
    "roc_auc",
    "brier",
    "ece",
)


def _validate_binary_inputs(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if targets.size != scores.size or targets.size == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if not np.all(np.isfinite(targets)):
        raise ValueError("labels must be finite")
    if not np.all(np.isfinite(scores)):
        raise ValueError("probabilities must be finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if np.any((targets != 0.0) & (targets != 1.0)):
        raise ValueError("labels must contain only 0 and 1")
    return targets, scores


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = WILSON_Z_95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute the area under the precision-recall curve (average precision).

    The result is defined as ``0.0`` when a split contains no positive labels.
    The implementation follows the step-wise AP definition and avoids an
    optional SciPy/scikit-learn dependency at runtime.
    """

    targets, scores = _validate_binary_inputs(labels, probabilities)
    if not np.any(targets > 0):
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_targets = targets[order]
    true_positives = np.cumsum(sorted_targets)
    predicted_positives = np.arange(1, len(sorted_targets) + 1, dtype=np.float64)
    precision = true_positives / predicted_positives
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute ROC-AUC using the Mann-Whitney U statistic with tie handling."""

    targets, scores = _validate_binary_inputs(labels, probabilities)
    positives = int(np.count_nonzero(targets > 0))
    negatives = int(np.count_nonzero(targets <= 0))
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    positive_rank_sum = float(np.sum(ranks[targets > 0]))
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Return equal-width expected calibration error."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    targets, scores = _validate_binary_inputs(labels, probabilities)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(scores, boundaries[1:-1], right=False), 0, n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(targets[mask]))
        error += (count / len(scores)) * abs(confidence - accuracy)
    return float(error)


def binary_classification_metrics(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float | int]:
    """Return discrimination, calibration, and Wilson-interval metrics."""

    targets, scores = _validate_binary_inputs(labels, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    predictions = scores >= threshold
    positives = targets > 0
    negatives = ~positives
    true_positive = int(np.count_nonzero(predictions & positives))
    false_positive = int(np.count_nonzero(predictions & negatives))
    false_negative = int(np.count_nonzero(~predictions & positives))
    true_negative = int(np.count_nonzero(~predictions & negatives))
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    fpr = false_positive / (false_positive + true_negative) if negatives.any() else 0.0
    recall_low, recall_high = wilson_interval(true_positive, int(np.count_nonzero(positives)))
    fpr_low, fpr_high = wilson_interval(false_positive, int(np.count_nonzero(negatives)))
    brier = float(np.mean((scores - targets) ** 2))
    epsilon = 1e-12
    calibrated = np.clip(scores, epsilon, 1.0 - epsilon)
    log_loss = float(
        -np.mean(targets * np.log(calibrated) + (1.0 - targets) * np.log(1.0 - calibrated))
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "recall_ci_low": recall_low,
        "recall_ci_high": recall_high,
        "fpr": float(fpr),
        "fpr_ci_low": fpr_low,
        "fpr_ci_high": fpr_high,
        "brier": brier,
        "log_loss": log_loss,
        "pr_auc": average_precision(targets, scores),
        "roc_auc": roc_auc(targets, scores),
        "ece": expected_calibration_error(targets, scores, n_bins=ece_bins),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "positive_count": int(np.count_nonzero(positives)),
        "negative_count": int(np.count_nonzero(negatives)),
    }


def detection_delay_metrics(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    session_ids: Sequence[str],
    prefix_indices: Sequence[int],
    threshold: float,
) -> dict[str, float | int | None | list[float] | list[int]]:
    """Summarize first-alert latency for malicious sessions.

    Delay zero means the model alerts on the first prefix whose label is
    malicious (the evidence onset). Missed sessions are excluded from
    ``median_detection_delay_commands`` and represented in the all-session
    censored median.
    """

    targets, scores = _validate_binary_inputs(labels, probabilities)
    if len(session_ids) != len(targets) or len(prefix_indices) != len(targets):
        raise ValueError("session_ids and prefix_indices must align with labels")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for session_id, prefix_index, target, score in zip(
        session_ids, prefix_indices, targets, scores, strict=True
    ):
        grouped[str(session_id)].append((int(prefix_index), float(target), float(score)))

    delays: list[int] = []
    censored_delays: list[int] = []
    alert_commands: list[int] = []
    missed_sessions = 0
    false_positive_sessions = 0
    malicious_sessions = 0
    for session_id in sorted(grouped):
        observations = sorted(grouped[session_id])
        positive_indices = [index for index, target, _ in observations if target > 0]
        if not positive_indices:
            if any(score >= threshold for _, _, score in observations):
                false_positive_sessions += 1
            continue
        malicious_sessions += 1
        onset = positive_indices[0]
        alert_index = next(
            (index for index, _, score in observations if index >= onset and score >= threshold),
            None,
        )
        final_index = max(index for index, _, _ in observations)
        if alert_index is None:
            missed_sessions += 1
            censored_delays.append(max(0, final_index - onset))
            continue
        delay = max(0, alert_index - onset)
        delays.append(delay)
        censored_delays.append(delay)
        alert_commands.append(alert_index + 1)

    def percentile(values: list[int], quantile: float) -> float | None:
        if not values:
            return None
        return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))

    detected = len(delays)
    return {
        "malicious_sessions": malicious_sessions,
        "detected_sessions": detected,
        "missed_sessions": missed_sessions,
        "false_positive_sessions": false_positive_sessions,
        "detection_rate": detected / malicious_sessions if malicious_sessions else 0.0,
        "median_detection_delay_commands": percentile(delays, 0.5),
        "p90_detection_delay_commands": percentile(delays, 0.9),
        "max_detection_delay_commands": max(delays) if delays else None,
        "median_censored_detection_delay_commands": percentile(censored_delays, 0.5),
        "median_alert_command_number": percentile(alert_commands, 0.5),
        "p90_alert_command_number": percentile(alert_commands, 0.9),
        "detection_delays": delays,
        "alert_command_numbers": alert_commands,
    }


def _normalize_support_floors(support_floors: Sequence[int]) -> tuple[int, ...]:
    """Validate and de-duplicate support floors while preserving order."""

    floors: list[int] = []
    for raw_floor in support_floors:
        if isinstance(raw_floor, bool) or not isinstance(raw_floor, (int, np.integer)):
            raise ValueError("support_floors must contain positive integers")
        floor = int(raw_floor)
        if floor < 1:
            raise ValueError("support_floors must contain positive integers")
        if floor not in floors:
            floors.append(floor)
    return tuple(floors)


def _mean_for_support(
    values: np.ndarray,
    support: np.ndarray,
    *,
    floor: int,
) -> float | None:
    """Return the mean over heads meeting ``support >= floor``, else ``None``."""

    eligible = support >= floor
    if not np.any(eligible):
        return None
    return float(np.mean(values[eligible]))


def multilabel_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Sequence[float] | float,
    support_floors: Sequence[int] = (1, 10, 50),
) -> dict[str, float | int | None]:
    targets = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != scores.shape or targets.ndim != 2:
        raise ValueError("labels and probabilities must be equal 2-D matrices")
    normalized_floors = _normalize_support_floors(support_floors)
    if targets.shape[1] == 0:
        result: dict[str, float | int | None] = {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_f1_all": 0.0,
            "supported_macro_precision": None,
            "supported_macro_recall": None,
            "supported_macro_f1": None,
            "supported_heads": 0,
            "total_heads": 0,
            "unsupported_heads": 0,
        }
        for floor in normalized_floors:
            result[f"macro_precision_support_ge_{floor}"] = None
            result[f"macro_recall_support_ge_{floor}"] = None
            result[f"macro_f1_support_ge_{floor}"] = None
            result[f"heads_support_ge_{floor}"] = 0
        return result
    if np.isscalar(thresholds):
        threshold_array = np.full(targets.shape[1], float(thresholds))
    else:
        threshold_array = np.asarray(thresholds, dtype=np.float64)
        if threshold_array.shape != (targets.shape[1],):
            raise ValueError("threshold count does not match label count")
    predictions = scores >= threshold_array
    true_positive = np.count_nonzero(predictions & (targets > 0), axis=0)
    false_positive = np.count_nonzero(predictions & (targets <= 0), axis=0)
    false_negative = np.count_nonzero(~predictions & (targets > 0), axis=0)
    support = true_positive + false_negative
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    floor_one = support >= 1
    supported_count = int(np.count_nonzero(floor_one))
    supported_macro_precision = _mean_for_support(precision, support, floor=1)
    supported_macro_recall = _mean_for_support(recall, support, floor=1)
    supported_macro_f1 = _mean_for_support(f1, support, floor=1)
    macro_f1_all = float(np.mean(f1))
    result = {
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": macro_f1_all,
        "macro_f1_all": macro_f1_all,
        "supported_macro_precision": supported_macro_precision,
        "supported_macro_recall": supported_macro_recall,
        "supported_macro_f1": supported_macro_f1,
        "supported_heads": supported_count,
        "total_heads": int(targets.shape[1]),
        "unsupported_heads": int(targets.shape[1]) - supported_count,
    }
    for floor in normalized_floors:
        eligible = support >= floor
        result[f"macro_precision_support_ge_{floor}"] = _mean_for_support(
            precision, support, floor=floor
        )
        result[f"macro_recall_support_ge_{floor}"] = _mean_for_support(recall, support, floor=floor)
        result[f"macro_f1_support_ge_{floor}"] = _mean_for_support(f1, support, floor=floor)
        result[f"heads_support_ge_{floor}"] = int(np.count_nonzero(eligible))
    return result


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _finite_metric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_metric(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _metric_checks(
    malicious_metrics: dict[str, Any],
    detection_delay: dict[str, Any],
    *,
    tactic_macro_f1: Any,
    technique_macro_f1: Any,
    targets: dict[str, float],
) -> tuple[dict[str, bool], list[str], dict[str, float | None]]:
    """Evaluate the point-estimate gates, preserving missing-value failures."""

    values: dict[str, float | None] = {
        "recall": _finite_metric(malicious_metrics.get("recall")),
        "fpr": _finite_metric(malicious_metrics.get("fpr")),
        "pr_auc": _finite_metric(malicious_metrics.get("pr_auc")),
        "roc_auc": _finite_metric(malicious_metrics.get("roc_auc")),
        "brier": _finite_metric(malicious_metrics.get("brier")),
        "ece": _finite_metric(malicious_metrics.get("ece")),
        "tactic_macro_f1": _finite_metric(tactic_macro_f1),
        "technique_macro_f1": _finite_metric(technique_macro_f1),
        "detection_rate": _finite_metric(detection_delay.get("detection_rate")),
        # Missed sessions must remain visible in the gate.  The censored median
        # assigns a miss the maximum observed delay for that session instead of
        # silently dropping it from an optimistic detected-only median.
        "detection_delay_commands": _finite_metric(
            detection_delay.get("median_censored_detection_delay_commands")
        ),
    }
    checks: dict[str, bool] = {}
    missing: list[str] = []
    for name, value in values.items():
        target = float(targets[name])
        if value is None:
            checks[name] = False
            missing.append(name)
        elif name in _LOWER_IS_BETTER:
            checks[name] = value <= target
        else:
            checks[name] = value >= target
    return checks, missing, values


def _proxy_audit_gate(
    report: Any,
    *,
    minimum_resolved_cases: int,
) -> tuple[bool, dict[str, Any]]:
    """Return a complete, fail-closed summary of the proxy-audit report."""

    payload = _mapping(report)
    if not payload:
        return False, {
            "available": False,
            "eligible_for_promotion": False,
            "resolved_cases": 0,
            "unresolved": None,
            "minimum_resolved_cases": minimum_resolved_cases,
            "blockers": ["proxy_audit_missing"],
        }

    prediction_metrics = _mapping(payload.get("prediction_metrics"))
    blockers = [str(item) for item in payload.get("promotion_blockers", ())]
    eligible = payload.get("eligible_for_promotion") is True
    resolved_cases_raw = payload.get("resolved_cases")
    resolved_cases = (
        int(resolved_cases_raw)
        if isinstance(resolved_cases_raw, (int, float)) and math.isfinite(resolved_cases_raw)
        else None
    )
    unresolved_raw = payload.get("unresolved")
    unresolved = (
        int(unresolved_raw)
        if isinstance(unresolved_raw, (int, float)) and math.isfinite(unresolved_raw)
        else None
    )
    scored_cases_raw = prediction_metrics.get("scored_cases")
    scored_cases = (
        int(scored_cases_raw)
        if isinstance(scored_cases_raw, (int, float)) and math.isfinite(scored_cases_raw)
        else None
    )
    unpredicted_raw = prediction_metrics.get("unpredicted_cases")
    unpredicted_cases = (
        int(unpredicted_raw)
        if isinstance(unpredicted_raw, (int, float)) and math.isfinite(unpredicted_raw)
        else None
    )

    complete_predictions = (
        resolved_cases is not None and scored_cases == resolved_cases and unpredicted_cases == 0
    )
    count_ok = resolved_cases is not None and resolved_cases >= minimum_resolved_cases
    unresolved_ok = unresolved == 0
    passed = bool(eligible and count_ok and unresolved_ok and complete_predictions)
    if not eligible and not blockers:
        blockers.append("proxy_audit_not_eligible")
    if not count_ok:
        blockers.append(
            "resolved_sessions_below_minimum:"
            f"{resolved_cases if resolved_cases is not None else 'missing'}"
            f"<{minimum_resolved_cases}"
        )
    if not unresolved_ok:
        blockers.append(
            f"unresolved_sessions_present:{unresolved if unresolved is not None else 'missing'}"
        )
    if not complete_predictions:
        blockers.append("prediction_labels_incomplete")
    return passed, {
        "available": True,
        "eligible_for_promotion": eligible,
        "resolved_cases": resolved_cases,
        "unresolved": unresolved,
        "scored_cases": scored_cases,
        "unpredicted_cases": unpredicted_cases,
        "minimum_resolved_cases": minimum_resolved_cases,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _tactic_label_coverage_gate(
    support: Sequence[int] | Mapping[str, int] | None,
    *,
    tactic_ids: Sequence[str] | None,
    minimum_covered_tactics: int,
    support_floor: int = TACTIC_SUPPORT_FLOOR,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate whether the evaluation set exercises enough tactic heads."""

    if support is None:
        return False, {
            "observed": None,
            "required": minimum_covered_tactics,
            "support_floor": support_floor,
            "covered_tactics": [],
            "uncovered_tactics": [],
            "blockers": ["tactic_label_support_missing"],
        }

    blockers: list[str] = []
    if isinstance(support, Mapping):
        named_support = [(str(name), value) for name, value in support.items()]
    else:
        values = list(support)
        if tactic_ids is not None and len(tactic_ids) != len(values):
            blockers.append("tactic_label_support_length_mismatch")
        names = (
            [str(name) for name in tactic_ids]
            if tactic_ids is not None
            else [f"tactic_{index}" for index in range(len(values))]
        )
        named_support = list(zip(names, values, strict=False))

    support_counts: dict[str, int] = {}
    for name, raw_count in named_support:
        try:
            numeric_count = float(raw_count)
        except (TypeError, ValueError):
            blockers.append(f"invalid_tactic_support:{name}")
            continue
        if not math.isfinite(numeric_count) or numeric_count < 0:
            blockers.append(f"invalid_tactic_support:{name}")
            continue
        if not numeric_count.is_integer():
            blockers.append(f"invalid_tactic_support:{name}")
            continue
        support_counts[name] = int(numeric_count)

    covered = [name for name, count in support_counts.items() if count >= support_floor]
    uncovered = [name for name, count in support_counts.items() if count < support_floor]
    observed = len(covered)
    if observed < minimum_covered_tactics:
        blockers.append(f"covered_tactics_below_minimum:{observed}<{minimum_covered_tactics}")
    return not blockers, {
        "observed": observed,
        "required": minimum_covered_tactics,
        "support_floor": support_floor,
        "covered_tactics": covered,
        "uncovered_tactics": uncovered,
        "support_counts": support_counts,
        "blockers": blockers,
    }


def promotion_report(
    malicious_metrics: dict[str, Any],
    detection_delay: dict[str, Any],
    *,
    tactic_macro_f1: Any,
    technique_macro_f1: Any,
    tactic_macro_f1_all_heads: Any = None,
    tactic_macro_f1_support_ge_1: Any = None,
    tactic_macro_f1_support_ge_50: Any = None,
    tactic_heads_support_ge_50: Any = None,
    technique_macro_f1_all_heads: Any = None,
    technique_macro_f1_support_ge_1: Any = None,
    technique_macro_f1_support_ge_50: Any = None,
    technique_heads_support_ge_50: Any = None,
    targets: dict[str, float] | None = None,
    proxy_audit_report: Any = None,
    require_proxy_audit: bool = True,
    tactic_label_support: Sequence[int] | Mapping[str, int] | None = None,
    tactic_ids: Sequence[str] | None = None,
    require_tactic_label_coverage: bool = False,
) -> dict[str, Any]:
    """Evaluate all acceptance targets without allowing missing values to pass."""

    selected_targets = dict(PROMOTION_TARGETS)
    selected_targets.update(targets or {})
    checks, missing_metrics, values = _metric_checks(
        malicious_metrics,
        detection_delay,
        tactic_macro_f1=tactic_macro_f1,
        technique_macro_f1=technique_macro_f1,
        targets=selected_targets,
    )
    tactic_label_coverage: dict[str, Any] = {"required": False}
    if require_tactic_label_coverage or tactic_label_support is not None:
        coverage_passed, tactic_label_coverage = _tactic_label_coverage_gate(
            tactic_label_support,
            tactic_ids=tactic_ids,
            minimum_covered_tactics=int(selected_targets["tactic_label_coverage"]),
            support_floor=TACTIC_SUPPORT_FLOOR,
        )
        tactic_label_coverage["gate_required"] = require_tactic_label_coverage
        checks["tactic_label_coverage"] = coverage_passed
        if tactic_label_support is None:
            missing_metrics.append("tactic_label_coverage")
    proxy_audit_passed = True
    proxy_audit: dict[str, Any] = {"required": False}
    if require_proxy_audit:
        proxy_audit_passed, proxy_audit = _proxy_audit_gate(
            proxy_audit_report,
            minimum_resolved_cases=int(selected_targets["minimum_resolved_audit_cases"]),
        )
        proxy_audit["required"] = True
        checks["proxy_audit"] = proxy_audit_passed

    violations = [name for name, passed in checks.items() if not passed]
    tactic_floor_50 = _finite_metric(tactic_macro_f1_support_ge_50)
    if tactic_floor_50 is None:
        tactic_floor_50 = values["tactic_macro_f1"]
    technique_floor_50 = _finite_metric(technique_macro_f1_support_ge_50)
    if technique_floor_50 is None:
        technique_floor_50 = values["technique_macro_f1"]
    return {
        "targets": selected_targets,
        "metric_values": values,
        "checks": checks,
        "missing_metrics": missing_metrics,
        "violations": violations,
        "proxy_audit": proxy_audit,
        "tactic_macro_f1": values["tactic_macro_f1"],
        "technique_macro_f1": values["technique_macro_f1"],
        "tactic_macro_f1_all_heads": _finite_metric(tactic_macro_f1_all_heads),
        "tactic_macro_f1_support_ge_1": _finite_metric(tactic_macro_f1_support_ge_1),
        "tactic_macro_f1_support_ge_50": tactic_floor_50,
        "tactic_heads_support_ge_50": _finite_int(tactic_heads_support_ge_50),
        "technique_macro_f1_all_heads": _finite_metric(technique_macro_f1_all_heads),
        "technique_macro_f1_support_ge_1": _finite_metric(technique_macro_f1_support_ge_1),
        "technique_macro_f1_support_ge_50": technique_floor_50,
        "technique_heads_support_ge_50": _finite_int(technique_heads_support_ge_50),
        "tactic_label_coverage": tactic_label_coverage,
        "passed": bool(all(checks.values())),
    }
