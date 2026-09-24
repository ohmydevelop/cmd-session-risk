from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dev_operating_point_screen.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dev_operating_point_screen", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = _load_module()


def test_family_split_is_deterministic_and_family_disjoint() -> None:
    family_ids = [f"family:{index}" for index in range(40) for _ in range(3)]
    calibration, evaluation = SCREEN.family_split_masks(family_ids, fraction=0.5, seed=42)
    assert calibration and evaluation
    assert len(calibration) + len(evaluation) == len(family_ids)
    calibration_families = {family_ids[index] for index in calibration}
    evaluation_families = {family_ids[index] for index in evaluation}
    assert not calibration_families & evaluation_families

    repeat = SCREEN.family_split_masks(family_ids, fraction=0.5, seed=42)
    assert repeat == (calibration, evaluation)

    # A different seed must repartition families, otherwise the "holdout" half
    # would be a constant subset of dev rather than a reproducible rotation.
    other = SCREEN.family_split_masks(family_ids, fraction=0.5, seed=7)
    assert {family_ids[index] for index in other[0]} != calibration_families


def test_family_split_matches_training_partition() -> None:
    from cmdrisk.training.data import PrefixExample
    from cmdrisk.training.pipeline import _stable_family_partition

    examples = [
        PrefixExample(
            session_id=f"session:{family}",
            family_id=f"family:{family}",
            source_kind="real_malicious",
            split="dev",
            command_count=4,
            prefix_index=0,
            commands=("id",),
            dts=(None,),
            malicious=1.0,
            tactics=(0.0,),
            techniques=(0.0,),
        )
        for family in range(17)
    ]
    _, calibration_examples = _stable_family_partition(examples, fraction=0.5, seed=42)
    expected = {example.family_id for example in calibration_examples}
    calibration, _ = SCREEN.family_split_masks(
        [example.family_id for example in examples], fraction=0.5, seed=42
    )
    assert {examples[index].family_id for index in calibration} == expected


def test_family_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        SCREEN.family_split_masks(["a"], fraction=1.5, seed=0)


def test_parse_candidates_requires_label_and_path() -> None:
    assert SCREEN.parse_candidates(["final=/tmp/x"]) == [("final", Path("/tmp/x"))]
    with pytest.raises(SystemExit):
        SCREEN.parse_candidates(["/tmp/x"])
    with pytest.raises(SystemExit):
        SCREEN.parse_candidates(["final=/tmp/x", "final=/tmp/y"])
    with pytest.raises(SystemExit):
        SCREEN.parse_candidates([])


def _results(multiplier: float) -> dict:
    row = {
        "max_fpr": 0.01,
        "prefix_recall": 0.90 * multiplier,
        "prefix_fpr": 0.010 * multiplier,
        "session_recall": 0.85 * multiplier,
        "session_fpr": 0.004 * multiplier,
    }
    return {"final": {"rows": [row]}, "soup": {"rows": [{**row, "max_fpr": 0.01}]}}


def test_verdicts_accept_a_pareto_improvement_and_reject_a_regression() -> None:
    improved = _results(1.02)
    improved["final"]["rows"][0]["prefix_recall"] = 0.90
    improved["soup"]["rows"][0]["prefix_recall"] = 0.91
    verdict = SCREEN._verdicts(improved, baseline="final", recall_tolerance=0.0)
    assert verdict["soup"]["dominates_all_budgets"] is True
    assert verdict["soup"]["dominates_budgets"] == 1

    worse = _results(1.0)
    worse["final"]["rows"][0]["prefix_recall"] = 0.90
    worse["soup"]["rows"][0]["prefix_recall"] = 0.89
    worse["soup"]["rows"][0]["prefix_fpr"] = 0.011
    verdict = SCREEN._verdicts(worse, baseline="final", recall_tolerance=0.0)
    assert verdict["soup"]["dominates_all_budgets"] is False
    assert verdict["soup"]["regresses_budgets"] == 1
    entry = verdict["soup"]["per_budget"][0]
    assert entry["prefix_recall_delta"] == pytest.approx(-0.01)
    assert entry["prefix_fpr_delta"] == pytest.approx(0.001)

    # A tolerance lets a candidate give back a little recall in exchange for FPR.
    lenient = SCREEN._verdicts(worse, baseline="final", recall_tolerance=0.02)
    assert lenient["soup"]["dominates_all_budgets"] is False  # FPR still worse
    both_worse = _results(1.0)
    both_worse["final"]["rows"][0]["prefix_recall"] = 0.90
    both_worse["soup"]["rows"][0]["prefix_recall"] = 0.89
    both_worse["soup"]["rows"][0]["prefix_fpr"] = 0.005
    both_worse["soup"]["rows"][0]["session_fpr"] = 0.002
    both_worse["soup"]["rows"][0]["session_recall"] = 0.845
    lenient = SCREEN._verdicts(both_worse, baseline="final", recall_tolerance=0.02)
    assert lenient["soup"]["dominates_all_budgets"] is True


def test_verdicts_without_baseline_is_empty() -> None:
    assert SCREEN._verdicts(_results(1.0), baseline=None, recall_tolerance=0.0) == {}
