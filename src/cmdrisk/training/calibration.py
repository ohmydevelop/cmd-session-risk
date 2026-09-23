"""Temperature scaling and FPR-constrained threshold calibration."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sigmoid(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _binary_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = np.clip(logits / max(float(temperature), 1e-6), -80.0, 80.0)
    probabilities = sigmoid(scaled)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
    )


def fit_temperature(
    logits: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.size != targets.size:
        raise ValueError("logits and labels must have the same length")
    if values.size == 0:
        return 1.0
    if np.unique(targets).size < 2:
        return 1.0

    try:  # scipy is part of the train extra and gives a smooth bounded optimum.
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            lambda temperature: _binary_nll(values, targets, temperature),
            bounds=(0.05, 20.0),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 200},
        )
        if result.success and np.isfinite(result.x):
            return float(result.x)
    except ImportError:  # pragma: no cover - only hit in a minimal installation
        pass

    grid = np.geomspace(0.05, 20.0, 101)
    losses = [_binary_nll(values, targets, value) for value in grid]
    return float(grid[int(np.argmin(losses))])


def find_threshold_at_fpr(
    labels: Sequence[int] | Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    max_fpr: float = 0.01,
) -> float:
    """Return the lowest inclusive threshold whose empirical FPR is allowed."""

    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.size != values.size or targets.size == 0:
        raise ValueError("labels and scores must have the same non-zero length")
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    negatives = values[targets <= 0]
    if negatives.size == 0:
        return 1.0

    candidates = sorted({float(value) for value in values}, reverse=True)
    # Ascending through candidate thresholds finds the smallest threshold that
    # satisfies the FPR budget because false positives decrease monotonically.
    for threshold in sorted(candidates):
        false_positives = int(np.count_nonzero(negatives >= threshold))
        if false_positives / negatives.size <= max_fpr:
            return float(threshold)

    highest = float(np.max(values))
    return float(min(1.0, np.nextafter(highest, np.inf)))


@dataclass(frozen=True)
class Calibration:
    malicious_temperature: float
    malicious_threshold: float
    tactic_temperature: float
    technique_temperature: float
    tactic_thresholds: dict[str, float]
    technique_thresholds: dict[str, float]
    target_fpr: float = 0.01
    version: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("malicious_temperature", self.malicious_temperature),
            ("tactic_temperature", self.tactic_temperature),
            ("technique_temperature", self.technique_temperature),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, value in (
            ("malicious_threshold", self.malicious_threshold),
            *((f"tactic_thresholds.{key}", item) for key, item in self.tactic_thresholds.items()),
            *(
                (f"technique_thresholds.{key}", item)
                for key, item in self.technique_thresholds.items()
            ),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

    def malicious_probabilities(self, logits: Sequence[float] | np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.malicious_temperature)

    def tactic_probabilities(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.tactic_temperature)

    def technique_probabilities(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.technique_temperature)

    def predict_malicious(self, logits: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.malicious_probabilities(logits) >= self.malicious_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target_fpr": self.target_fpr,
            "malicious_temperature": self.malicious_temperature,
            "malicious_threshold": self.malicious_threshold,
            "tactic_temperature": self.tactic_temperature,
            "technique_temperature": self.technique_temperature,
            "tactic_thresholds": dict(sorted(self.tactic_thresholds.items())),
            "technique_thresholds": dict(sorted(self.technique_thresholds.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Calibration:
        return cls(
            malicious_temperature=float(value["malicious_temperature"]),
            malicious_threshold=float(value["malicious_threshold"]),
            tactic_temperature=float(value.get("tactic_temperature", 1.0)),
            technique_temperature=float(value.get("technique_temperature", 1.0)),
            tactic_thresholds={
                str(key): float(item)
                for key, item in value.get("tactic_thresholds", {}).items()
            },
            technique_thresholds={
                str(key): float(item)
                for key, item in value.get("technique_thresholds", {}).items()
            },
            target_fpr=float(value.get("target_fpr", 0.01)),
            version=int(value.get("version", 1)),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_calibration(
    *,
    malicious_logits: Sequence[float] | np.ndarray,
    malicious_labels: Sequence[int] | Sequence[float] | np.ndarray,
    tactic_logits: np.ndarray,
    tactic_labels: np.ndarray,
    tactic_ids: Sequence[str],
    technique_logits: np.ndarray,
    technique_labels: np.ndarray,
    technique_ids: Sequence[str],
    max_fpr: float = 0.01,
) -> Calibration:
    malicious_temperature = fit_temperature(malicious_logits, malicious_labels)
    tactic_temperature = fit_temperature(tactic_logits, tactic_labels)
    technique_temperature = fit_temperature(technique_logits, technique_labels)

    tactic_probabilities = sigmoid(tactic_logits / tactic_temperature)
    technique_probabilities = sigmoid(technique_logits / technique_temperature)
    if tactic_probabilities.shape[1] != len(tactic_ids):
        raise ValueError("tactic label matrix does not match tactic IDs")
    if technique_probabilities.shape[1] != len(technique_ids):
        raise ValueError("technique label matrix does not match technique IDs")

    malicious_probabilities = sigmoid(
        np.asarray(malicious_logits, dtype=np.float64) / malicious_temperature
    )
    malicious_threshold = find_threshold_at_fpr(
        malicious_labels, malicious_probabilities, max_fpr=max_fpr
    )
    tactic_thresholds = {
        label: find_threshold_at_fpr(
            tactic_labels[:, index], tactic_probabilities[:, index], max_fpr=max_fpr
        )
        for index, label in enumerate(tactic_ids)
    }
    technique_thresholds = {
        label: find_threshold_at_fpr(
            technique_labels[:, index],
            technique_probabilities[:, index],
            max_fpr=max_fpr,
        )
        for index, label in enumerate(technique_ids)
    }
    return Calibration(
        malicious_temperature=malicious_temperature,
        malicious_threshold=malicious_threshold,
        tactic_temperature=tactic_temperature,
        technique_temperature=technique_temperature,
        tactic_thresholds=tactic_thresholds,
        technique_thresholds=technique_thresholds,
        target_fpr=max_fpr,
    )


def calibration_as_training_features(
    logits: Iterable[float], labels: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Small public helper used by tests and external calibration scripts."""

    return np.asarray(list(logits), dtype=np.float64), np.asarray(list(labels), dtype=np.float64)
