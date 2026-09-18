"""ONNX Runtime-backed scoring for Unix command prefixes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cmdrisk.schema import ScoreEntry, SequenceScore, TacticScore, TechniqueScore
from cmdrisk.taxonomy import (
    TACTIC_BY_ID,
    TECHNIQUE_NAMES,
    attack_stage,
    stage_severity,
)
from cmdrisk.tokenization import (
    DEFAULT_COMMAND_MAX_TOKENS,
    DEFAULT_MAX_LENGTH,
    CommandTokenizer,
)


class ScoringError(RuntimeError):
    """Base exception for runtime scoring failures."""


class ArtifactLoadError(ScoringError):
    """Raised when an ONNX scoring artifact cannot be loaded."""


class InferenceError(ScoringError):
    """Raised when an ONNX inference call fails."""


@dataclass(frozen=True)
class _Label:
    id: str
    name: str


def dt_bucket(dt: float | None) -> str:
    """Return the stable time bucket token used during training and inference."""

    if dt is None:
        return "<DT_UNKNOWN>"
    if not isinstance(dt, (int, float)) or isinstance(dt, bool):
        raise ValueError("dt must be a non-negative number or None")
    value = float(dt)
    if not math.isfinite(value) or value < 0:
        raise ValueError("dt must be a finite, non-negative number or None")
    if value == 0:
        return "<DT_0S>"
    if value <= 1:
        return "<DT_LE_1S>"
    if value <= 10:
        return "<DT_LE_10S>"
    if value <= 60:
        return "<DT_LE_1M>"
    if value <= 600:
        return "<DT_LE_10M>"
    return "<DT_GT_10M>"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactLoadError(f"required artifact file not found: {path.name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(f"cannot load artifact file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactLoadError(f"artifact file {path.name} must contain a JSON object")
    return value


def _first_existing(artifact_dir: Path, names: Sequence[str], *, required: bool) -> Path | None:
    for name in names:
        candidate = artifact_dir / name
        if candidate.is_file():
            return candidate
    if required:
        choices = ", ".join(names)
        raise ArtifactLoadError(f"missing artifact file (expected one of: {choices})")
    return None


def _normalise_labels(value: Any, names: Mapping[str, str]) -> list[_Label]:
    labels: list[_Label] = []
    if value is None:
        return labels
    if not isinstance(value, list):
        raise ArtifactLoadError("label vocab entries must be a JSON list")

    for item in value:
        if isinstance(item, str):
            label_id = item
            label_name = names.get(item, item)
        elif isinstance(item, dict):
            label_id = str(item.get("id", "")).strip()
            label_name = str(item.get("name") or names.get(label_id) or label_id).strip()
        else:
            raise ArtifactLoadError("label vocab entries must be strings or objects")
        if not label_id:
            raise ArtifactLoadError("label vocab contains an empty id")
        labels.append(_Label(id=label_id, name=label_name))

    ids = [item.id for item in labels]
    if len(ids) != len(set(ids)):
        raise ArtifactLoadError("label vocab contains duplicate ids")
    return labels


def _probability(value: Any, *, name: str, default: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactLoadError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ArtifactLoadError(f"{name} must be finite and between 0 and 1")
    return result


def _threshold_map(value: Any) -> tuple[float, dict[str, float]]:
    """Accept both runtime envelopes and the training calibration mapping.

    Training writes a flat ``{label_id: threshold}`` mapping.  Older runtime
    artifacts use ``{"default": ..., "per_label": {...}}``.  Supporting both
    keeps exported artifacts portable without a conversion step.
    """

    if value is None:
        return 0.5, {}
    if isinstance(value, bool):
        raise ArtifactLoadError("label thresholds must be a number or object")
    if isinstance(value, (int, float)):
        return _probability(value, name="label threshold", default=0.5), {}
    if not isinstance(value, Mapping):
        raise ArtifactLoadError("label thresholds must be a number or object")

    if any(key in value for key in ("default", "threshold", "per_label", "thresholds")):
        default = _probability(
            value.get("default", value.get("threshold")),
            name="label threshold",
            default=0.5,
        )
        per_label = value.get("per_label", value.get("thresholds", {}))
    else:
        default = 0.5
        per_label = value
    if not isinstance(per_label, Mapping):
        raise ArtifactLoadError("per-label thresholds must be an object")
    return default, {
        str(key): _probability(score, name=f"threshold for {key}", default=0.5)
        for key, score in per_label.items()
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class _TokenizerAdapter:
    """Normalise lightweight test tokenizers to the HF ``encode`` contract."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __len__(self) -> int:
        try:
            return len(self._tokenizer)
        except TypeError:
            vocab_size = getattr(self._tokenizer, "vocab_size", None)
            if vocab_size is None:
                raise
            return int(vocab_size)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        encode = getattr(self._tokenizer, "encode", None)
        if callable(encode):
            values = encode(text, add_special_tokens=add_special_tokens)
        else:
            try:
                values = self._tokenizer(text, add_special_tokens=add_special_tokens)
            except TypeError:
                values = self._tokenizer(text)
        if isinstance(values, Mapping):
            values = values.get("input_ids", ())
        elif hasattr(values, "input_ids"):
            values = values.input_ids
        array = _to_numpy(values)
        if array.ndim > 1:
            array = array.reshape(-1)
        return [int(value) for value in array.reshape(-1)]


class SessionScorer:
    """Load a self-contained ONNX artifact and score command prefixes."""

    def __init__(
        self,
        onnx_session: Any,
        tokenizer: Any,
        *,
        tactics: Sequence[_Label] = (),
        techniques: Sequence[_Label] = (),
        malicious_threshold: float = 0.5,
        tactic_threshold: float = 0.5,
        technique_threshold: float = 0.5,
        tactic_thresholds: Mapping[str, float] | None = None,
        technique_thresholds: Mapping[str, float] | None = None,
        temperature: float = 1.0,
        malicious_temperature: float | None = None,
        tactic_temperature: float | None = None,
        technique_temperature: float | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        command_max_tokens: int = DEFAULT_COMMAND_MAX_TOKENS,
        session_head_tokens: int | None = None,
        session_tail_tokens: int | None = None,
        input_names: Sequence[str] | None = None,
        output_names: Mapping[str, str] | None = None,
        model_version: str = "unknown",
    ) -> None:
        malicious_threshold = _probability(
            malicious_threshold, name="malicious threshold", default=0.5
        )
        tactic_threshold = _probability(
            tactic_threshold, name="tactic threshold", default=0.5
        )
        technique_threshold = _probability(
            technique_threshold, name="technique threshold", default=0.5
        )
        temperatures = {
            "malicious temperature": (
                temperature if malicious_temperature is None else malicious_temperature
            ),
            "tactic temperature": (
                temperature if tactic_temperature is None else tactic_temperature
            ),
            "technique temperature": (
                temperature if technique_temperature is None else technique_temperature
            ),
        }
        for name, value in temperatures.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArtifactLoadError(f"{name} must be a number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ArtifactLoadError(f"{name} must be positive and finite")
        if max_length < 1:
            raise ArtifactLoadError("max_length must be greater than zero")
        if command_max_tokens < 1:
            raise ArtifactLoadError("command_max_tokens must be greater than zero")

        if not isinstance(tokenizer, CommandTokenizer):
            tokenizer = CommandTokenizer(
                _TokenizerAdapter(tokenizer),
                max_length=int(max_length),
                command_max_tokens=int(command_max_tokens),
                session_head_tokens=session_head_tokens,
                session_tail_tokens=session_tail_tokens,
            )

        self._onnx_session = onnx_session
        self._tokenizer = tokenizer
        self._tactics = tuple(tactics)
        self._techniques = tuple(techniques)
        self.malicious_threshold = float(malicious_threshold)
        self._tactic_threshold = float(tactic_threshold)
        self._technique_threshold = float(technique_threshold)
        self._tactic_thresholds = {
            str(key): _probability(value, name=f"threshold for {key}", default=0.5)
            for key, value in (tactic_thresholds or {}).items()
        }
        self._technique_thresholds = {
            str(key): _probability(value, name=f"threshold for {key}", default=0.5)
            for key, value in (technique_thresholds or {}).items()
        }
        self.temperature = float(temperatures["malicious temperature"])
        self.malicious_temperature = self.temperature
        self.tactic_temperature = float(temperatures["tactic temperature"])
        self.technique_temperature = float(temperatures["technique temperature"])
        self.max_length = tokenizer.max_length
        self.model_version = model_version
        self._output_names = dict(output_names or {})
        self._input_names = tuple(input_names or ())
        self._sessions: dict[str, SessionState] = {}

    @classmethod
    def from_onnx(
        cls,
        artifact_dir: str | Path,
        *,
        onnx_session: Any | None = None,
        tokenizer: Any | None = None,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> SessionScorer:
        """Load ``model.onnx`` and its adjacent tokenizer and metadata files."""

        root = Path(artifact_dir).expanduser().resolve()
        if not root.is_dir():
            raise ArtifactLoadError(f"artifact directory does not exist: {root}")

        runtime_path = _first_existing(
            root, ("runtime_config.json", "runtime.json", "config.json"), required=False
        )
        runtime_config = _read_json(runtime_path) if runtime_path else {}
        label_path = _first_existing(
            root, ("label_vocab.json", "labels.json"), required=False
        )
        label_config = _read_json(label_path) if label_path else runtime_config
        calibration_path = _first_existing(
            root, ("calibration.json", "thresholds.json"), required=False
        )
        calibration = _read_json(calibration_path) if calibration_path else {}
        tokenization_path = root / "command_tokenization.json"
        tokenization_config = _read_json(tokenization_path) if tokenization_path.is_file() else {}

        tactic_value = label_config.get("tactics", label_config.get("tactic_labels"))
        technique_value = label_config.get("techniques", label_config.get("technique_labels"))
        tactic_names = {item.id: item.name for item in TACTIC_BY_ID.values()}
        tactics = _normalise_labels(tactic_value, tactic_names)
        techniques = _normalise_labels(technique_value, TECHNIQUE_NAMES)

        legacy_temperature = calibration.get("temperature", 1.0)
        malicious_section = calibration.get("malicious")
        if isinstance(malicious_section, Mapping):
            malicious_default = malicious_section.get(
                "threshold", calibration.get("malicious_threshold", 0.5)
            )
            malicious_temperature = calibration.get(
                "malicious_temperature",
                malicious_section.get("temperature", legacy_temperature),
            )
        else:
            malicious_default = (
                calibration.get("malicious_threshold", 0.5)
                if malicious_section is None
                else malicious_section
            )
            malicious_temperature = calibration.get(
                "malicious_temperature", legacy_temperature
            )
        tactic_temperature = calibration.get("tactic_temperature", legacy_temperature)
        technique_temperature = calibration.get(
            "technique_temperature", legacy_temperature
        )

        tactic_threshold_value = calibration.get("tactics")
        if tactic_threshold_value is None:
            tactic_threshold_value = calibration.get("tactic_thresholds")
        tactic_default, tactic_thresholds = _threshold_map(tactic_threshold_value)

        technique_threshold_value = calibration.get("techniques")
        if technique_threshold_value is None:
            technique_threshold_value = calibration.get("technique_thresholds")
        technique_default, technique_thresholds = _threshold_map(
            technique_threshold_value
        )

        output_names = runtime_config.get(
            "onnx_outputs", label_config.get("onnx_outputs", {})
        )
        if not isinstance(output_names, dict):
            raise ArtifactLoadError("onnx_outputs must be an object")
        output_names = {str(key): str(value) for key, value in output_names.items()}

        max_length = int(
            runtime_config.get(
                "max_length",
                label_config.get(
                    "max_length",
                    tokenization_config.get("max_length", DEFAULT_MAX_LENGTH),
                ),
            )
        )
        command_max_tokens = int(
            runtime_config.get(
                "command_max_tokens",
                label_config.get(
                    "command_max_tokens",
                    tokenization_config.get(
                        "command_max_tokens", DEFAULT_COMMAND_MAX_TOKENS
                    ),
                ),
            )
        )
        session_head_tokens = runtime_config.get(
            "session_head_tokens",
            label_config.get(
                "session_head_tokens", tokenization_config.get("session_head_tokens")
            ),
        )
        session_tail_tokens = runtime_config.get(
            "session_tail_tokens",
            label_config.get(
                "session_tail_tokens", tokenization_config.get("session_tail_tokens")
            ),
        )
        model_version = str(
            runtime_config.get(
                "model_version",
                label_config.get("model_version", calibration.get("model_version", "unknown")),
            )
        )

        if onnx_session is None:
            model_path = _first_existing(
                root,
                ("model.int8.onnx", "model_int8.onnx", "model.onnx"),
                required=True,
            )
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ArtifactLoadError(
                    "onnxruntime is required; install the 'train' extra"
                ) from exc
            options = ort.SessionOptions()
            if intra_op_num_threads is not None:
                options.intra_op_num_threads = int(intra_op_num_threads)
            if inter_op_num_threads is not None:
                options.inter_op_num_threads = int(inter_op_num_threads)
            try:
                onnx_session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=list(providers or ("CPUExecutionProvider",)),
                )
            except Exception as exc:
                raise ArtifactLoadError(f"cannot create ONNX Runtime session: {exc}") from exc

        if tokenizer is None:
            try:
                tokenizer = CommandTokenizer.from_pretrained(
                    root, use_fast=True, local_files_only=True
                )
            except ImportError as exc:
                raise ArtifactLoadError(
                    "transformers is required to load the artifact tokenizer; "
                    "install the 'train' extra or inject a tokenizer"
                ) from exc
            except Exception as exc:
                raise ArtifactLoadError(f"cannot load artifact tokenizer: {exc}") from exc

        input_names = runtime_config.get("onnx_inputs")
        if input_names is None:
            try:
                input_names = [item.name for item in onnx_session.get_inputs()]
            except (AttributeError, TypeError):
                input_names = ("input_ids", "attention_mask")

        return cls(
            onnx_session,
            tokenizer,
            tactics=tactics,
            techniques=techniques,
            malicious_threshold=malicious_default,
            tactic_threshold=tactic_default,
            technique_threshold=technique_default,
            tactic_thresholds=tactic_thresholds,
            technique_thresholds=technique_thresholds,
            malicious_temperature=malicious_temperature,
            tactic_temperature=tactic_temperature,
            technique_temperature=technique_temperature,
            max_length=max_length,
            command_max_tokens=command_max_tokens,
            session_head_tokens=(
                int(session_head_tokens) if session_head_tokens is not None else None
            ),
            session_tail_tokens=(
                int(session_tail_tokens) if session_tail_tokens is not None else None
            ),
            input_names=tuple(str(item) for item in input_names),
            output_names=output_names,
            model_version=model_version,
        )

    @property
    def tactic_labels(self) -> tuple[tuple[str, str], ...]:
        return tuple((item.id, item.name) for item in self._tactics)

    @property
    def technique_labels(self) -> tuple[tuple[str, str], ...]:
        return tuple((item.id, item.name) for item in self._techniques)

    def start_session(self, session_id: str) -> SessionState:
        """Start (or restart) a named online scoring session."""

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must contain non-whitespace characters")
        state = SessionState(scorer=self, session_id=session_id)
        self._sessions[session_id] = state
        return state

    def score_commands(
        self, commands: Sequence[tuple[str, float | None]] | Sequence[object]
    ) -> ScoreEntry:
        """Score one complete prefix; primarily useful for batch and benchmarks."""

        normalised: list[tuple[str, float | None]] = []
        for item in commands:
            if isinstance(item, tuple):
                if len(item) != 2:
                    raise ValueError("command tuples must contain (cmd, dt)")
                cmd, dt = item
            elif isinstance(item, Mapping):
                cmd, dt = item.get("cmd"), item.get("dt")
            else:
                cmd, dt = item.cmd, item.dt  # type: ignore[attr-defined]
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError("cmd must contain non-whitespace characters")
            dt_bucket(dt)
            normalised.append((cmd, dt))
        if not normalised:
            raise ValueError("at least one command is required")

        return self._infer_prefix(normalised)

    def _encode(self, commands: Sequence[tuple[str, float | None]]) -> dict[str, np.ndarray]:
        try:
            encoded = self._tokenizer.encode(
                [{"cmd": cmd, "dt": dt} for cmd, dt in commands]
            )
        except Exception as exc:
            raise InferenceError(f"tokenizer failed: {exc}") from exc

        if encoded is None or not getattr(encoded, "input_ids", None):
            raise InferenceError("tokenizer output does not contain input_ids")
        input_ids = np.asarray([encoded.input_ids], dtype=np.int64)
        attention_mask = np.asarray([encoded.attention_mask], dtype=np.int64)
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise InferenceError("tokenizer produced inconsistent input arrays")
        if input_ids.shape[1] > self.max_length:
            raise InferenceError("tokenizer did not apply its configured truncation")
        token_type_ids = np.zeros_like(input_ids)

        available = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        names = self._input_names or tuple(available)
        feed = {name: available[name] for name in names if name in available}
        if "input_ids" not in feed:
            raise InferenceError("ONNX model input names do not include input_ids")
        return feed

    def _infer_prefix(self, commands: Sequence[tuple[str, float | None]]) -> ScoreEntry:
        feed = self._encode(commands)
        try:
            raw_outputs = self._onnx_session.run(None, feed)
        except Exception as exc:
            raise InferenceError(f"ONNX inference failed: {exc}") from exc

        outputs = self._named_outputs(raw_outputs)
        malicious_logits = self._select_output(
            outputs, "malicious", ("malicious", "malicious_logits", "logits")
        )
        if malicious_logits is None:
            raise InferenceError("ONNX output does not contain malicious logits")

        malicious_values = np.asarray(malicious_logits, dtype=np.float64).reshape(-1)
        if malicious_values.size < 1:
            raise InferenceError("malicious logits are empty")
        scaled_malicious = malicious_values / self.malicious_temperature
        if malicious_values.size == 2 and self._output_names.get("malicious_output") == "softmax":
            malicious_probability = float(_softmax(scaled_malicious)[1])
        else:
            malicious_probability = float(_sigmoid(scaled_malicious[:1])[0])

        tactic_logits = self._select_output(
            outputs, "tactics", ("tactics", "tactics_logits", "tactic_logits")
        )
        technique_logits = self._select_output(
            outputs, "techniques", ("techniques", "techniques_logits", "technique_logits")
        )
        tactic_scores = self._label_scores(
            tactic_logits,
            self._tactics,
            default_threshold=self._tactic_threshold,
            thresholds=self._tactic_thresholds,
            temperature=self.tactic_temperature,
        )
        technique_scores = self._label_scores(
            technique_logits,
            self._techniques,
            default_threshold=self._technique_threshold,
            thresholds=self._technique_thresholds,
            temperature=self.technique_temperature,
        )

        tactic_ids = [item.id for item in tactic_scores]
        stage = attack_stage(tactic_ids)
        severity = stage_severity(tactic_ids)
        risk_score = min(1.0, max(0.0, malicious_probability * severity))
        return ScoreEntry(
            command_index=len(commands) - 1,
            malicious_probability=malicious_probability,
            risk_score=risk_score,
            attack_stage=stage,
            tactics=tactic_scores,
            techniques=technique_scores,
            alert=malicious_probability >= self.malicious_threshold,
        )

    def _named_outputs(self, raw_outputs: Any) -> dict[str, Any]:
        if isinstance(raw_outputs, Mapping):
            return {str(name): value for name, value in raw_outputs.items()}
        if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, (str, bytes)):
            return {"malicious": raw_outputs}

        names: list[str] = []
        try:
            names = [item.name for item in self._onnx_session.get_outputs()]
        except (AttributeError, TypeError):
            pass
        if len(names) == len(raw_outputs):
            return {name: value for name, value in zip(names, raw_outputs, strict=True)}
        if len(raw_outputs) == 1:
            return {"malicious": raw_outputs[0]}
        return {f"output_{index}": value for index, value in enumerate(raw_outputs)}

    def _select_output(
        self, outputs: Mapping[str, Any], field: str, aliases: Sequence[str]
    ) -> Any | None:
        configured = self._output_names.get(field)
        if configured and configured in outputs:
            return outputs[configured]
        for alias in aliases:
            if alias in outputs:
                return outputs[alias]
        return None

    def _label_scores(
        self,
        logits: Any | None,
        labels: Sequence[_Label],
        *,
        default_threshold: float,
        thresholds: Mapping[str, float],
        temperature: float,
    ) -> list[TacticScore] | list[TechniqueScore]:
        if logits is None or not labels:
            return []
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        probabilities = _sigmoid(values / temperature)
        if probabilities.size < len(labels):
            probabilities = np.pad(
                probabilities, (0, len(labels) - probabilities.size), constant_values=0.0
            )
        result: list[TacticScore | TechniqueScore] = []
        score_type = TacticScore if labels and labels[0].id.startswith("TA") else TechniqueScore
        for label, probability in zip(labels, probabilities, strict=False):
            threshold = float(thresholds.get(label.id, default_threshold))
            if probability >= threshold:
                result.append(
                    score_type(
                        id=label.id,
                        name=label.name,
                        score=float(probability),
                    )
                )
        return result


class SessionState:
    """Mutable state for one online command-session stream."""

    def __init__(self, *, scorer: SessionScorer, session_id: str) -> None:
        self._scorer = scorer
        self.session_id = session_id
        self._commands: list[tuple[str, float | None]] = []
        self._entries: list[ScoreEntry] = []
        self._alerted = False
        self._peak_probability = 0.0

    @property
    def entries(self) -> tuple[ScoreEntry, ...]:
        return tuple(self._entries)

    @property
    def alerted(self) -> bool:
        return self._alerted

    @property
    def peak_probability(self) -> float:
        return self._peak_probability

    def update(self, cmd: str, dt: float | None = None) -> ScoreEntry:
        """Append one command and score the resulting prefix."""

        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("cmd must contain non-whitespace characters")
        dt_bucket(dt)
        self._commands.append((cmd, dt))
        entry = self._scorer.score_commands(self._commands)
        self._peak_probability = max(self._peak_probability, entry.malicious_probability)
        self._alerted = self._alerted or entry.alert
        if entry.alert != self._alerted:
            entry = entry.model_copy(update={"alert": self._alerted})
        elif self._alerted and not entry.alert:
            entry = entry.model_copy(update={"alert": True})
        self._entries.append(entry)
        return entry

    def result(self) -> SequenceScore:
        """Return the stable scoring result for all observed prefixes."""

        return SequenceScore(
            session_id=self.session_id,
            entries=list(self._entries),
            peak_probability=self._peak_probability,
            alerted=self._alerted,
            model_version=self._scorer.model_version,
        )
