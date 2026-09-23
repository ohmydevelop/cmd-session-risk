"""End-to-end two-phase training and calibration pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cmdrisk.model_factory import load_risk_backbone, load_risk_model
from cmdrisk.tokenization import CommandTokenizer
from cmdrisk.training.calibration import Calibration, fit_calibration
from cmdrisk.training.config import TrainingRunConfig, choose_batch_config
from cmdrisk.training.data import PrefixExample, expand_records, load_records
from cmdrisk.training.dataset import CommandRiskCollator, CommandRiskDataset
from cmdrisk.training.labels import LabelVocabulary, compute_pos_weight
from cmdrisk.training.metrics import (
    TACTIC_SUPPORT_FLOOR,
    TECHNIQUE_SUPPORT_FLOOR,
    binary_classification_metrics,
    detection_delay_metrics,
    multilabel_metrics,
    promotion_report,
)
from cmdrisk.training.trainer import train_phase

REQUIRED_EVALUATION_SLICES = {
    "real_only",
    "real_plus_semi_real",
    "synthetic_stress",
    "ood_2024",
}

# A required evaluation slice must carry both classes before it can evidence a
# production promotion.  A single-class slice makes ECE degenerate (for an
# all-positive slice ECE == 1 - mean(p), i.e. it measures confidence rather
# than calibration) and leaves FPR/ROC undefined.  Missing support is reported
# as an explicit ``insufficient_positives`` / ``insufficient_negatives``
# violation so a class-degenerate slice can never pass vacuously.
MIN_SLICE_POSITIVES = 1
MIN_SLICE_NEGATIVES = 1

_SLICE_METRIC_ORDER = (
    "recall",
    "fpr",
    "precision",
    "pr_auc",
    "roc_auc",
    "brier",
    "ece",
    "detection_rate",
    "detection_delay_commands",
    "tactic_macro_f1",
    "technique_macro_f1",
)


def _distributed_world_size() -> int:
    value = os.environ.get("WORLD_SIZE")
    if value and value.isdigit():
        return max(1, int(value))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _resize_model_to_tokenizer(
    model: Any, tokenizer: CommandTokenizer
) -> None:
    """Match the embedding table to the marker-extended tokenizer vocabulary."""

    target_vocab_size = len(tokenizer.tokenizer)
    current_vocab_size = int(model.get_input_embeddings().num_embeddings)
    if current_vocab_size != target_vocab_size:
        model.resize_token_embeddings(target_vocab_size)


def detect_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return float(properties.total_memory / (1024**3))


def _stable_family_partition(
    examples: Sequence[PrefixExample],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[PrefixExample], list[PrefixExample]]:
    """Split by family so calibration and selection remain family-disjoint."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")

    family_ids = sorted({example.family_id for example in examples})
    calibration_families: set[str] = set()
    target = max(1, int(round(len(family_ids) * fraction))) if family_ids else 0
    ranked = sorted(
        family_ids,
        key=lambda family_id: hashlib.sha256(f"{seed}:{family_id}".encode()).digest(),
    )
    calibration_families.update(ranked[:target])
    first: list[PrefixExample] = []
    second: list[PrefixExample] = []
    for example in examples:
        (second if example.family_id in calibration_families else first).append(example)

    if set(example.family_id for example in first) & set(
        example.family_id for example in second
    ):
        raise AssertionError("calibration and selection families must be disjoint")
    return first, second


def _collect_logits(
    *,
    model: Any,
    examples: Sequence[PrefixExample],
    tokenizer: CommandTokenizer,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = CommandRiskDataset(examples, tokenizer)
    collator = CommandRiskCollator(tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    model.eval()
    malicious: list[np.ndarray] = []
    tactics: list[np.ndarray] = []
    techniques: list[np.ndarray] = []
    malicious_labels: list[np.ndarray] = []
    tactic_labels: list[np.ndarray] = []
    technique_labels: list[np.ndarray] = []
    session_ids: list[str] = []
    prefix_indices: list[int] = []
    command_counts: list[int] = []
    source_kinds: list[str] = []
    evaluation_slices: list[str] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            malicious.append(outputs.malicious_logits.detach().cpu().numpy())
            tactics.append(outputs.tactic_logits.detach().cpu().numpy())
            techniques.append(outputs.technique_logits.detach().cpu().numpy())
            malicious_labels.append(labels["malicious"].numpy())
            tactic_labels.append(labels["tactics"].numpy())
            technique_labels.append(labels["techniques"].numpy())
            batch_examples = dataset.examples[
                offset : offset + len(batch["input_ids"])
            ]
            session_ids.extend(example.session_id for example in batch_examples)
            prefix_indices.extend(example.prefix_index for example in batch_examples)
            command_counts.extend(example.command_count for example in batch_examples)
            source_kinds.extend(example.source_kind for example in batch_examples)
            evaluation_slices.extend(
                example.evaluation_slice for example in batch_examples
            )
            offset += len(batch_examples)
    return {
        "malicious_logits": np.concatenate(malicious),
        "tactic_logits": np.concatenate(tactics),
        "technique_logits": np.concatenate(techniques),
        "malicious_labels": np.concatenate(malicious_labels),
        "tactic_labels": np.concatenate(tactic_labels),
        "technique_labels": np.concatenate(technique_labels),
        "session_ids": np.asarray(session_ids, dtype=object),
        "prefix_indices": np.asarray(prefix_indices, dtype=np.int64),
        "command_counts": np.asarray(command_counts, dtype=np.int64),
        "source_kinds": np.asarray(source_kinds, dtype=object),
        "evaluation_slices": np.asarray(evaluation_slices, dtype=object),
    }


def _slice_metric_definitions(slice_metrics: dict[str, Any]) -> dict[str, bool]:
    """Return whether each slice metric has the required label support."""

    malicious = slice_metrics["malicious"]
    detection_delay = slice_metrics["detection_delay"]
    positive_count = int(malicious.get("positive_count", 0))
    negative_count = int(malicious.get("negative_count", 0))
    has_positives = positive_count > 0
    has_negatives = negative_count > 0
    has_sessions = positive_count + negative_count > 0
    has_malicious_sessions = int(detection_delay.get("malicious_sessions", 0)) > 0
    tactic_supported = (
        int(
            slice_metrics["tactic"].get(
                f"heads_support_ge_{TACTIC_SUPPORT_FLOOR}"
            )
            or 0
        )
        > 0
    )
    technique_supported = (
        int(
            slice_metrics["technique"].get(
                f"heads_support_ge_{TECHNIQUE_SUPPORT_FLOOR}"
            )
            or 0
        )
        > 0
    )
    both_classes = has_positives and has_negatives
    return {
        "recall": both_classes,
        "fpr": both_classes,
        "precision": both_classes,
        "pr_auc": both_classes,
        "roc_auc": both_classes,
        "brier": has_sessions,
        # ECE is a calibration metric: it needs both classes to be meaningful.
        "ece": both_classes,
        "detection_rate": has_malicious_sessions,
        "detection_delay_commands": has_malicious_sessions,
        "tactic_macro_f1": tactic_supported,
        "technique_macro_f1": technique_supported,
    }


def _make_slice_promotion(
    slice_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a slice report where undefined metrics are explicit, not zero."""

    tactic = slice_metrics["tactic"]
    technique = slice_metrics["technique"]
    slice_promotion = promotion_report(
        slice_metrics["malicious"],
        slice_metrics["detection_delay"],
        tactic_macro_f1=tactic[f"macro_f1_support_ge_{TACTIC_SUPPORT_FLOOR}"],
        technique_macro_f1=technique[
            f"macro_f1_support_ge_{TECHNIQUE_SUPPORT_FLOOR}"
        ],
        tactic_macro_f1_all_heads=tactic["macro_f1_all"],
        tactic_macro_f1_support_ge_1=tactic["macro_f1_support_ge_1"],
        tactic_macro_f1_support_ge_50=tactic[
            f"macro_f1_support_ge_{TACTIC_SUPPORT_FLOOR}"
        ],
        tactic_heads_support_ge_50=tactic[
            f"heads_support_ge_{TACTIC_SUPPORT_FLOOR}"
        ],
        technique_macro_f1_all_heads=technique["macro_f1_all"],
        technique_macro_f1_support_ge_1=technique["macro_f1_support_ge_1"],
        technique_macro_f1_support_ge_50=technique[
            f"macro_f1_support_ge_{TECHNIQUE_SUPPORT_FLOOR}"
        ],
        technique_heads_support_ge_50=technique[
            f"heads_support_ge_{TECHNIQUE_SUPPORT_FLOOR}"
        ],
        require_proxy_audit=False,
    )
    definitions = _slice_metric_definitions(slice_metrics)
    defined_metrics = [name for name in _SLICE_METRIC_ORDER if definitions[name]]
    undefined_metrics = [name for name in _SLICE_METRIC_ORDER if not definitions[name]]
    slice_promotion["defined_metrics"] = defined_metrics
    slice_promotion["undefined_metrics"] = undefined_metrics

    metric_values = slice_promotion["metric_values"]
    precision = slice_metrics["malicious"].get("precision")
    metric_values["precision"] = (
        float(precision)
        if definitions["precision"] and precision is not None
        else None
    )
    for name in undefined_metrics:
        if name in metric_values:
            metric_values[name] = None
        if name in slice_promotion["checks"]:
            slice_promotion["checks"][name] = True
    slice_promotion["missing_metrics"] = [
        name
        for name in slice_promotion["missing_metrics"]
        if name not in undefined_metrics
    ]
    violations = [
        name for name, passed in slice_promotion["checks"].items() if not passed
    ]
    class_balance = _slice_class_balance(slice_metrics["malicious"])
    violations.extend(class_balance["violations"])
    slice_promotion["class_balance"] = class_balance
    slice_promotion["violations"] = list(dict.fromkeys(violations))
    slice_promotion["passed"] = bool(
        all(slice_promotion["checks"].values()) and not class_balance["violations"]
    )
    return slice_promotion


def _slice_class_balance(malicious: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when a slice cannot support classification evidence.

    Undefined metrics used to be treated as passing, which let a slice with no
    benign sessions (or no malicious sessions) satisfy the promotion gate.  A
    class-degenerate slice is now an explicit, non-silent blocker.
    """

    positive_count = int(malicious.get("positive_count", 0) or 0)
    negative_count = int(malicious.get("negative_count", 0) or 0)
    violations: list[str] = []
    if positive_count < MIN_SLICE_POSITIVES:
        violations.append("insufficient_positives")
    if negative_count < MIN_SLICE_NEGATIVES:
        violations.append("insufficient_negatives")
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "minimum_positives": MIN_SLICE_POSITIVES,
        "minimum_negatives": MIN_SLICE_NEGATIVES,
        "deficient": bool(violations),
        "violations": violations,
    }


def _metrics(
    predictions: dict[str, np.ndarray],
    calibration: Calibration,
    *,
    tactic_ids: Sequence[str],
    technique_ids: Sequence[str],
    proxy_audit_report: Any = None,
) -> dict[str, Any]:
    malicious_probability = calibration.malicious_probabilities(
        predictions["malicious_logits"]
    )
    malicious = binary_classification_metrics(
        predictions["malicious_labels"],
        malicious_probability,
        threshold=calibration.malicious_threshold,
    )
    tactic_probability = calibration.tactic_probabilities(predictions["tactic_logits"])
    technique_probability = calibration.technique_probabilities(
        predictions["technique_logits"]
    )
    tactic_thresholds = [calibration.tactic_thresholds[item] for item in tactic_ids]
    technique_thresholds = [
        calibration.technique_thresholds[item] for item in technique_ids
    ]
    tactic = multilabel_metrics(
        predictions["tactic_labels"], tactic_probability, thresholds=tactic_thresholds
    )
    technique = multilabel_metrics(
        predictions["technique_labels"],
        technique_probability,
        thresholds=technique_thresholds,
    )
    detection_delay = detection_delay_metrics(
        predictions["malicious_labels"],
        malicious_probability,
        session_ids=predictions["session_ids"].tolist(),
        prefix_indices=predictions["prefix_indices"].tolist(),
        threshold=calibration.malicious_threshold,
    )
    tactic_support = np.count_nonzero(predictions["tactic_labels"] > 0, axis=0)
    tactic_label_support = {
        tactic_id: int(count)
        for tactic_id, count in zip(tactic_ids, tactic_support, strict=True)
    }

    slices: dict[str, dict[str, Any]] = {}
    for slice_name in sorted(set(predictions["evaluation_slices"].tolist())):
        mask = predictions["evaluation_slices"] == slice_name
        if not np.any(mask):
            continue
        slice_malicious = binary_classification_metrics(
            predictions["malicious_labels"][mask],
            malicious_probability[mask],
            threshold=calibration.malicious_threshold,
        )
        slice_tactic = multilabel_metrics(
            predictions["tactic_labels"][mask],
            tactic_probability[mask],
            thresholds=tactic_thresholds,
        )
        slice_technique = multilabel_metrics(
            predictions["technique_labels"][mask],
            technique_probability[mask],
            thresholds=technique_thresholds,
        )
        slices[slice_name] = {
            "malicious": slice_malicious,
            "tactic": slice_tactic,
            "technique": slice_technique,
            "detection_delay": detection_delay_metrics(
                predictions["malicious_labels"][mask],
                malicious_probability[mask],
                session_ids=predictions["session_ids"][mask].tolist(),
                prefix_indices=predictions["prefix_indices"][mask].tolist(),
                threshold=calibration.malicious_threshold,
            ),
        }

    promotion = promotion_report(
        malicious,
        detection_delay,
        tactic_macro_f1=tactic[f"macro_f1_support_ge_{TACTIC_SUPPORT_FLOOR}"],
        technique_macro_f1=technique[
            f"macro_f1_support_ge_{TECHNIQUE_SUPPORT_FLOOR}"
        ],
        tactic_macro_f1_all_heads=tactic["macro_f1_all"],
        tactic_macro_f1_support_ge_1=tactic["macro_f1_support_ge_1"],
        tactic_macro_f1_support_ge_50=tactic[
            f"macro_f1_support_ge_{TACTIC_SUPPORT_FLOOR}"
        ],
        tactic_heads_support_ge_50=tactic[
            f"heads_support_ge_{TACTIC_SUPPORT_FLOOR}"
        ],
        technique_macro_f1_all_heads=technique["macro_f1_all"],
        technique_macro_f1_support_ge_1=technique["macro_f1_support_ge_1"],
        technique_macro_f1_support_ge_50=technique[
            f"macro_f1_support_ge_{TECHNIQUE_SUPPORT_FLOOR}"
        ],
        technique_heads_support_ge_50=technique[
            f"heads_support_ge_{TECHNIQUE_SUPPORT_FLOOR}"
        ],
        proxy_audit_report=proxy_audit_report,
        tactic_label_support=tactic_label_support,
        tactic_ids=tactic_ids,
        require_tactic_label_coverage=True,
    )
    promotion["required_evaluation_slices"] = sorted(REQUIRED_EVALUATION_SLICES)
    promotion["missing_evaluation_slices"] = sorted(
        REQUIRED_EVALUATION_SLICES - set(slices)
    )
    slice_checks: dict[str, dict[str, Any]] = {}
    for slice_name in sorted(REQUIRED_EVALUATION_SLICES):
        slice_metrics = slices.get(slice_name)
        if slice_metrics is None:
            slice_checks[slice_name] = {
                "passed": False,
                "missing_metrics": ["evaluation_slice"],
                "violations": ["evaluation_slice"],
                "defined_metrics": [],
                "undefined_metrics": [],
            }
            continue
        slice_checks[slice_name] = _make_slice_promotion(slice_metrics)
    promotion["checks"]["evaluation_slices_present"] = not promotion[
        "missing_evaluation_slices"
    ]
    promotion["checks"]["evaluation_slices_passed"] = all(
        bool(item.get("passed")) for item in slice_checks.values()
    )
    promotion["slice_checks"] = slice_checks
    promotion["violations"] = list(
        dict.fromkeys(
            [
                *promotion["violations"],
                *(
                    [
                        f"evaluation_slice:{name}"
                        for name in slice_checks
                        if not slice_checks[name]["passed"]
                    ]
                ),
            ]
        )
    )
    promotion["passed"] = bool(
        promotion["passed"]
        and promotion["checks"]["evaluation_slices_present"]
        and promotion["checks"]["evaluation_slices_passed"]
    )
    return {
        "malicious": malicious,
        "tactic": tactic,
        "technique": technique,
        "detection_delay": detection_delay,
        "slices": slices,
        "promotion": promotion,
    }


def _calibrate_and_evaluate(
    *,
    model: Any,
    tokenizer: CommandTokenizer,
    calibration_examples: Sequence[PrefixExample],
    test_examples: Sequence[PrefixExample],
    batch_size: int,
    device: torch.device,
    vocabulary: LabelVocabulary,
    max_fpr: float,
    proxy_audit_report: Any = None,
    collect_logits: Callable[..., dict[str, np.ndarray]] | None = None,
    fit: Callable[..., Calibration] | None = None,
    evaluate: Callable[..., dict[str, Any]] | None = None,
) -> tuple[Calibration, dict[str, Any]]:
    """Fit calibration on dev data and evaluate once on untouched test data."""

    if not calibration_examples or not test_examples:
        raise ValueError("calibration and test examples must both be non-empty")
    collect = collect_logits or _collect_logits
    calibration_fitter = fit or fit_calibration
    metric_builder = evaluate or _metrics

    calibration_predictions = collect(
        model=model,
        examples=calibration_examples,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    calibration = calibration_fitter(
        malicious_logits=calibration_predictions["malicious_logits"],
        malicious_labels=calibration_predictions["malicious_labels"],
        tactic_logits=calibration_predictions["tactic_logits"],
        tactic_labels=calibration_predictions["tactic_labels"],
        tactic_ids=vocabulary.tactics,
        technique_logits=calibration_predictions["technique_logits"],
        technique_labels=calibration_predictions["technique_labels"],
        technique_ids=vocabulary.techniques,
        max_fpr=max_fpr,
    )
    test_predictions = collect(
        model=model,
        examples=test_examples,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    metrics = metric_builder(
        test_predictions,
        calibration,
        tactic_ids=vocabulary.tactics,
        technique_ids=vocabulary.techniques,
        proxy_audit_report=proxy_audit_report,
    )
    return calibration, metrics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training(config: TrainingRunConfig) -> Path:
    """Run short/long fine-tuning, calibrate, and persist the training artifact.

    Torchrun is intentionally not spawned here.  Launch with ``torchrun`` for
    multi-GPU training; Hugging Face Trainer then enables DDP automatically.
    """

    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    world_size = config.world_size or _distributed_world_size()
    vram_gb = config.vram_gb if config.vram_gb is not None else detect_vram_gb()
    if vram_gb <= 0 and torch.cuda.is_available():
        vram_gb = detect_vram_gb()

    records = load_records(config.data_path)
    train_records = [item for item in records if item.get("split") in (None, "train")]
    dev_records = [item for item in records if item.get("split") == "dev"]
    test_records = [item for item in records if item.get("split") == "test"]
    if not train_records or not dev_records or not test_records:
        raise ValueError("training data must contain train, dev, and test splits")

    if config.base_checkpoint:
        checkpoint_vocab = Path(config.base_checkpoint) / "label_vocabulary.json"
        if checkpoint_vocab.is_file():
            vocabulary = LabelVocabulary.load(checkpoint_vocab)
        else:
            raise FileNotFoundError(
                f"--base-checkpoint missing label_vocabulary.json: {checkpoint_vocab}")
    else:
        vocabulary = LabelVocabulary.build(
            train_records,
            min_technique_sessions=config.min_technique_sessions,
            min_technique_families=config.min_technique_families,
        )
    if _is_main_process():
        vocabulary.save(output_dir / "label_vocabulary.json")
    _distributed_barrier()
    train_examples = expand_records(
        train_records, vocabulary, max_prefixes=config.max_prefixes
    )
    dev_examples = expand_records(dev_records, vocabulary, max_prefixes=config.max_prefixes)
    test_examples = expand_records(test_records, vocabulary, max_prefixes=config.max_prefixes)
    selection_examples, calibration_examples = _stable_family_partition(
        dev_examples, fraction=0.5, seed=config.seed
    )
    if not selection_examples or not calibration_examples:
        raise ValueError("dev split must contain enough families for selection and calibration")

    malicious_weights = compute_pos_weight(
        ([example.malicious] for example in train_examples),
        cap=config.pos_weight_cap,
    )
    tactic_weights = compute_pos_weight(
        (example.tactics for example in train_examples),
        cap=config.pos_weight_cap,
    )
    technique_weights = compute_pos_weight(
        (example.techniques for example in train_examples),
        cap=config.pos_weight_cap,
    )

    short_batch = choose_batch_config(
        vram_gb=vram_gb, phase="short", world_size=world_size
    )
    short_tokenizer = CommandTokenizer.from_pretrained(
        config.model_name,
        max_length=config.short_max_length,
        command_max_tokens=config.command_max_tokens,
    )
    if config.base_checkpoint:
        model = load_risk_model(config.base_checkpoint)
    else:
        model = load_risk_backbone(
            config.model_name,
            label_vocabulary=vocabulary,
            dropout=config.dropout,
            pos_weight_cap=config.pos_weight_cap,
            malicious_pos_weight=malicious_weights[0],
            tactic_pos_weight=tactic_weights,
            technique_pos_weight=technique_weights,
        )
    _resize_model_to_tokenizer(model, short_tokenizer)
    collator = CommandRiskCollator(short_tokenizer)
    short_dir = output_dir / "short_phase"
    train_phase(
        model=model,
        train_dataset=CommandRiskDataset(train_examples, short_tokenizer),
        eval_dataset=CommandRiskDataset(selection_examples, short_tokenizer),
        collator=collator,
        config=config,
        batch=short_batch,
        phase="short",
        output_dir=short_dir,
        max_length=config.short_max_length,
        learning_rate=config.learning_rate_short,
    )
    if _is_main_process():
        short_tokenizer.save_pretrained(short_dir)
    _distributed_barrier()

    final_dir = short_dir
    final_tokenizer = short_tokenizer
    final_model = model
    if not config.skip_long_phase:
        long_batch = choose_batch_config(
            vram_gb=vram_gb, phase="long", world_size=world_size
        )
        final_tokenizer = CommandTokenizer.from_pretrained(
            config.model_name,
            max_length=config.long_max_length,
            command_max_tokens=config.command_max_tokens,
        )
        final_model = load_risk_model(short_dir)
        final_dir = output_dir / "long_phase"
        long_collator = CommandRiskCollator(final_tokenizer)
        train_phase(
            model=final_model,
            train_dataset=CommandRiskDataset(train_examples, final_tokenizer),
            eval_dataset=CommandRiskDataset(selection_examples, final_tokenizer),
            collator=long_collator,
            config=config,
            batch=long_batch,
            phase="long",
            output_dir=final_dir,
            max_length=config.long_max_length,
            learning_rate=config.learning_rate_long,
        )
        if _is_main_process():
            final_tokenizer.save_pretrained(final_dir)
        _distributed_barrier()

    # Calibration can run after DDP training completes.  Each rank computes the
    # same deterministic result; rank zero writes the artifact.
    calibration_batch = choose_batch_config(
        vram_gb=vram_gb,
        phase="long" if not config.skip_long_phase else "short",
        world_size=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)
    proxy_audit_report: Any = None
    if config.proxy_audit_path:
        proxy_audit_report = json.loads(
            Path(config.proxy_audit_path).read_text(encoding="utf-8")
        )
    calibration, metrics = _calibrate_and_evaluate(
        model=final_model,
        tokenizer=final_tokenizer,
        calibration_examples=calibration_examples,
        test_examples=test_examples,
        batch_size=calibration_batch.micro_batch_size,
        device=device,
        vocabulary=vocabulary,
        max_fpr=config.calibration_fpr,
        proxy_audit_report=proxy_audit_report,
    )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        calibration.save(output_dir / "calibration.json")
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(
            output_dir / "training_config.json",
            {
                **config.__dict__,
                "vram_gb": vram_gb,
                "world_size": world_size,
                "effective_train_examples": len(train_examples),
                "effective_dev_examples": len(dev_examples),
                "selection_examples": len(selection_examples),
                "calibration_examples": len(calibration_examples),
                "effective_test_examples": len(test_examples),
                "calibration_family_count": len(
                    {example.family_id for example in calibration_examples}
                ),
                "test_session_count": len({example.session_id for example in test_examples}),
                "technique_vocabulary_size": len(vocabulary.techniques),
            },
        )
    return final_dir


def finite_or_raise(values: Sequence[float]) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("training produced non-finite values")
