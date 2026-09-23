"""ONNX FP32 export, parity checks, and dynamic INT8 quantization."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cmdrisk.artifact_manifest import (
    EXPORT_MANIFEST_FILENAME,
    EXPORT_MANIFEST_SCHEMA,
    EXPORT_MANIFEST_VERSION,
    artifact_entry,
    build_artifact_entries,
    verify_export_manifest,
    write_manifest_atomic,
)
from cmdrisk.model_factory import load_risk_model
from cmdrisk.tokenization import CommandTokenizer


def _max_abs_error(expected: Any, actual: Any) -> float:
    """Max absolute difference that tolerates empty (zero-width) heads."""

    difference = np.abs(np.asarray(expected) - np.asarray(actual))
    if difference.size == 0:
        return 0.0
    return float(np.max(difference))


def _strip_value_info(path: str | Path) -> None:
    """Drop exporter-provided intermediate shape annotations.

    ``torch.export`` records intermediate shapes using the traced batch size.
    ONNX Runtime's dynamic-quantization pass rewrites ``Gemm`` into
    ``MatMul`` + ``Transpose`` and then re-runs shape inference, which trips
    over the now-stale annotations (for example the transposed weight keeps its
    pre-transpose shape) and aborts with an ``InferenceError``.  The
    annotations are redundant, so drop them and let ONNX Runtime infer shapes.
    """

    import onnx

    model = onnx.load(str(path))
    if model.graph.value_info:
        del model.graph.value_info[:]
        onnx.save(model, str(path))


def _supports_dynamo_export(torch: Any) -> bool:
    """Return whether ``torch.onnx.export`` accepts ``dynamo=True``."""

    import inspect

    try:
        parameters = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
        return False
    return "dynamo" in parameters


class _OnnxExportWrapper:
    """Created lazily to avoid importing Torch APIs at module import time."""

    @staticmethod
    def build(model: Any) -> Any:
        from torch import nn

        class Wrapper(nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any, Any]:
                output = self.wrapped(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                return (
                    output.malicious_logits,
                    output.tactic_logits,
                    output.technique_logits,
                )

        return Wrapper(model)


def _copy_optional(source: str | Path | None, destination: Path) -> None:
    if source is None:
        return
    source_path = Path(source)
    if source_path.resolve() == destination.resolve():
        return
    if source_path.is_file():
        shutil.copy2(source_path, destination)
    elif source_path.is_dir():
        for child in source_path.iterdir():
            if child.is_file():
                shutil.copy2(child, destination / child.name)


RUNTIME_CONFIG_FILENAME = "runtime_config.json"


def write_runtime_config(
    destination: str | Path,
    *,
    model_version: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Record the served model identity inside an ONNX artifact directory.

    ``model_version`` defaults to the artifact directory's parent name, which is
    the run identifier produced by the training pipeline.  The runtime loader
    reads this value back and stamps it onto every score, so stored detections
    can be attributed to a specific artifact revision instead of ``unknown``.

    Returns the resolved version string.
    """

    root = Path(destination)
    version = str(model_version).strip() if model_version is not None else ""
    version = version or root.parent.name or "unknown"
    path = root / RUNTIME_CONFIG_FILENAME
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = _read_json_object(path, description="runtime config")
    payload["version"] = 1
    payload["model_version"] = version
    if extra is not None:
        payload.update(dict(extra))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return version


def _write_model_card(
    output_dir: Path,
    *,
    checkpoint: Path,
    model_name: str,
    metrics: dict[str, Any] | None,
) -> None:
    metric_block = (
        json.dumps(metrics, indent=2, sort_keys=True) if metrics else "Not available"
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Command Session Risk ONNX model",
                "",
                f"- Base checkpoint: `{checkpoint}`",
                f"- Backbone: `{model_name}` (Apache-2.0)",
                "- Outputs: malicious logit, 14 ATT&CK tactic logits, and dynamic "
                "technique logits",
                "- Intended use: Unix/Linux command-prefix risk scoring and alerting",
                "- Not intended for automatic blocking without independent validation",
                "",
                "## Metrics",
                "",
                "```json",
                metric_block,
                "```",
                "",
                "The INT8 model is dynamically quantized with ONNX Runtime.  Compare",
                "parity, recall, FPR, and detection-delay metrics before deployment.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return dict(value)


def _metrics_promotion_status(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a fail-closed summary of the training promotion report."""

    if metrics is None:
        return {
            "available": False,
            "passed": False,
            "failures": ["metrics_missing"],
        }
    promotion = metrics.get("promotion")
    if not isinstance(promotion, Mapping):
        return {
            "available": False,
            "passed": False,
            "failures": ["promotion_report_missing"],
        }

    failures: list[str] = []
    if promotion.get("passed") is not True:
        failures.append("promotion_gate_failed")
    missing_metrics = promotion.get("missing_metrics")
    if isinstance(missing_metrics, (list, tuple)) and missing_metrics:
        failures.append("promotion_metrics_missing")
    violations = promotion.get("violations")
    if isinstance(violations, (list, tuple)) and violations:
        failures.append("promotion_violations_present")

    checks = promotion.get("checks")
    if isinstance(checks, Mapping):
        for name in (
            "evaluation_slices_present",
            "evaluation_slices_passed",
            "proxy_audit",
        ):
            if name in checks and checks[name] is not True:
                failures.append(name)

    passed = promotion.get("passed") is True and not failures
    return {
        "available": True,
        "passed": passed,
        "failures": list(dict.fromkeys(failures)),
    }


def _estimated_latency_value(
    report: Mapping[str, Any], percentile_name: str
) -> float | None:
    direct_names = (
        (percentile_name, f"max_{percentile_name}", f"summary_{percentile_name}")
        if percentile_name == "p95_ms"
        else (percentile_name, f"max_{percentile_name}", f"summary_{percentile_name}")
    )
    for name in direct_names:
        value = report.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number >= 0:
                return number
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        value = summary.get(percentile_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    results = report.get("results")
    if isinstance(results, list):
        values: list[float] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            value = item.get(percentile_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            return max(values)
    return None


def _latency_report_status(
    report: Mapping[str, Any] | None,
    *,
    relative_path: str | None = None,
    integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the latency evidence required by the promotion gate."""

    if report is None:
        return {
            "present": False,
            "path": None,
            "sha256": None,
            "size": None,
            "profile": None,
            "threshold_ms": None,
            "p95_threshold_ms": None,
            "p99_threshold_ms": None,
            "tolerance_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "profile_passed": False,
            "passed": False,
            "failures": ["latency_report_missing"],
        }

    failures: list[str] = []
    profile = report.get("profile")
    if not isinstance(profile, Mapping):
        profile = None
        failures.append("latency_profile_missing")
    profile_passed = report.get("profile_passed")
    if profile_passed is None and profile is not None:
        profile_passed = profile.get("passed")
    if profile_passed is not True:
        failures.append("latency_profile_failed")

    p95_threshold = report.get("p95_threshold_ms", report.get("threshold_ms"))
    p99_threshold = report.get("p99_threshold_ms")
    tolerance = report.get("tolerance_ms")
    numeric_thresholds: dict[str, float] = {}
    for name, value in (
        ("p95_threshold_ms", p95_threshold),
        ("p99_threshold_ms", p99_threshold),
        ("tolerance_ms", tolerance),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= 0:
            numeric_thresholds[name] = float(value)
        else:
            failures.append(f"{name}_missing")

    p95_ms = _estimated_latency_value(report, "p95_ms")
    p99_ms = _estimated_latency_value(report, "p99_ms")
    if p95_ms is None:
        failures.append("p95_ms_missing")
    if p99_ms is None:
        failures.append("p99_ms_missing")

    if report.get("passed") is not True:
        failures.append("latency_gate_failed")
    if p95_ms is not None and p95_threshold is not None:
        allowed = float(p95_threshold) + float(tolerance or 0.0)
        if p95_ms > allowed:
            failures.append("p95_exceeds_tolerance")
    if p99_ms is not None and p99_threshold is not None:
        allowed = float(p99_threshold) + float(tolerance or 0.0)
        if p99_ms > allowed:
            failures.append("p99_exceeds_tolerance")

    profile_value: dict[str, Any] | None = dict(profile) if profile is not None else None
    if profile_value is not None:
        profile_value["passed"] = profile_passed is True
    result: dict[str, Any] = {
        "present": True,
        "path": relative_path,
        "sha256": integrity.get("sha256") if integrity else None,
        "size": integrity.get("size") if integrity else None,
        "profile": profile_value,
        "threshold_ms": float(p95_threshold) if p95_threshold is not None else None,
        "p95_threshold_ms": float(p95_threshold) if p95_threshold is not None else None,
        "p99_threshold_ms": float(p99_threshold) if p99_threshold is not None else None,
        "tolerance_ms": float(tolerance) if tolerance is not None else None,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "profile_passed": profile_passed is True,
        "passed": not failures,
        "failures": list(dict.fromkeys(failures)),
    }
    return result


def _collect_artifact_files(root: Path) -> list[str]:
    manifest_path = root / EXPORT_MANIFEST_FILENAME
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )


def _promotion_status(
    *,
    metrics_status: Mapping[str, Any],
    latency_status: Mapping[str, Any],
    required: bool,
) -> dict[str, Any]:
    failures = [
        *[str(item) for item in metrics_status.get("failures", ())],
        *[str(item) for item in latency_status.get("failures", ())],
    ]
    passed = metrics_status.get("passed") is True and latency_status.get("passed") is True
    return {
        "required": bool(required),
        "metrics_passed": metrics_status.get("passed") is True,
        "latency_passed": latency_status.get("passed") is True,
        "eligible_for_promotion": passed,
        "passed": passed,
        "failures": list(dict.fromkeys(failures)),
    }


def update_export_manifest_latency(
    *,
    artifact_dir: str | Path,
    latency_report_path: str | Path,
    require_promotion: bool = False,
) -> dict[str, Any]:
    """Attach post-export latency evidence to an existing export manifest."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / EXPORT_MANIFEST_FILENAME
    verify_export_manifest(root, manifest_path)
    source = Path(latency_report_path).expanduser()
    if not source.is_file():
        raise ValueError(f"latency report does not exist: {source}")
    report = _read_json_object(source, description="latency report")

    destination = root / "latency_report.json"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

    manifest = _read_json_object(manifest_path, description="export manifest")
    files = _collect_artifact_files(root)
    manifest["artifacts"] = build_artifact_entries(root, files)
    manifest["checksums"] = {
        name: entry["sha256"] for name, entry in manifest["artifacts"].items()
    }
    manifest["sizes"] = {
        name: entry["size"] for name, entry in manifest["artifacts"].items()
    }
    latency_integrity = artifact_entry(root, "latency_report.json")
    latency_status = _latency_report_status(
        report,
        relative_path="latency_report.json",
        integrity=latency_integrity,
    )

    metrics: dict[str, Any] | None = None
    metrics_name = manifest.get("metrics")
    if isinstance(metrics_name, str):
        metrics_path = root / metrics_name
        if metrics_path.is_file():
            metrics = _read_json_object(metrics_path, description="metrics")
    metrics_status = _metrics_promotion_status(metrics)
    manifest["latency"] = latency_status
    manifest["promotion"] = _promotion_status(
        metrics_status=metrics_status,
        latency_status=latency_status,
        required=require_promotion,
    )
    write_manifest_atomic(manifest_path, manifest)
    verify_export_manifest(root, manifest_path)

    if require_promotion and manifest["promotion"]["passed"] is not True:
        failures = ", ".join(manifest["promotion"]["failures"]) or "unknown failure"
        raise ValueError(f"model is not eligible for promotion: {failures}")
    return manifest


def export_onnx(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    tokenizer_source: str | Path | None = None,
    calibration_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    opset_version: int = 18,
    parity_atol: float = 1e-4,
    int8_parity_atol: float = 0.05,
    skip_int8: bool = False,
    require_promotion: bool = False,
    latency_report_path: str | Path | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort
    import torch

    if opset_version < 18:
        # The ``torch.export``-based ONNX exporter lowers reductions such as
        # ModernBERT's attention Split with ``num_outputs``, which the opset-17
        # converter cannot represent.  Opset 18 also has broader ONNX Runtime
        # operator coverage, so require it rather than emit a graph ORT rejects.
        raise ValueError("opset_version must be >= 18")
    if not _supports_dynamo_export(torch):
        raise RuntimeError(
            "ONNX export requires the torch.export-based exporter "
            "(torch.onnx.export(dynamo=True), PyTorch >= 2.6)"
        )
    checkpoint_path = Path(checkpoint)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # Export the decomposed eager-attention graph.  SDPA is what training uses
    # (it is far cheaper in VRAM), but the torch.export/ONNX lowering of fused
    # scaled-dot-product attention is version sensitive, so the exported graph
    # is built from the equivalent eager path that ONNX Runtime executes well.
    model = load_risk_model(checkpoint_path)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    model.eval()
    # Preserve the truncation budgets written by the phase that produced
    # this checkpoint.  Forcing long-phase defaults here would silently
    # change runtime prefixes for short-only or custom-length runs.
    tokenizer = CommandTokenizer.from_pretrained(tokenizer_source or checkpoint_path)
    sample = tokenizer.encode(
        ["whoami", {"cmd": "curl -o /tmp/a https://example.invalid/a", "dt": 2.0}]
    )
    if not sample.input_ids:
        raise RuntimeError("tokenizer produced an empty sample")
    input_ids = torch.tensor([list(sample.input_ids)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    wrapper = _OnnxExportWrapper.build(model)
    wrapper.eval()
    fp32_path = destination / "model.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["malicious_logits", "tactic_logits", "technique_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "malicious_logits": {0: "batch"},
            "tactic_logits": {0: "batch"},
            "technique_logits": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=True,
    )

    _strip_value_info(fp32_path)

    with torch.inference_mode():
        reference = wrapper(input_ids, attention_mask)
    reference_values = [value.detach().cpu().numpy() for value in reference]
    session = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    onnx_values = session.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        },
    )
    max_abs_error = max(
        _max_abs_error(expected, actual)
        for expected, actual in zip(reference_values, onnx_values, strict=True)
    )
    if max_abs_error > parity_atol:
        raise RuntimeError(
            f"ONNX parity check failed: max absolute error {max_abs_error} > {parity_atol}"
        )

    int8_path: Path | None = None
    int8_max_abs_error: float | None = None
    if not skip_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = destination / "model.int8.onnx"
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        int8_session = ort.InferenceSession(
            str(int8_path), providers=["CPUExecutionProvider"]
        )
        int8_values = int8_session.run(
            None,
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
            },
        )
        int8_max_abs_error = max(
            _max_abs_error(expected, actual)
            for expected, actual in zip(reference_values, int8_values, strict=True)
        )
        if int8_max_abs_error > int8_parity_atol:
            raise RuntimeError(
                "INT8 ONNX parity check failed: max absolute error "
                f"{int8_max_abs_error} > {int8_parity_atol}"
            )

    tokenizer.save_pretrained(destination)
    model.label_vocabulary.save(destination / "labels.json")
    _copy_optional(calibration_path, destination / "calibration.json")
    resolved_version = write_runtime_config(destination, model_version=model_version)
    copied_metrics = destination / "metrics.json"
    _copy_optional(metrics_path, copied_metrics)
    metrics: dict[str, Any] | None = None
    if copied_metrics.is_file():
        metrics = json.loads(copied_metrics.read_text(encoding="utf-8"))
    _write_model_card(
        destination,
        checkpoint=checkpoint_path,
        model_name=model.config.name_or_path or "answerdotai/ModernBERT-base",
        metrics=metrics,
    )
    if latency_report_path is not None:
        latency_source = Path(latency_report_path).expanduser()
        if not latency_source.is_file():
            raise ValueError(f"latency report does not exist: {latency_source}")
        destination_report = destination / "latency_report.json"
        if latency_source.resolve() != destination_report.resolve():
            shutil.copy2(latency_source, destination_report)

    artifact_files = _collect_artifact_files(destination)
    artifacts = build_artifact_entries(destination, artifact_files)
    metrics_status = _metrics_promotion_status(metrics)
    latency_report: dict[str, Any] | None = None
    latency_integrity: dict[str, Any] | None = None
    if (destination / "latency_report.json").is_file():
        latency_report = _read_json_object(
            destination / "latency_report.json", description="latency report"
        )
        latency_integrity = artifact_entry(destination, "latency_report.json")
    latency_status = _latency_report_status(
        latency_report,
        relative_path="latency_report.json" if latency_report is not None else None,
        integrity=latency_integrity,
    )
    promotion = _promotion_status(
        metrics_status=metrics_status,
        latency_status=latency_status,
        required=require_promotion,
    )

    manifest = {
        "schema": EXPORT_MANIFEST_SCHEMA,
        "version": EXPORT_MANIFEST_VERSION,
        "format": "onnx",
        "opset": opset_version,
        "fp32_model": fp32_path.name,
        "int8_model": int8_path.name if int8_path else None,
        "parity_max_abs_error": max_abs_error,
        "parity_atol": parity_atol,
        "int8_parity_max_abs_error": int8_max_abs_error,
        "int8_parity_atol": int8_parity_atol if int8_path else None,
        "label_vocabulary": "labels.json",
        "runtime_config": RUNTIME_CONFIG_FILENAME,
        "model_version": resolved_version,
        "calibration": "calibration.json"
        if (destination / "calibration.json").is_file()
        else None,
        "metrics": "metrics.json" if copied_metrics.is_file() else None,
        "artifacts": artifacts,
        "checksums": {
            name: entry["sha256"] for name, entry in artifacts.items()
        },
        "sizes": {name: entry["size"] for name, entry in artifacts.items()},
        "latency": latency_status,
        "promotion": promotion,
    }

    if require_promotion and not promotion["passed"]:
        failures = ", ".join(promotion["failures"]) or "unknown failure"
        raise ValueError(f"model is not eligible for promotion: {failures}")

    manifest_path = destination / EXPORT_MANIFEST_FILENAME
    write_manifest_atomic(manifest_path, manifest)
    verify_export_manifest(destination, manifest_path)
    return manifest
