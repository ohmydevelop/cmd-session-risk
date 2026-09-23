# Quality Gates

Promotion is fail-closed. A failure of any gate below blocks promotion, and a
model that cannot produce the required evidence is treated exactly like a model
that failed the gate. There is no "warn only" path in the promotion code.

## Point-estimate gates

| Metric | Threshold | Direction |
| --- | --- | --- |
| Recall | >= 0.90 | higher is better |
| False positive rate (FPR) | <= 0.02 | lower is better |
| PR-AUC | >= 0.90 | higher is better |
| ROC-AUC | >= 0.95 | higher is better |
| Brier score | <= 0.10 | lower is better |
| Expected calibration error (ECE) | <= 0.05 | lower is better |
| Tactic macro-F1 (heads with support >= 50) | >= 0.65 | higher is better |
| Technique macro-F1 (heads with support >= 50) | >= 0.65 | higher is better |
| Tactic label coverage | >= 10 heads with support >= 50 | higher is better |
| Detection rate | >= 0.90 | higher is better |
| Median censored detection delay | <= 3 commands | lower is better |

Recall and FPR are reported with Wilson confidence intervals. The point
estimates above are the promotion decision; the intervals are diagnostic and
recorded for reviewers.

## Multivariate support floors

Multi-label macro metrics are reported at explicit per-head support floors. A
head's support is `true_positive + false_negative` in the evaluated label
matrix, which is the number of positive examples for that head. For every
configured floor `F`, including `F = 1`, `10`, and `50`, the metric block emits:

- `macro_precision_support_ge_<F>`
- `macro_recall_support_ge_<F>`
- `macro_f1_support_ge_<F>`
- `heads_support_ge_<F>`

Each macro value is the mean over heads with support at least `F`. If no head
qualifies, the macro value is `null`; it is never silently converted to `0.0`.
The count is still emitted as `0`.

The historical all-heads values remain unchanged and are still published:
`macro_precision`, `macro_recall`, `macro_f1`, and the exact alias
`macro_f1_all`. Floor-1 values remain available in both their explicit
`*_support_ge_1` keys and the continuity aliases `supported_macro_precision`,
`supported_macro_recall`, and `supported_macro_f1`. `supported_heads`,
`total_heads`, and `unsupported_heads` are retained as diagnostic counts.

Macro-F1 is only meaningful when every averaged class has enough evaluation
support. The promotion floors are therefore
`TACTIC_SUPPORT_FLOOR = 50` and `TECHNIQUE_SUPPORT_FLOOR = 50`. The headline
`tactic_macro_f1` and `technique_macro_f1` gates use the corresponding
floor-50 macro-F1. A floor with no qualifying head is undefined and fails the
headline gate rather than being treated as a valid zero score.

The promotion payload preserves the all-heads and lower-floor measurements:
`tactic_macro_f1_all_heads`, `tactic_macro_f1_support_ge_1`,
`tactic_macro_f1_support_ge_50`, `tactic_heads_support_ge_50`, and the same
four keys for techniques.

`tactic_label_coverage` is a separate headline hard gate. It counts tactic
heads whose positive support in the complete evaluation label matrix is at
least **50**, and requires at least **10** such heads. Its payload records
`observed`, `required`, `support_floor`, `covered_tactics`, and
`uncovered_tactics`; it is not averaged across evaluation slices. Missing
support evidence fails closed. A corpus containing, for example, seven tactics
with at least 50 sessions fails this gate even if all seven achieve high F1.

This is a **semantics correction, not a threshold relaxation**. The floor is a
measurement-validity rule: classes with only one or two positives must not
dominate a macro average. The all-heads and floor-1 scores remain published for
audit and comparison, and the explicit coverage requirement makes corpus gaps
fail honestly.

## Required evaluation slices

Every evaluation must report all four slices. Missing or empty slices fail the
`evaluation_slices_present` / `evaluation_slices_passed` checks.

- `real_only`
- `real_plus_semi_real`
- `synthetic_stress`
- `ood_2024`

Slice metrics are not averaged into the headline numbers: the headline metrics
are computed on the held-out real evaluation split, and each slice is gated and
reported separately.

Each slice check records `defined_metrics` and `undefined_metrics`. A metric is
undefined when the slice does not contain the labels required to compute it:

- `recall`, `fpr`, `precision`, `pr_auc`, and `roc_auc` require both positive
  and negative labels.
- `brier` and `ece` remain defined with one class and require at least one row.
- `detection_rate` and `detection_delay_commands` require at least one malicious
  session.
- `tactic_macro_f1` and `technique_macro_f1` require at least one tactic or
  technique head meeting the corresponding support floor (`50`).

An undefined metric is reported as `null`, removed from `missing_metrics`, and
does not fail the slice. A slice fails only when a metric in
`PROMOTION_TARGETS` is defined and below its threshold, when the slice is
missing, or when the slice has zero sessions. The old coercion of undefined
discrimination metrics to `0.0` is not a valid gate failure and is no longer
used. Defined metrics still fail closed when their values are missing,
non-finite, or outside the target direction.

## Censored detection delay

Detection delay is measured in commands from the labelled onset
(`first_evidence_index`) to the first alert.

- A detected session contributes `alert_index - onset`.
- A **missed** session is censored, not dropped: it contributes the number of
  commands observed after onset through the end of the session
  (`final_index - onset`). A session that never alerts therefore contributes its
  full remaining length, so silently missing an attack cannot improve the
  metric.
- The headline delay is the median over detected **and** censored sessions
  (`median_censored_detection_delay_commands`).
- Detection rate is the fraction of malicious sessions that alerted at or after
  onset.

## Proxy audit

The audit is a model-proxy gate; it is not human gold labelling and does not by
itself establish production FPR. Promotion requires:

- At least **500 resolved** audit cases.
- **Zero unresolved** cases.
- **All** resolved cases scored by the model.
- **Zero unpredicted** cases.
- Reviewer model revisions, prompt hashes, and Fleiss' kappa recorded.
- The audit must be marked eligible, with a clean leakage verdict.

## Latency and hardware profile

Latency is measured end-to-end through ONNX Runtime on the documented CPU
profile: **8 or more logical x86-64 cores with AVX2** using the ONNX Runtime CPU
execution provider.

Default thresholds (per prefix, worst case across benchmarked lengths):

| Field | Default |
| --- | --- |
| `p95_threshold_ms` | 50.0 |
| `p99_threshold_ms` | 70.0 |
| `tolerance_ms` | 10.0 |

A run passes when `p95_ms <= p95_threshold_ms + tolerance_ms` and
`p99_ms <= p99_threshold_ms + tolerance_ms`, **and** the hardware profile gate
passes, **and** the measured execution providers are exactly
`CPUExecutionProvider`. `scripts/latency_bench.py --provider <name>` can measure
an accelerated profile (for example `CUDAExecutionProvider`); such a report is
recorded with `deployment_profile="accelerated"` and always fails the gate with
`latency_not_cpu_deployment_profile` so that accelerated hardware can never be
mistaken for the documented CPU deployment profile. A missing AVX2 flag, an unknown architecture, or fewer than 8 logical
CPUs fails the profile gate. The benchmark exits non-zero when the gate fails
(use `--allow-failure` only for diagnostics, never for promotion).

The benchmark writes a `cmdrisk.latency_report` JSON document containing
`profile`, `profile_passed`, `p95_ms`, `p99_ms`, `p95_threshold_ms`,
`p99_threshold_ms`, `tolerance_ms`, and `passed`. That file is the latency
evidence consumed by the export manifest.

## Export manifest and parity

Every exported ONNX bundle must contain `export_manifest.json` with:

- `schema` = `cmdrisk.onnx.artifact` and `version` = `1`.
- The `fp32_model` and `int8_model` file names, both present under `artifacts`.
- A `size` and SHA256 `sha256` for **every** file in the bundle.
- The FP32/INT8 parity errors (`parity_max_abs_error`,
  `int8_parity_max_abs_error`) within their tolerances.
- The `latency` evidence block and the aggregated `promotion` block.

Runtime loaders verify the manifest **before** reading any model or tokenizer
bytes. Missing files, size mismatches, SHA256 mismatches, undeclared models,
and path traversal (absolute paths, `..`, backslashes, self-reference) are all
hard failures. Only explicit injection points (`verify_manifest=False` for unit
tests) may bypass verification, and that is never suitable for deployment.

## Promotion

`cmdrisk.training.export.export_onnx(..., require_promotion=True)` and
`update_export_manifest_latency(..., require_promotion=True)` refuse to return a
promotable manifest unless:

- `metrics.json` contains a `promotion` block with `passed: true`, no missing
  metrics, no violations, and truthy slice and proxy-audit checks; and
- latency evidence is present, on-profile, and within thresholds plus
  tolerance.

`kaggle/export_onnx.py --require-promotion` exposes the same behaviour on the
command line. Validation runs both when the manifest is first written and again
after latency evidence is attached.

Any family, parent sample, or synthetic child crossing splits invalidates the
run. Grouped, leakage-aware splits are mandatory.
