# Quality Gates

The 500-session audit is a model-proxy gate. It does not establish human gold
labels or production-grade FPR.

Required report fields:

- Recall and FPR with Wilson confidence intervals.
- Detection delay relative to `first_evidence_index`, plus alert command number.
- PR-AUC, ROC-AUC, Brier score, ECE, and ATT&CK macro-F1.
- Slices for real-only, real plus semi-real, synthetic stress, and 2024 OOD.
- FP32/INT8 parity and latency on an 8-core x86 AVX2 CPU profile.
- Reviewer model revisions, prompt hashes, Fleiss' kappa, and unresolved count.

Promotion requires the planned point-estimate thresholds and a clean leakage
audit. Any family, parent sample, or synthetic child crossing splits invalidates
the run.
