# cmd-session-risk

Research implementation for Unix command-session malicious probability and
MITRE ATT&CK stage/technique prediction.

The first release scores command prefixes, emits alert-only decisions, and
exports a CPU INT8 ONNX artifact. It does not block commands.

See `configs/` for prompts, `scripts/` for data/teacher preparation, `kaggle/`
for the fallback GPU entry points, and `src/cmdrisk/` for the package.
