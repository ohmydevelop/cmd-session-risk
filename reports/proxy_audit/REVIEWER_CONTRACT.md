# Blind Proxy-Review Contract (v1)

You are an independent reviewer in a **blind** command-session risk audit.

## What you may read

- `reports/proxy_audit/blind/chunk_*.jsonl` — the ONLY permitted input.
- Each line: `{"session_id": "...", "commands": [{"cmd": "...", "dt": 0.0|null}, ...]}`

## What you must NOT read

Anything that reveals the dataset label or another reviewer's answer:

- `data/**` (especially `data/processed/v2/test.jsonl`)
- `reports/proxy_audit/cases.jsonl`, `manifest.json`
- `reports/proxy_audit/votes/<other reviewer>/`
- any file containing `label`, `gold`, `malicious: 0|1`, or `source.kind`

If you accidentally see a label, say so in your final report and mark those
sessions `uncertain` instead of using the leaked information.

## Task

For each session decide whether the **command sequence as a whole** is a
realistic malicious intrusion chain, using only the commands and their ordering.

Judge the *chain*, not individual commands. `whoami`, `uname -a`, `curl`,
`chmod +x`, `./install.sh` are each ordinary; only their combination, target
paths, encodings, and ordering carry attack meaning. A download-then-chmod-then-run
sequence is also exactly what a legitimate installer does: call it malicious only
when the surrounding evidence (credential access, persistence, C2, defense
evasion, destructive impact, or explicit attacker tooling) supports it.

## Output schema (strict)

Write one JSON object per line, in the same order as the input chunk:

```json
{"session_id": "...", "decision": "malicious", "first_evidence_index": 3,
 "technique_evidence": {"T1105": [3], "T1059.004": [4]}, "tactics": ["Command and Control", "Execution"],
 "confidence": 0.82}
```

Rules enforced by the aggregator:

- `decision` is one of `malicious`, `benign`, `uncertain`.
- `first_evidence_index` is the **0-based index** of the earliest command that
  is only explicable as part of an intrusion. Required when `malicious`,
  must be `null` (or omitted) otherwise.
- `technique_evidence` maps technique IDs to the 0-based command indices that
  show them. Indices must be `>= first_evidence_index`.
- `tactics` may be MITRE tactic IDs or names. Must be empty when not `malicious`.
- `confidence` is a float in `[0, 1]`.
- Use `uncertain` only when the chain is genuinely indistinguishable without
  host context. `uncertain` is a real answer, not a hedge.

## Allowed technique IDs

T1016 System Network Configuration Discovery · T1021 Remote Services ·
T1027 Obfuscated Files or Information · T1033 System Owner/User Discovery ·
T1041 Exfiltration Over C2 Channel · T1046 Network Service Discovery ·
T1053.003 Cron · T1059.004 Unix Shell · T1070 Indicator Removal ·
T1071 Application Layer Protocol · T1082 System Information Discovery ·
T1083 File and Directory Discovery · T1105 Ingress Tool Transfer ·
T1110 Brute Force · T1140 Deobfuscate/Decode Files or Information ·
T1222 File and Directory Permissions Modification ·
T1486 Data Encrypted for Impact · T1490 Inhibit System Recovery ·
T1562 Impair Defenses

## Output location

`reports/proxy_audit/votes/<REVIEWER_ID>/chunk_XXX.jsonl`, one file per input
chunk, same basename. Every input session must appear exactly once.
