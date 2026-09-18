# Data Governance

## Source classes

- `real_malicious`: public honeypot or sandbox attack sessions.
- `real_benign`: reconstructed authorized-user sessions.
- `script_benign`: commands extracted from permissively licensed shell,
  Docker, CI, Ansible, or Kubernetes source.
- `llm_variant`: a semantic-preserving child of a real malicious session.
- `hard_negative`: a generated benign session with attack-like surface form.

## Required provenance

Every source entry records its URL, revision or content hash, declared license,
redistribution status, retrieval date, local password (when required), and any
known contamination. Raw source files are never committed.

Microsoft IoT honeypot data remains local-only until redistribution terms are
confirmed. Shell Attack Evolution and Zenodo/KYPO require CC BY 4.0 attribution.
Atomic Red Team and allowlisted script repositories require their declared
permissive licenses to be retained in the source manifest.

## Split safety

Exact duplicates, near-duplicate clusters, campaigns, and synthetic parent-child
relationships share one `family_id`. Families, not rows, are split. Parent and
child records must remain in the same split.

## Reporting

The generated data card reports requested versus realized quotas, source and
license counts, duplicate/family reduction, class balance, command-length
distribution, missing `dt` rate, split checksum, and all provenance drift.
