"""Data ingestion, normalization, splitting, and prefix generation."""

from .adapters import (
    AtomicRedTeamAdapter,
    MicrosoftIoTAdapter,
    ScriptRepoAdapter,
    ShellAttackEvolutionAdapter,
    ZenodoAdapter,
    adapter_for_kind,
)
from .dedup import cap_families, exact_deduplicate, family_aware_deduplicate
from .io import read_jsonl, write_jsonl
from .normalization import clean_command, normalize_record, session_skeleton, skeletonize_command
from .pipeline import prepare_sessions, provenance_report
from .prefixes import build_prefixes, build_prefix_records
from .sampling import sample_training_quota
from .splitting import family_aware_split

__all__ = [
    "AtomicRedTeamAdapter",
    "MicrosoftIoTAdapter",
    "ScriptRepoAdapter",
    "ShellAttackEvolutionAdapter",
    "ZenodoAdapter",
    "adapter_for_kind",
    "build_prefix_records",
    "build_prefixes",
    "cap_families",
    "clean_command",
    "exact_deduplicate",
    "family_aware_deduplicate",
    "family_aware_split",
    "normalize_record",
    "prepare_sessions",
    "provenance_report",
    "read_jsonl",
    "sample_training_quota",
    "session_skeleton",
    "skeletonize_command",
    "write_jsonl",
]
