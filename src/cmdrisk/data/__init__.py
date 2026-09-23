"""Data ingestion, normalization, splitting, and prefix generation."""

from .adapters import (
    AtomicRedTeamAdapter,
    MicrosoftIoTAdapter,
    ScriptRepoAdapter,
    ShellAttackEvolutionAdapter,
    ZenodoAdapter,
    adapter_for_kind,
)
from .augmentation import (
    AUGMENTATION_METHOD,
    AUGMENTATION_VERSION,
    generate_deterministic_stress,
)
from .dedup import cap_families, exact_deduplicate, family_aware_deduplicate
from .holdout import HOLDOUT_PERIOD, OOD_SLICE_NAME, is_temporal_holdout, record_period
from .io import read_jsonl, write_jsonl
from .normalization import clean_command, normalize_record, session_skeleton, skeletonize_command
from .pipeline import (
    prepare_sessions,
    prepare_sessions_v2,
    prepare_sessions_v2_with_stats,
    provenance_report,
)
from .prefixes import build_prefix_records, build_prefixes
from .sampling import allocate_quota, sample_training_quota
from .splitting import audit_split_leakage, family_aware_split

__all__ = [
    "AUGMENTATION_METHOD",
    "AUGMENTATION_VERSION",
    "AtomicRedTeamAdapter",
    "HOLDOUT_PERIOD",
    "OOD_SLICE_NAME",
    "MicrosoftIoTAdapter",
    "ScriptRepoAdapter",
    "ShellAttackEvolutionAdapter",
    "ZenodoAdapter",
    "adapter_for_kind",
    "allocate_quota",
    "audit_split_leakage",
    "build_prefix_records",
    "build_prefixes",
    "cap_families",
    "clean_command",
    "exact_deduplicate",
    "family_aware_deduplicate",
    "family_aware_split",
    "generate_deterministic_stress",
    "is_temporal_holdout",
    "normalize_record",
    "prepare_sessions",
    "prepare_sessions_v2",
    "prepare_sessions_v2_with_stats",
    "provenance_report",
    "read_jsonl",
    "record_period",
    "sample_training_quota",
    "session_skeleton",
    "skeletonize_command",
    "write_jsonl",
]
