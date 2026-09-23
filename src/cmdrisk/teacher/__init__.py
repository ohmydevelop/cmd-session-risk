"""Teacher labeling, augmentation, and blind proxy-audit utilities."""

from cmdrisk.teacher.audit import (
    AuditCase,
    ProxyAuditMetrics,
    ProxyAuditor,
    ProxyAuditReport,
    ProxyAuditResult,
)
from cmdrisk.teacher.augmentation import (
    AugmentationError,
    AugmentationResult,
    HardNegativeGenerator,
    HardNegativeResult,
    InsufficientVariantsError,
    MaliciousAugmenter,
)
from cmdrisk.teacher.client import (
    ChatAPIError,
    ConfigurationError,
    CostLedger,
    CostLimitExceeded,
    DryRunError,
    JsonChatClient,
    ModelConfig,
    OpenAICompatibleClient,
    SchemaValidationError,
    TeacherError,
)
from cmdrisk.teacher.labeling import ConsensusDecision, ConsensusLabeler
from cmdrisk.teacher.prompts import PromptedModel, PromptError, PromptStore
from cmdrisk.teacher.responses import (
    HardNegativeBatchResponse,
    HardNegativeValidationResponse,
    MaliciousValidationResponse,
    ProxyReviewResponse,
    RewriteBatchResponse,
    RewriteVariant,
    TeacherLabelResponse,
)

__all__ = [
    "AuditCase",
    "AugmentationError",
    "AugmentationResult",
    "ChatAPIError",
    "ConfigurationError",
    "ConsensusDecision",
    "ConsensusLabeler",
    "CostLedger",
    "CostLimitExceeded",
    "DryRunError",
    "HardNegativeBatchResponse",
    "HardNegativeGenerator",
    "HardNegativeResult",
    "HardNegativeValidationResponse",
    "InsufficientVariantsError",
    "JsonChatClient",
    "MaliciousAugmenter",
    "MaliciousValidationResponse",
    "ModelConfig",
    "OpenAICompatibleClient",
    "PromptError",
    "PromptStore",
    "PromptedModel",
    "ProxyAuditMetrics",
    "ProxyAuditReport",
    "ProxyAuditResult",
    "ProxyAuditor",
    "ProxyReviewResponse",
    "RewriteBatchResponse",
    "RewriteVariant",
    "SchemaValidationError",
    "TeacherError",
    "TeacherLabelResponse",
]
