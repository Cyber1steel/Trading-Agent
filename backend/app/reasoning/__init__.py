"""Evidence-bound, non-executing LLM reasoning foundation."""

from app.reasoning.contracts import (
    EvidenceTier,
    ReasoningContext,
    ReasoningRequest,
    ReasoningResult,
    ReasoningStatus,
)
from app.reasoning.service import ReasoningService
from app.reasoning.gateway import ReasoningProviderGateway
from app.reasoning.provider import ProviderAvailability, ProviderCapabilities, ProviderReply, ReasoningProvider

__all__ = [
    "EvidenceTier",
    "ReasoningContext",
    "ReasoningRequest",
    "ReasoningResult",
    "ReasoningService",
    "ReasoningStatus",
    "ReasoningProvider",
    "ReasoningProviderGateway",
    "ProviderAvailability",
    "ProviderCapabilities",
    "ProviderReply",
]
