"""Provider-neutral LLM boundary; adapters return raw JSON for deterministic validation."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from app.reasoning.prompt import PromptEnvelope
from app.reasoning.fingerprints import PROMPT_CONTRACT_VERSION


class ProviderAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DISABLED = "DISABLED"


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider_id: str
    model_id: str
    structured_output: bool
    max_input_bytes: int
    max_output_tokens: int
    streaming: bool
    supported_contract_versions: tuple[str, ...]
    availability: ProviderAvailability

    def __post_init__(self):
        if not self.provider_id or not self.model_id:
            raise ValueError("provider and model identifiers are required")
        if self.max_input_bytes <= 0 or self.max_output_tokens <= 0:
            raise ValueError("provider input/output limits must be positive")
        if tuple(sorted(set(self.supported_contract_versions))) != self.supported_contract_versions:
            raise ValueError("supported contract versions must be sorted and unique")


@dataclass(frozen=True, slots=True)
class ProviderReply:
    provider_id: str
    model_id: str
    raw_response: str
    request_metadata: tuple[tuple[str, str | int | bool | None], ...] = ()
    telemetry: tuple[tuple[str, int | float | str | bool | None], ...] = ()


class ReasoningProvider(Protocol):
    provider_id: str
    model_id: str

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Explicit capabilities declared by this configured provider adapter."""

    def generate(self, prompt: PromptEnvelope) -> ProviderReply:
        """Return provider output without treating it as validated or authoritative."""


def prompt_input_bytes(prompt: PromptEnvelope) -> int:
    """Measure complete, canonical prompt material, including the output schema."""
    return len(prompt.deterministic_payload().encode("utf-8"))
