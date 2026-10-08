"""Provider-neutral LLM boundary; adapters return raw JSON for deterministic validation."""

from dataclasses import dataclass
from typing import Protocol

from app.reasoning.prompt import PromptEnvelope


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

    def generate(self, prompt: PromptEnvelope) -> ProviderReply:
        """Return provider output without treating it as validated or authoritative."""
