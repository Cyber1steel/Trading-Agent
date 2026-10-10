"""Application wiring for the configured reasoning provider gateway."""

from app.core.config import Settings, get_settings
from app.reasoning.gateway import ReasoningProviderGateway
from app.reasoning.providers.groq import GroqReasoningProvider
from app.reasoning.providers.openai import OpenAIReasoningProvider


def create_reasoning_gateway(settings: Settings | None = None, *, provider_order=None):
    configured = settings or get_settings()
    order = tuple(provider_order or configured.reasoning_provider_order.split(","))
    providers = {
        "groq": GroqReasoningProvider(configured),
        "openai": OpenAIReasoningProvider(configured),
    }
    return ReasoningProviderGateway(
        providers,
        order,
        required_max_output_tokens=configured.reasoning_max_output_tokens,
    )


def create_reasoning_provider(settings: Settings | None = None):
    """Backward-compatible factory name; returns the provider-independent gateway."""
    return create_reasoning_gateway(settings)
