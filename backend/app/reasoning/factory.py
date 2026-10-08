"""Application wiring for configured real reasoning providers."""

from app.core.config import Settings, get_settings
from app.reasoning.providers.openai import OpenAIReasoningProvider


def create_reasoning_provider(settings: Settings | None = None):
    configured = settings or get_settings()
    if configured.reasoning_provider == "openai":
        # Client and SDK loading are lazy; missing credentials are reported on generate().
        return OpenAIReasoningProvider(configured)
    raise ValueError("unsupported reasoning provider")
