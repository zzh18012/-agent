from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider

from mmos.config.settings import LLMProviderName, Settings


class MissingAPIKeyError(RuntimeError):
    pass


class UnsupportedProviderError(RuntimeError):
    pass


def build_model(settings: Settings) -> Model:
    """Build the configured chat model. Construction performs no network I/O."""
    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value():
        raise MissingAPIKeyError(
            "MMOS_LLM_API_KEY is not configured; copy .env.example to .env and set it"
        )
    api_key = settings.llm_api_key.get_secret_value()

    # A bounded request timeout prevents a single stalled HTTP call from hanging a
    # pipeline indefinitely; provider retries handle transient errors.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )

    if settings.llm_provider is LLMProviderName.DEEPSEEK:
        return OpenAIChatModel(settings.llm_model, provider=DeepSeekProvider(openai_client=client))

    if settings.llm_provider is LLMProviderName.OPENAI:
        return OpenAIChatModel(settings.llm_model, provider=OpenAIProvider(openai_client=client))

    raise UnsupportedProviderError(
        f"provider {settings.llm_provider.value!r} is not installed in this build; "
        "use 'deepseek' or 'openai'"
    )
