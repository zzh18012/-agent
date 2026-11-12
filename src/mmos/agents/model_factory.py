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

    if settings.llm_provider is LLMProviderName.DEEPSEEK:
        if settings.llm_base_url:
            provider = DeepSeekProvider(
                openai_client=AsyncOpenAI(api_key=api_key, base_url=settings.llm_base_url)
            )
        else:
            provider = DeepSeekProvider(api_key=api_key)
        return OpenAIChatModel(settings.llm_model, provider=provider)

    if settings.llm_provider is LLMProviderName.OPENAI:
        provider = OpenAIProvider(api_key=api_key, base_url=settings.llm_base_url)
        return OpenAIChatModel(settings.llm_model, provider=provider)

    raise UnsupportedProviderError(
        f"provider {settings.llm_provider.value!r} is not installed in this build; "
        "use 'deepseek' or 'openai'"
    )
