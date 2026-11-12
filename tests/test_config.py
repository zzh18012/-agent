from mmos.config import Settings
from mmos.config.settings import LLMProviderName


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider is LLMProviderName.DEEPSEEK
    assert settings.llm_model == "deepseek-chat"
    assert settings.log_level == "INFO"


def test_settings_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("MMOS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MMOS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MMOS_LLM_API_KEY", "secret-value")

    settings = Settings(_env_file=None)
    assert settings.llm_provider is LLMProviderName.OPENAI
    assert settings.log_level == "DEBUG"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "secret-value"
