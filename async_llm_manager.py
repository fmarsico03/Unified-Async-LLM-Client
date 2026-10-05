from collections.abc import AsyncIterator

from clients.anthropic_client import AnthropicClient
from clients.base_client import BaseClient
from clients.gemini_client import GeminiClient
from clients.openai_client import OpenAIClient
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider

_CLIENTS: dict[Provider, type[BaseClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}


class AsyncLLMManager:
    def __init__(self, config: LLMConfig) -> None:
        try:
            llm_client = _CLIENTS[config.provider]
        except KeyError:
            raise ValueError(f"Proveedor no soportado: {config.provider}") from None
        self._client: BaseClient = llm_client(config.api_key.get_secret_value(), config.model, config.temperature, config.max_tokens)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    def generate_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        return self._client.generate_stream(messages)