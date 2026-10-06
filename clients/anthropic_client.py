from anthropic import (
    AsyncAnthropic,
    NOT_GIVEN,
    APIError as AnthropicAPIError,
    RateLimitError as AnthropicRateLimitError,
    APIConnectionError as AnthropicConnectionError,
)
from schemas import ChatMessage, ModelResponse, Provider
from typing import AsyncGenerator, List, Tuple
from clients.base_client import BaseClient

class AnthropicClient(BaseClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _create_send_message(self, messages: List[ChatMessage]) -> Tuple[List[ChatMessage], str]:
        system_parts: List[str] = []
        messages_to_send: List[ChatMessage] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
            else:
                messages_to_send.append(message)
        system_prompt = "\n".join(system_parts)
        return messages_to_send, system_prompt

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        messages_to_send, system_prompt = self._create_send_message(messages)
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or NOT_GIVEN,
                messages=[m.model_dump() for m in messages_to_send],
            )
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content=response.content[0].text,
            )
        except AnthropicRateLimitError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Límite de cuota excedido: {e}")
        except AnthropicConnectionError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error de conexión: {e}")
        except AnthropicAPIError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error de la API de Anthropic: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        messages_to_send, system_prompt = self._create_send_message(messages)
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or NOT_GIVEN,
                messages=[m.model_dump() for m in messages_to_send],
            ) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"