from abc import ABC, abstractmethod
from typing import AsyncGenerator, List
from schemas import ChatMessage, LLMConfig, ModelResponse


class BaseClient(ABC):
    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Envía una lista de mensajes al modelo y devuelve su respuesta."""
        ...

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Genera la respuesta token a token (modo streaming)."""
        raise NotImplementedError
        yield 