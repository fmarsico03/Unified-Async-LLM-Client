from abc import ABC, abstractmethod

from schemas import ChatMessage, ModelResponse


class BaseClient(ABC):
    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Envía una lista de mensajes al modelo y devuelve su respuesta."""
        ...
