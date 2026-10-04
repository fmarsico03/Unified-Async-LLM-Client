from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, SecretStr


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelResponse(BaseModel):
    content: str
    model: str
    provider: str
    error: Optional[str] = None

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class LLMConfig(BaseModel):
    provider: Provider
    model: str
    api_key: Optional[SecretStr] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
