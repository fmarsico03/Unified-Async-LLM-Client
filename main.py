# main.py
import asyncio
import os

from dotenv import load_dotenv

from async_llm_manager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig, Provider

# (proveedor, env de la API key, env del modelo, modelo por defecto)
_PROVIDERS: list[tuple[Provider, str, str, str]] = [
    (Provider.OPENAI, "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    (Provider.ANTHROPIC, "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    (Provider.NVIDIA, "NVIDIA_API_KEY", "NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash"),
    (Provider.GEMINI, "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.5-flash"),
]


def build_config() -> LLMConfig:
    load_dotenv()
    for provider, key_env, model_env, default_model in _PROVIDERS:
        api_key = os.getenv(key_env)
        if api_key:
            return LLMConfig(
                provider=provider,
                model=os.getenv(model_env) or default_model,
                api_key=api_key,
                temperature=0.5,
                max_tokens=1500,
            )
    raise RuntimeError(
        "No se encontró ninguna API key"
    )


async def main() -> None:
    config = build_config()
    print(f"Usando proveedor: {config.provider.value} ({config.model})")
    llm = AsyncLLMManager(config)

    messages = [
        ChatMessage(role="system", content="Sos un asistente conciso."),
        ChatMessage(role="user", content="¿Qué es la entropía?"),
    ]

    response = await llm.generate(messages)
    if response.error:
        print(f"Falló: {response.error}")
    else:
        print(response.content)

    # 2) Streaming
    async for chunk in llm.generate_stream(messages):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        print(f"Error: {e}")
