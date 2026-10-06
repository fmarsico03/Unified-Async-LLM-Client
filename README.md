# Unified Async LLM Client

Cliente asíncrono en Python que unifica el acceso a distintos proveedores de LLM (OpenAI, Anthropic, Google Gemini y NVIDIA NIM) detrás de una única interfaz.
## Características

- **Interfaz única** (`AsyncLLMManager`) para todos los proveedores.
- **Asíncrono** (`asyncio`, clientes `Async*` de cada SDK).
- **Respuesta completa o streaming** token a token.
- **Validación con Pydantic** de mensajes, configuración y respuestas.
- **Manejo de errores uniforme**: los errores de cuota, conexión o API se devuelven en `ModelResponse.error`.
- **API keys protegidas** con `SecretStr`.

## Proveedores soportados

| Proveedor | Variables de entorno | Modelo por defecto |
|-----------|----------------------|--------------------|
| OpenAI    | `OPENAI_API_KEY`, `OPENAI_MODEL`       | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | `claude-sonnet-4-5` |
| NVIDIA    | `NVIDIA_API_KEY`, `NVIDIA_MODEL`       | `deepseek-ai/deepseek-v4.1-flash` |
| Gemini    | `GEMINI_API_KEY`, `GEMINI_MODEL`       | `gemini-3.5-flash` |

NVIDIA NIM expone una API compatible con OpenAI, por lo que `NvidiaClient` reutiliza `OpenAIClient` cambiando solo el `base_url`.

## Estructura del proyecto

```
.
├── main.py                 # Ejemplo de uso (generate + streaming)
├── async_llm_manager.py    # Fachada que elige el cliente según el proveedor
├── schemas.py              # Modelos Pydantic: ChatMessage, LLMConfig, ModelResponse, Provider
├── clients/
│   ├── base_client.py      # Clase abstracta BaseClient
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── gemini_client.py
│   └── nvidia_client.py
├── requirements.txt
└── .env.example
```

## Instalación

Requiere Python 3.12.

```bash
# Windows
py -3.12 -m venv venv
venv\Scripts\activate
# Linux / macOS
python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Configuración

Copiá el archivo de ejemplo, elegí el proveedor con `LLM_PROVIDER` y completá su API key:

```bash
cp .env.example .env
```

```env
LLM_PROVIDER=openai    # openai | anthropic | gemini | nvidia
OPENAI_API_KEY=sk-...
OPENAI_MODEL=          # opcional, si queda vacío se usa el modelo por defecto
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=
GEMINI_API_KEY=
GEMINI_MODEL=
NVIDIA_API_KEY=
NVIDIA_MODEL=
```

`main.py` usa el proveedor indicado en `LLM_PROVIDER` (por defecto `openai`). Si el valor no es válido o falta la API key de ese proveedor, el script termina con un mensaje de error claro. Para cambiar de proveedor alcanza con modificar esa línea del `.env`.

Envía una pregunta al modelo, imprime la respuesta completa y luego la vuelve a pedir en modo streaming.

## Agregar un nuevo proveedor

1. Agregar el valor al enum `Provider` en [schemas.py](schemas.py).
2. Crear un cliente en `clients/` que herede de `BaseClient` e implemente `generate` y `generate_stream`.
   Si el proveedor es compatible con la API de OpenAI, alcanza con heredar de `OpenAIClient` y definir `provider` y `base_url` (como hace `NvidiaClient`).
3. Registrarlo en el diccionario `_CLIENTS` de [async_llm_manager.py](async_llm_manager.py).
4. (Opcional) Agregar sus variables a `.env.example` y a `_PROVIDERS` en [main.py](main.py).
