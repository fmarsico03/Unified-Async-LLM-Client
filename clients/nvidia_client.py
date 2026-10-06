from clients.openai_client import OpenAIClient
from schemas import Provider


class NvidiaClient(OpenAIClient):
    """NVIDIA NIM expone una API compatible con OpenAI; solo cambia el endpoint."""
    provider = Provider.NVIDIA
    base_url = "https://integrate.api.nvidia.com/v1"
