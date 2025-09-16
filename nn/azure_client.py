# azure_client.py
import os
from typing import List, Dict, Any, Optional
import numpy as np
from openai import AzureOpenAI

_client: Optional[AzureOpenAI] = None
_cfg = {
    "endpoint": None,
    "api_key": None,
    "api_version": None,
    "chat_deployment": None,
    "embed_deployment": None,
}

def _load_env():
    _cfg["endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://mcphack-oai-01-swedencentral.openai.azure.com
    _cfg["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    _cfg["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    _cfg["chat_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    _cfg["embed_deployment"] = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")  # optional

def _get_client() -> AzureOpenAI:
    global _client
    if _client is not None:
        return _client
    _load_env()
    if not _cfg["endpoint"] or not _cfg["api_key"]:
        raise RuntimeError("Azure OpenAI client is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    _client = AzureOpenAI(
        azure_endpoint=_cfg["endpoint"],
        api_key=_cfg["api_key"],
        api_version=_cfg["api_version"],
    )
    return _client

def chat(messages: List[Dict[str, Any]], temperature: float = 0.1, model: Optional[str] = None) -> str:
    client = _get_client()
    deployment = model or _cfg["chat_deployment"] or "gpt-4o"
    resp = client.chat.completions.create(
        model=deployment,  # must be your deployment name
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""

def embed(texts: List[str], model: Optional[str] = None, batch_size: int = 64) -> np.ndarray:
    client = _get_client()
    deployment = model or _cfg["embed_deployment"]
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is not set. Create an embeddings deployment or pass model=.")
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=deployment, input=chunk)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)
