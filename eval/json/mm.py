import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

_client: Optional[AzureOpenAI] = None
_cfg = {
    "endpoint": None,
    "api_key": None,
    "api_version": None,
    "chat_deployment": None,
    "embedding_deployment": None,
}

def _load_env():
    _cfg["endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    _cfg["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    _cfg["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    _cfg["chat_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    _cfg["embedding_deployment"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", _cfg["chat_deployment"])

def _get_client() -> AzureOpenAI:
    global _client
    if _client is not None:
        return _client

    _load_env()
    if not _cfg["endpoint"] or not _cfg["api_key"] or not _cfg["chat_deployment"]:
        raise RuntimeError(
            "Azure OpenAI client is not configured. "
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT in your .env file."
        )

    _client = AzureOpenAI(
        azure_endpoint=_cfg["endpoint"],
        api_key=_cfg["api_key"],
        api_version=_cfg["api_version"],
    )
    return _client

def chat(messages: List[Dict[str, Any]], temperature: float = 0.1, model: Optional[str] = None) -> str:
    client = _get_client()

    deployment = model or _cfg["chat_deployment"]

    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""

def get_embeddings(texts: List[str], model: Optional[str]=None) -> List[List[float]]:
    client = _get_client()
    deployment = model or _cfg['embedding_deployment']
    response = client.embeddings.create(
        input=texts,
        model=deployment
    )
    return [item.embedding for item in response.data]
