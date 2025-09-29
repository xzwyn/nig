# test_azure_openai.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os, sys

def mask(s: str, keep: int = 4) -> str:
    if not s: return ""
    return s[:keep] + "..." if len(s) > keep else "****"

def print_config():
    print("Azure OpenAI configuration:")
    print(f"  AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"  AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    print(f"  AZURE_OPENAI_DEPLOYMENT (chat): {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")
    print(f"  AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: {os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT') or '(not set)'}")
    print(f"  AZURE_OPENAI_API_KEY: {mask(os.getenv('AZURE_OPENAI_API_KEY'))}\n")

def test_chat():
    from azure_client import chat
    print("Running chat test...")
    try:
        reply = chat([{"role": "user", "content": "Reply with the single word: ok"}], temperature=0.0)
        print("Chat response:", repr(reply))
        if reply.strip().lower().startswith("ok"):
            print("Chat test: PASS"); return True
        print("Chat test: WARN (unexpected content)"); return False
    except Exception as e:
        print("Chat test: FAIL"); print("Error:", e); return False

def test_embeddings():
    emb = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    if not emb:
        print("Embeddings test skipped (AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT not set)."); return None
    try:
        from azure_client import embed
        import numpy as np
        vecs = embed(["hello world", "guten tag"])
        print("Embeddings shape:", vecs.shape)
        print("Norms:", np.linalg.norm(vecs, axis=1).tolist())
        print("Embeddings test: PASS"); return True
    except Exception as e:
        print("Embeddings test: FAIL"); print("Error:", e); return False

def main():
    print_config()
    ok_chat = test_chat()
    ok_emb = test_embeddings()
    if ok_chat and (ok_emb in (True, None)):
        print("\nOverall: PASS"); sys.exit(0)
    print("\nOverall: FAIL"); sys.exit(1)

if __name__ == "__main__":
    main()
