"""Thin wrapper around the Gemini API (free tier) -- generation and
embeddings both via the API, deliberately avoiding a local embedding model
(sentence-transformers/torch). This app is deployed on Render's free tier
(512MB RAM); torch alone is memory-heavy enough to cause OOM kills there
mid-request. Using Gemini for embeddings too, instead of running inference
locally, keeps the deployed app's memory footprint small."""

import os

import numpy as np
from google import genai

MODEL_NAME = "gemini-flash-lite-latest"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable.")
        _client = genai.Client(api_key=api_key)
    return _client


def generate(prompt: str, model: str = MODEL_NAME) -> str:
    client = _get_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Returns an (len(texts), dim) array of L2-normalized embeddings."""
    client = _get_client()
    result = client.models.embed_content(model=model, contents=texts)
    vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


if __name__ == "__main__":
    print(generate("Say hello in one short sentence."))
