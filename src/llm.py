"""Thin wrapper around the Gemini API (free tier)."""

import os

from google import genai

MODEL_NAME = "gemini-flash-lite-latest"

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


if __name__ == "__main__":
    print(generate("Say hello in one short sentence."))
