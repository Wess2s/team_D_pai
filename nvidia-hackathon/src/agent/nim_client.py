"""
NIM client — thin wrapper over the OpenAI SDK pointed at a local NVIDIA NIM.

NIM exposes an OpenAI-compatible API, so this is identical to using the OpenAI
client except the base_url points at the DGX NIM and the api_key is unused locally.
"""
from __future__ import annotations

import os

from openai import OpenAI

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://0.0.0.0:8000/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")


def get_client() -> OpenAI:
    """Return an OpenAI client configured for the local NIM."""
    return OpenAI(base_url=NIM_BASE_URL, api_key="not-used-locally")


def chat(messages: list[dict], tools: list[dict] | None = None, **kwargs):
    """
    Single chat completion call. temperature defaults to 0.0 for deterministic
    tool selection in a control loop.
    """
    client = get_client()
    return client.chat.completions.create(
        model=NIM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else None,
        temperature=kwargs.pop("temperature", 0.0),
        **kwargs,
    )


if __name__ == "__main__":
    # Smoke test — run once NIM is up: python -m src.agent.nim_client
    resp = chat([{"role": "user", "content": "Reply with a single word: ready"}])
    print(resp.choices[0].message.content)
