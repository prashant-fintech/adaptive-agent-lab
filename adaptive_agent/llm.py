"""Thin client for any OpenAI-compatible chat endpoint.

No framework on top: one function returns text, one returns parsed JSON.
The defaults point at a local Ollama server running a Qwen model (see
config.py), but any OpenAI-compatible URL/key/model works.
"""

import json
import re

from openai import OpenAI

from adaptive_agent import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    response = get_client().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*$", re.MULTILINE)


def chat_json(system: str, user: str) -> dict:
    """chat(), then parse the reply as one JSON object.

    Strips <think> blocks (Qwen3 reasoning) and Markdown code fences before
    parsing, and re-raises with the raw reply attached so a malformed
    response is debuggable instead of a mystery.
    """
    raw = chat(system, user)
    cleaned = _CODE_FENCE.sub("", _THINK_BLOCK.sub("", raw)).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM reply contains no JSON object:\n{raw}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from exc
