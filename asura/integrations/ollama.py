"""Opt-in access to Ollama's OpenAI-compatible hosted chat API."""

from collections.abc import Mapping, Sequence
import math
import os

OLLAMA_BASE_URL = "https://ollama.com/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
ALLOWED_DATA_POLICIES = {"public", "synthetic"}


def _validate_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise ValueError("messages must be a non-empty sequence of role/content mappings")

    safe_messages = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("each message must contain only role and content")
        role, content = message["role"], message["content"]
        if (
            not isinstance(role, str)
            or role not in {"system", "user", "assistant"}
            or not isinstance(content, str)
        ):
            raise ValueError("messages require a supported role and string content")
        safe_messages.append({"role": role, "content": content})
    return safe_messages


def _create_client(api_key: str | None = None):
    key = api_key or os.environ.get("OLLAMA_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError(
            "Set OLLAMA_API_KEY in your environment; create a key at "
            "https://ollama.com/settings/keys"
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Ollama API access requires the optional 'openai' package; "
            "install requirements-integrations.txt"
        ) from exc

    return OpenAI(api_key=key, base_url=OLLAMA_BASE_URL)


def chat_completion(
    messages: Sequence[Mapping[str, str]],
    *,
    data_policy: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.2,
    api_key: str | None = None,
) -> str:
    """Request a completion, restricted to explicitly public or synthetic inputs."""
    if not isinstance(data_policy, str) or data_policy not in ALLOWED_DATA_POLICIES:
        raise ValueError("Ollama requests require data_policy='public' or 'synthetic'")
    safe_messages = _validate_messages(messages)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature must be a finite number between 0 and 2")

    client = _create_client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=safe_messages,
        max_tokens=max_tokens,
        temperature=float(temperature),
    )
    choices = response.choices
    content = choices[0].message.content if choices else None
    if not isinstance(content, str):
        raise RuntimeError("Ollama returned no text completion")
    return content
