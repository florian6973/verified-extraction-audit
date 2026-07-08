"""Generic OpenAI-compatible chat backend (vLLM, llama.cpp, LocalAI, …).

Points at any server that exposes ``POST {base_url}/chat/completions`` with the
OpenAI schema — so a locally served model just needs its address/port:

    --api openai --api-base http://<host>:<port>/v1 --model <served-model-name>

``base_url`` / ``api_key`` also fall back to the ``OPENAI_API_BASE`` /
``OPENAI_API_KEY`` env vars. Uses ``requests`` (no extra dependency) and sends a
minimal, standards-compliant payload so it works across servers.

Reasoning models
----------------
DeepSeek-R1 / Qwen3-style "thinking" models (served via vLLM or llama.cpp) return
their chain-of-thought either in a separate ``reasoning_content`` field (leaving
``content`` empty) or inline as ``<think>…</think>`` inside ``content`` — and, if
the token budget is too small, they can spend it all thinking and emit no final
answer at all. :func:`extract_message_text` recovers the answer from whichever
field holds it, ``max_tokens`` defaults high enough to leave room for the answer,
and ``no_think=True`` asks the server to disable thinking.

``no_think`` sends the Qwen3 ``/no_think`` hint plus ``chat_template_kwargs``
``{enable_thinking: false, thinking: false}`` (the two keys used by the Qwen3 and
DeepSeek/Granite template families). Caveats: llama.cpp only honors
``chat_template_kwargs`` when started with ``--jinja`` (otherwise disable thinking
server-side via ``--reasoning-budget 0``); pure DeepSeek-R1 cannot be disabled at
all. A caller that passes its own ``chat_template_kwargs`` via ``extra_body`` owns
it entirely (``no_think`` won't override it).
"""

import os
import re

import requests
from loguru import logger

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _as_text(value):
    """Coerce a chat ``content`` / ``reasoning`` field to a string.

    Most servers return a plain string, but some emit the OpenAI content-parts
    schema — a list like ``[{"type": "text", "text": "..."}]``. Join the text
    parts; anything else (None, number) becomes ``""``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in value)
    return ""


def _strip_think(text):
    """Drop any balanced ``<think>…</think>`` reasoning block inlined in content.

    An *unclosed* ``<think>`` (a truncated response) is intentionally left in
    place: the answer may sit just before the cutoff, and ``json_repair``
    downstream still recovers it — whereas blindly deleting to end-of-string
    would throw that answer away.
    """
    return _THINK_RE.sub("", _as_text(text)).strip()


def extract_message_text(message):
    """Usable text from an OpenAI chat ``message``, tolerant of reasoning models.

    Prefer ``content`` (with any ``<think>`` block stripped); if that is empty —
    the common reasoning-model case — fall back to ``reasoning_content`` then
    ``reasoning`` (vLLM renamed the field; both are checked). Downstream JSON
    parsing (``json_repair``) still recovers the answer amid reasoning prose.
    """
    if not isinstance(message, dict):
        return ""
    content = _strip_think(message.get("content"))
    if content:
        return content
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return _strip_think(reasoning)


def call_openai_server(prompt, model="local", base_url=None, api_key=None,
                       temperature=0.0, max_tokens=1024, note="<unspecified>",
                       extra_body=None, no_think=False, **kwargs):
    base_url = (base_url or os.environ.get("OPENAI_API_BASE") or "http://localhost:8000/v1").rstrip("/")
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    url = f"{base_url}/chat/completions"

    if no_think:
        prompt = prompt + " /no_think"   # Qwen3 convention; a no-op for other models

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Pass-through for server-specific knobs, e.g. disabling thinking on vLLM /
    # llama.cpp: extra_body={"chat_template_kwargs": {"enable_thinking": False}}.
    body_extra = dict(extra_body or {})
    # no_think fills in the disable-thinking template kwargs only if the caller
    # did not supply their own chat_template_kwargs (they own it if they did).
    # Both keys are set: Qwen3 reads `enable_thinking`, DeepSeek/Granite `thinking`.
    if no_think and "chat_template_kwargs" not in body_extra:
        body_extra["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
    payload.update(body_extra)

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        text = extract_message_text(message)
        src = "content" if _strip_think(message.get("content") or "") else "reasoning"
        logger.info(f"[openai backend] note {note}, model {model}, {len(text)} chars ({src}) from {url}")
        if not text:
            logger.warning(
                f"[openai backend] note {note}: empty content AND reasoning — the model likely "
                f"spent all {max_tokens} tokens thinking. Raise max_tokens or pass no_think=True."
            )
        return text
    except Exception as e:
        logger.error(f"[openai backend] error calling {url}: {e}")
        return None
