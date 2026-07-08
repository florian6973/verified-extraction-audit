"""Generic OpenAI-compatible chat backend (vLLM, llama.cpp, LocalAI, …).

Points at any server that exposes ``POST {base_url}/chat/completions`` with the
OpenAI schema — so a locally served model just needs its address/port:

    --api openai --api-base http://<host>:<port>/v1 --model <served-model-name>

``base_url`` / ``api_key`` also fall back to the ``OPENAI_API_BASE`` /
``OPENAI_API_KEY`` env vars. Uses ``requests`` (no extra dependency) and sends a
minimal, standards-compliant payload so it works across servers.
"""

import os

import requests
from loguru import logger


def call_openai_server(prompt, model="local", base_url=None, api_key=None,
                       temperature=0.0, max_tokens=512, note="<unspecified>", **kwargs):
    base_url = (base_url or os.environ.get("OPENAI_API_BASE") or "http://localhost:8000/v1").rstrip("/")
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    url = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        logger.info(f"[openai backend] note {note}, model {model}, {len(content)} chars from {url}")
        return content
    except Exception as e:
        logger.error(f"[openai backend] error calling {url}: {e}")
        return None
