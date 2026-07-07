"""Offline, deterministic LLM backend.

This backend implements the same ``call_llm`` contract as the Gemini and vLLM
backends but performs **no network or GPU calls**. It returns deterministic,
well-formed JSON so that the PII-injection code path
(:func:`src.llm.call_llm`) can be exercised end-to-end on a
machine with no API access — for example in a smoke test on a fresh cluster node.

It is intended for *testing the code path*, not for producing a realistic audit:
the filled values are placeholders (``value_1``, ``value_2``, ...), not coherent
clinical text. For a fully deterministic injection that fills blanks with real
synthetic-persona identifiers, use
``python -m src.dataset.prepare.inject --classifier label`` (no LLM) instead.
"""

import json
import re

from loguru import logger

try:  # json_repair is a hard dependency, but keep the import defensive.
    from json_repair import json_repair

    def _loads(text):
        return json_repair.loads(text)
except Exception:  # pragma: no cover - fallback to stdlib
    def _loads(text):
        return json.loads(text)


def _extract_placeholder_keys(prompt):
    """Return the ``[k]`` placeholder numbers embedded in a ``generate_text`` prompt."""
    return re.findall(r"\[(\d+)\]", prompt)


def _extract_embedded_json(prompt):
    """Pull the ```json ...``` block out of a ``classify_json`` prompt, best effort."""
    if "```json" in prompt:
        block = prompt.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in prompt:
        block = prompt.split("```", 1)[1].split("```", 1)[0]
    else:
        match = re.search(r"\{.*\}", prompt, re.DOTALL)
        block = match.group(0) if match else "{}"
    try:
        obj = _loads(block.strip())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _guess_category(value):
    """Cheap heuristic tagging so ``--check-output`` produces plausible tags."""
    text = str(value)
    if "@" in text:
        return "contact"
    if re.search(r"\d", text) and not re.search(r"[A-Za-z]{3,}", text):
        return "id"
    if len(text.split()) == 2 and all(part[:1].isalpha() for part in text.split()):
        return "name-patient"
    return "other"


def call_mock(prompt, model=None, task=None, note=None, **kwargs):
    """Deterministic stand-in for ``call_llm``.

    Parameters mirror the other backends (``model``/``task``/``note`` come from the
    ``kwargs`` dict passed at the call site). Always returns a fenced ```json block
    so the lenient ``parse_json`` in the injection step can consume it.
    """
    if task == "generate_text":
        keys = _extract_placeholder_keys(prompt)
        payload = {k: f"value_{k}" for k in keys}
    elif task == "classify_json":
        embedded = _extract_embedded_json(prompt)
        payload = {k: _guess_category(v) for k, v in embedded.items()}
    elif task == "classify_blanks":
        # Classify each [k] blank by the note label preceding it (deterministic).
        from src.dataset.prepare.di_types import detect_di_type, get_di_type
        default = get_di_type("name")
        payload = {m.group(1): detect_di_type(prompt[:m.start()], default).category
                   for m in re.finditer(r"\[(\d+)\]", prompt)}
    else:  # Unknown task: echo an empty object rather than crash the caller.
        payload = {}

    logger.info(f"[mock backend] note {note}, task {task}, {len(payload)} keys")
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
