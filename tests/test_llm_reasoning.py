"""Tests for the LLM blank-classifier, including reasoning ("thinking") models.

Reasoning models (DeepSeek-R1, Qwen3, …) served via vLLM / llama.cpp return their
chain-of-thought in a separate ``reasoning_content`` field or inline as
``<think>…</think>``, often leaving ``content`` empty. These tests pin down that
the backend recovers the answer from wherever it lands, and that the classifier
still parses a valid category map.

Offline tests (no network) always run. A live end-to-end test against a real
server is included but skipped unless you point it at one::

    VEAUDIT_LLM_BASE=http://10.116.2.38:8000/v1 VEAUDIT_LLM_MODEL='<served-model>' \
        python -m pytest tests/test_llm_reasoning.py -k live -s
"""

import os

import pytest

import src.llm.openai.utils as openai_utils
from src.llm.openai.utils import call_openai_server, extract_message_text
from src.dataset.prepare.inject import CATEGORIES, classify_llm, _parse_categories


def _msg(**fields):
    """Build a chat 'message' dict with only the given fields set."""
    return {k: v for k, v in fields.items() if v is not None}


# --------------------------------------------------------------------------- #
# extract_message_text: recover the answer across reasoning-model response shapes
# --------------------------------------------------------------------------- #
class TestExtractMessageText:
    def test_plain_content(self):
        assert extract_message_text(_msg(content='{"1": "name-patient"}')) == '{"1": "name-patient"}'

    def test_strips_inline_think_block(self):
        m = _msg(content='<think>blank 1 follows "Name:"</think>{"1": "name-patient"}')
        assert extract_message_text(m) == '{"1": "name-patient"}'

    def test_multiline_think_block(self):
        m = _msg(content="<think>\nlong\nreasoning\n</think>\n{\"1\": \"id\"}")
        assert extract_message_text(m) == '{"1": "id"}'

    def test_falls_back_to_reasoning_content_when_content_empty(self):
        m = _msg(content="", reasoning_content='reasoning... {"1": "name-patient"}')
        assert '{"1": "name-patient"}' in extract_message_text(m)

    def test_falls_back_to_reasoning_when_content_missing(self):
        assert extract_message_text(_msg(reasoning='{"1": "id"}')) == '{"1": "id"}'

    def test_content_wins_over_reasoning(self):
        m = _msg(content='{"1": "name-patient"}', reasoning_content='{"1": "id"}')
        assert extract_message_text(m) == '{"1": "name-patient"}'

    def test_content_as_list_of_parts(self):
        # OpenAI content-parts schema: content is a list, not a str.
        m = _msg(content=[{"type": "text", "text": '{"1": "name-patient"}'}])
        assert extract_message_text(m) == '{"1": "name-patient"}'

    def test_whitespace_or_think_only_content_falls_back_to_reasoning(self):
        assert extract_message_text(_msg(content="  \n\t ", reasoning_content='{"1": "id"}')) == '{"1": "id"}'
        assert extract_message_text(
            _msg(content="<think>reasoning</think>", reasoning_content='{"1": "id"}')) == '{"1": "id"}'

    def test_unclosed_think_is_left_for_downstream_recovery(self):
        # A truncated (unclosed) <think> is NOT stripped: the answer may sit just
        # before the cutoff, and json_repair still recovers it in _parse_categories.
        m = _msg(content='<think>ran out of tokens mid-thought\n{"1": "name-patient"}')
        text = extract_message_text(m)
        assert text.startswith("<think>")
        assert _parse_categories(text, 1) == {"1": "name-patient"}

    def test_empty_everything_returns_empty_string(self):
        assert extract_message_text(_msg(content="")) == ""
        assert extract_message_text({}) == ""
        assert extract_message_text(None) == ""


# --------------------------------------------------------------------------- #
# _parse_categories: the extracted text must still yield a valid category map
# --------------------------------------------------------------------------- #
class TestParseCategories:
    def test_parses_reasoning_model_answer(self):
        text = extract_message_text(_msg(content="", reasoning_content=(
            "<think>lengthy reasoning that never closes properly "
            'Final: {"1": "name-patient", "2": "id", "3": "name-attending"}')))
        assert _parse_categories(text, 3) == {"1": "name-patient", "2": "id", "3": "name-attending"}

    def test_parses_fenced_json(self):
        text = '```json\n{"1": "name-patient", "2": "id"}\n```'
        assert _parse_categories(text, 2) == {"1": "name-patient", "2": "id"}

    def test_two_separate_json_objects_are_merged(self):
        # json_repair returns a LIST here; the HIGH bug was mapping all to 'other'.
        assert _parse_categories('{"1": "id"}\n{"2": "name-patient"}', 2) == {"1": "id", "2": "name-patient"}

    def test_example_then_final_answer_last_wins(self):
        # Reasoning model restates the prompt's example, then gives the real answer.
        text = ('The example was {"1": "name-patient", "2": "id"}. '
                'Final answer: {"1": "name-patient", "2": "id", "3": "name-attending"}')
        assert _parse_categories(text, 3) == {"1": "name-patient", "2": "id", "3": "name-attending"}

    def test_multiple_fenced_blocks_takes_last(self):
        text = '```json\n{"1": "other"}\n```\nrevised:\n```json\n{"1": "name-patient"}\n```'
        assert _parse_categories(text, 1) == {"1": "name-patient"}

    def test_stray_trailing_fence_does_not_drop_answer(self):
        # A lone ``` after the answer must not cause the JSON to be discarded.
        assert _parse_categories('{"1": "name-patient", "2": "id"}\n```\nfoo', 2) == \
            {"1": "name-patient", "2": "id"}

    def test_reasoning_prose_without_json_defaults_to_other(self):
        text = extract_message_text(_msg(content="", reasoning_content="blank one is probably a name, unsure."))
        assert _parse_categories(text, 2) == {"1": "other", "2": "other"}

    def test_unknown_category_becomes_other(self):
        assert _parse_categories('{"1": "social-security"}', 1) == {"1": "other"}

    def test_empty_defaults_all_to_other(self):
        assert _parse_categories("", 3) == {"1": "other", "2": "other", "3": "other"}
        assert _parse_categories(None, 2) == {"1": "other", "2": "other"}


# --------------------------------------------------------------------------- #
# call_openai_server: request shaping + response handling, with requests mocked
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, message):
        self._message = message

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": self._message}]}


class TestCallOpenAIServerMocked:
    def _patch(self, monkeypatch, message):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResp(message)

        monkeypatch.setattr(openai_utils.requests, "post", fake_post)
        return captured

    def test_reads_reasoning_content(self, monkeypatch):
        captured = self._patch(monkeypatch, {"content": "", "reasoning_content": '{"1": "name-patient"}'})
        out = call_openai_server("classify", model="r1", base_url="http://x/v1", max_tokens=256)
        assert '{"1": "name-patient"}' in out
        assert captured["payload"]["max_tokens"] == 256

    def test_no_think_shapes_payload(self, monkeypatch):
        captured = self._patch(monkeypatch, {"content": '{"1": "id"}'})
        out = call_openai_server("classify", base_url="http://x/v1", no_think=True)
        assert out == '{"1": "id"}'
        # /no_think hint appended to the user message
        assert captured["payload"]["messages"][0]["content"].endswith("/no_think")
        # thinking disabled via BOTH template keys (Qwen3 enable_thinking, DeepSeek thinking)
        assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False, "thinking": False}

    def test_no_think_does_not_override_user_chat_template_kwargs(self, monkeypatch):
        # A caller that supplies chat_template_kwargs owns it; no_think must not clobber it.
        captured = self._patch(monkeypatch, {"content": "{}"})
        call_openai_server("classify", base_url="http://x/v1", no_think=True,
                           extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}   # user wins
        assert captured["payload"]["messages"][0]["content"].endswith("/no_think")        # hint still sent

    def test_extra_body_passthrough(self, monkeypatch):
        captured = self._patch(monkeypatch, {"content": "{}"})
        call_openai_server("x", base_url="http://x/v1", extra_body={"top_p": 0.5})
        assert captured["payload"]["top_p"] == 0.5

    def test_content_as_list_of_parts(self, monkeypatch):
        # Some servers/proxies return content as OpenAI content-parts, not a str.
        captured = self._patch(monkeypatch, {"content": [{"type": "text", "text": '{"1": "name-patient"}'}]})
        out = call_openai_server("x", base_url="http://x/v1")
        assert out == '{"1": "name-patient"}'

    def test_empty_content_and_reasoning_returns_empty_string_not_none(self, monkeypatch):
        # Budget exhausted on thinking -> "" (distinct from None on transport failure).
        self._patch(monkeypatch, {"content": "", "reasoning_content": ""})
        assert call_openai_server("x", base_url="http://x/v1", max_tokens=8) == ""

    def test_network_error_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(openai_utils.requests, "post", boom)
        assert call_openai_server("x", base_url="http://x/v1") is None


# --------------------------------------------------------------------------- #
# classify_llm end-to-end via the deterministic offline 'mock' backend
# --------------------------------------------------------------------------- #
class TestClassifyLLMMock:
    def test_mock_backend_classifies_blanks_by_label(self):
        note = "Name: ___  Unit No: ___  Attending: ___"
        cats = classify_llm(note, "n1", "mock", {"model": "mock"})
        assert set(cats) == {"1", "2", "3"}
        assert all(c in CATEGORIES for c in cats.values())
        assert cats["1"] == "name-patient"   # Name:
        assert cats["2"] == "id"             # Unit No: (MRN alias)
        assert cats["3"] == "name-attending" # Attending:

    def test_classify_llm_with_mocked_openai(self, monkeypatch):
        # A reasoning model that only ever fills reasoning_content must still work.
        msg = {"content": "", "reasoning_content": '{"1": "name-patient", "2": "id"}'}
        monkeypatch.setattr(openai_utils.requests, "post",
                            lambda *a, **k: _FakeResp(msg))
        cats = classify_llm("Name: ___  Unit No: ___", "n2", "openai",
                            {"model": "r1", "base_url": "http://x/v1", "no_think": True})
        assert cats == {"1": "name-patient", "2": "id"}


# --------------------------------------------------------------------------- #
# Live end-to-end (opt-in): hit a real OpenAI-compatible server.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.environ.get("VEAUDIT_LLM_BASE"),
                    reason="set VEAUDIT_LLM_BASE (+ optional VEAUDIT_LLM_MODEL) to test a live server")
def test_live_server_classifies_a_note():
    base = os.environ["VEAUDIT_LLM_BASE"]
    model = os.environ.get("VEAUDIT_LLM_MODEL", "local")
    note = "Name: ___  Unit No: ___  Attending: ___  Discharge Date: ___"
    cats = classify_llm(note, "live", "openai",
                        {"model": model, "base_url": base, "no_think": True, "max_tokens": 2048})
    assert cats, "classifier returned no categories"
    assert all(c in CATEGORIES for c in cats.values())
    # A working model should recognise at least the Name: blank, not label everything 'other'.
    assert any(v != "other" for v in cats.values()), f"server returned only 'other': {cats}"
