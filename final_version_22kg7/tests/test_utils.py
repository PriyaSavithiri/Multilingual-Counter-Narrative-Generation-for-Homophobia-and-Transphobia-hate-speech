"""
Regression coverage for a real bug found via live testing: for one real
LLM-judge call, qwen2.5:7b-instruct returned a flat JSON ARRAY
(["relevance", 4, "factuality", 3, ...]) instead of a JSON object.
json.loads() parses a JSON array just fine, and the caller's schema check
(`k in parsed for k in response_schema`) uses Python's `in` operator, which
does *membership* on a list rather than key lookup - since the array
happened to contain all the expected field-name strings as elements, the
check passed by accident. This silently corrupted downstream code that
assumed a dict: pandas turned the list into positionally-indexed
judge_0/judge_1/... columns instead of named ones, showing up as a wall of
None values in the Streamlit "Model performance across languages" table for
every row that didn't happen to share that exact failure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import parse_json_object


def test_parse_json_object_rejects_a_top_level_json_array():
    # The exact shape observed live: field names and scores flattened into
    # one array instead of a proper {"relevance": 4, "factuality": 3, ...} object.
    raw = '["relevance", 4, "factuality", 3, "empathy", 3]'
    assert parse_json_object(raw) == {}


def test_parse_json_object_rejects_an_array_embedded_in_prose():
    raw = 'Here are the scores: ["relevance", 4, "factuality", 3] - hope that helps!'
    assert parse_json_object(raw) == {}


def test_parse_json_object_still_accepts_a_real_object():
    raw = '{"relevance": 4, "factuality": 3}'
    assert parse_json_object(raw) == {"relevance": 4, "factuality": 3}


def test_parse_json_object_still_accepts_an_object_embedded_in_prose():
    raw = 'Sure, here is the JSON: {"relevance": 4, "factuality": 3} - let me know if needed.'
    assert parse_json_object(raw) == {"relevance": 4, "factuality": 3}


def test_parse_json_object_rejects_a_bare_string_or_number():
    assert parse_json_object('"just a string"') == {}
    assert parse_json_object("42") == {}


def test_parse_json_object_strips_a_qwen3_style_think_block():
    # Real shape observed on Colab: Qwen3-8B via the hf-transformers backend
    # emits a <think>...</think> reasoning block before its real JSON answer
    # by default (the Ollama build of the same tag does not, which is why
    # this never showed up in local runs). The reasoning text discussing
    # JSON structure introduces stray braces that broke the naive
    # first-'{'/last-'}' bracket extraction, causing spurious parse failures.
    raw = ('<think>Let me think about the right fields, like {"target_group": '
           '...} and so on.</think>\n{"relevance": 4, "factuality": 3}')
    assert parse_json_object(raw) == {"relevance": 4, "factuality": 3}


def test_parse_json_object_handles_unclosed_trailing_think_block():
    # Generation was cut off mid-reasoning (max_tokens exhausted before any
    # real answer was produced) - nothing recoverable, must return {}, not
    # crash or return a garbage partial parse.
    raw = '<think>Still reasoning about the case when the budget ran out...'
    assert parse_json_object(raw) == {}


if __name__ == "__main__":
    test_parse_json_object_rejects_a_top_level_json_array()
    test_parse_json_object_rejects_an_array_embedded_in_prose()
    test_parse_json_object_still_accepts_a_real_object()
    test_parse_json_object_still_accepts_an_object_embedded_in_prose()
    test_parse_json_object_rejects_a_bare_string_or_number()
    test_parse_json_object_strips_a_qwen3_style_think_block()
    test_parse_json_object_handles_unclosed_trailing_think_block()
    print("test_utils.py: ALL PASSED")
