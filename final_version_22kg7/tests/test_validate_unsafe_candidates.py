"""
Covers scripts/validate_unsafe_candidates.py with a STUBBED client - no
real LLM call (no API key configured for local testing; the real live
validation is exercised by actually running the script against
outputs/unsafe_candidates_<bucket>_filtered.csv, not a unit test). These
tests check that OUR handling of the response is correct: fields get
validated/defaulted properly, failures fail toward maximum caution, and the
term/language themselves are never altered by validation.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from model_api import ModelClient
from validate_unsafe_candidates import _REQUIRED_FIELDS, validate_candidate, validate_dataframe


class _CannedValidationClient(ModelClient):
    backend_name = "stub"

    def __init__(self, canned: dict):
        self._canned = canned

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema == _REQUIRED_FIELDS:
            return dict(self._canned)
        return "n/a"


def test_validate_candidate_returns_all_five_fields():
    client = _CannedValidationClient({
        "decision": "unsafe", "category": "insult_or_slur", "severity": "high",
        "notes": "a known slur", "needs_human_review": False,
    })
    result = validate_candidate(client, "maricones", "es", ["ex1"], 2.4)
    assert set(result.keys()) == {"decision", "category", "severity", "notes", "needs_human_review"}
    assert result["decision"] == "unsafe"
    assert result["needs_human_review"] is False


def test_neutral_identity_term_decision_is_preserved():
    client = _CannedValidationClient({
        "decision": "neutral_identity_term", "category": "neutral_identity_term", "severity": "low",
        "notes": "just names an identity", "needs_human_review": False,
    })
    result = validate_candidate(client, "gay", "en", [], None)
    assert result["decision"] == "neutral_identity_term"


def test_invalid_decision_value_falls_back_to_unclear():
    """A model returning something outside the allowed decision labels must
    not be trusted verbatim - falls back to "unclear", not silently accepted."""
    client = _CannedValidationClient({
        "decision": "definitely_bad", "category": "insult_or_slur", "severity": "high",
        "notes": "x", "needs_human_review": False,
    })
    result = validate_candidate(client, "term", "en", [], None)
    assert result["decision"] == "unclear"
    assert result["needs_human_review"] is True  # forced true when decision resolves to unclear


def test_unclear_decision_always_forces_needs_human_review():
    """Even if the model explicitly said needs_human_review=false, "unclear"
    must still force review - the two fields must never disagree."""
    client = _CannedValidationClient({
        "decision": "unclear", "category": "unclear", "severity": "low",
        "notes": "not sure", "needs_human_review": False,
    })
    result = validate_candidate(client, "term", "en", [], None)
    assert result["needs_human_review"] is True


def test_parse_failure_fails_toward_maximum_caution():
    class _EmptyClient(ModelClient):
        backend_name = "empty"

        def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
            return {} if response_schema else "n/a"

    result = validate_candidate(_EmptyClient(), "term", "en", [], None)
    assert result["decision"] == "unclear"
    assert result["needs_human_review"] is True


def test_validate_dataframe_respects_limit_and_prioritizes_by_score():
    client = _CannedValidationClient({
        "decision": "unclear", "category": "unclear", "severity": "low",
        "notes": "x", "needs_human_review": True,
    })
    df = pd.DataFrame([
        {"language": "en", "term": "low", "hate_association_score": 0.1, "example_hate_snippets": ""},
        {"language": "en", "term": "high", "hate_association_score": 3.0, "example_hate_snippets": ""},
        {"language": "en", "term": "mid", "hate_association_score": 1.5, "example_hate_snippets": ""},
    ])
    result = validate_dataframe(client, df, limit=2)
    assert len(result) == 2
    assert set(result["term"]) == {"high", "mid"}


def test_validate_dataframe_no_limit_validates_everything():
    client = _CannedValidationClient({
        "decision": "unclear", "category": "unclear", "severity": "low",
        "notes": "x", "needs_human_review": True,
    })
    df = pd.DataFrame([
        {"language": "en", "term": "a", "hate_association_score": 0.1, "example_hate_snippets": ""},
        {"language": "en", "term": "b", "hate_association_score": 3.0, "example_hate_snippets": ""},
    ])
    result = validate_dataframe(client, df, limit=0)
    assert len(result) == 2


def test_validate_dataframe_preserves_original_columns():
    client = _CannedValidationClient({
        "decision": "unclear", "category": "unclear", "severity": "low",
        "notes": "x", "needs_human_review": True,
    })
    df = pd.DataFrame([{"language": "en", "term": "x", "hate_count": 5, "reference_count": 0,
                         "hate_association_score": 1.0, "example_hate_snippets": ""}])
    result = validate_dataframe(client, df, limit=0)
    assert result.iloc[0]["hate_count"] == 5
    assert result.iloc[0]["term"] == "x"


def test_validate_dataframe_empty_input_returns_empty():
    client = _CannedValidationClient({
        "decision": "unclear", "category": "unclear", "severity": "low",
        "notes": "x", "needs_human_review": True,
    })
    df = pd.DataFrame(columns=["language", "term", "hate_association_score", "example_hate_snippets"])
    result = validate_dataframe(client, df)
    assert result.empty


if __name__ == "__main__":
    test_validate_candidate_returns_all_five_fields()
    test_neutral_identity_term_decision_is_preserved()
    test_invalid_decision_value_falls_back_to_unclear()
    test_unclear_decision_always_forces_needs_human_review()
    test_parse_failure_fails_toward_maximum_caution()
    test_validate_dataframe_respects_limit_and_prioritizes_by_score()
    test_validate_dataframe_no_limit_validates_everything()
    test_validate_dataframe_preserves_original_columns()
    test_validate_dataframe_empty_input_returns_empty()
    print("test_validate_unsafe_candidates.py: ALL PASSED")
