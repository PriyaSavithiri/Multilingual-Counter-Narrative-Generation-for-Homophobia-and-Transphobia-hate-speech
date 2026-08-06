import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evidence_verifier
from model_api import ModelClient
from schemas import VALID_VERDICTS


class _ExplodingClient(ModelClient):
    """Raises if generate() is ever called - used to prove no_rag mode
    never even formulates a retrieval query, let alone retrieves anything."""
    backend_name = "exploding"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        raise AssertionError("No LLM call (including query formulation) should happen in no_rag mode.")


def test_no_rag_mode_generates_zero_queries_and_zero_evidence():
    items, block = evidence_verifier.gather_evidence_for_claim(
        _ExplodingClient(), claim="some claim", language="en", region=None,
        rag_mode="no_rag", filter_target=True, query_strategy="both",
    )
    assert items == []
    assert "No relevant retrieved evidence" in block


def test_valid_verdict_labels_are_exactly_three():
    assert set(VALID_VERDICTS) == {"SUPPORTED", "REFUTED", "NEI"}


class _StubQueryClient(ModelClient):
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        return {"input_language_query": "consulta en espanol", "english_query": "english query"}


def test_bilingual_query_formulation_returns_both_variants():
    input_q, english_q = evidence_verifier.formulate_bilingual_queries(_StubQueryClient(), "a claim", "es")
    assert input_q == "consulta en espanol"
    assert english_q == "english query"


if __name__ == "__main__":
    test_no_rag_mode_generates_zero_queries_and_zero_evidence()
    test_valid_verdict_labels_are_exactly_three()
    test_bilingual_query_formulation_returns_both_variants()
    print("test_evidence_verification.py: ALL PASSED")
