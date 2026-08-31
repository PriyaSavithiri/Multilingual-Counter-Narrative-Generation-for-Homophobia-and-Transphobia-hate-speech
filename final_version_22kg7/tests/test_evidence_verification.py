import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evidence_verifier
import rag_pipeline
from model_api import ModelClient
from schemas import VALID_VERDICTS
from utils import clean_evidence_text


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


def test_clean_evidence_text_strips_eos_markers():
    dirty = "First sentence.<EOS> Second sentence.<EOS>Third sentence."
    cleaned = clean_evidence_text(dirty)
    assert "<EOS>" not in cleaned
    assert "eos" not in cleaned.lower()


def test_clean_evidence_text_handles_empty_input():
    assert clean_evidence_text("") == ""
    assert clean_evidence_text(None) == ""


def test_local_rag_items_passage_never_contains_eos(monkeypatch):
    """Confirms the cleaning is actually wired into the real evidence path
    (_local_rag_items), not just available as a standalone utility function -
    corpus.jsonl is known to contain literal '<EOS>' in ~10% of knowledge_text
    records (see rag_pipeline docs), so this simulates one of those."""
    def _fake_retrieve_bilingual(**kwargs):
        return [{
            "corpus_id": 42, "dataset_name": "test-source", "language": "en", "_score": 0.8,
            "knowledge_text": "Some fact.<EOS> Another fact.<EOS>", "reference_counter_narrative": "",
            "_query_used": "q", "_query_language": "input_language",
        }]
    monkeypatch.setattr(rag_pipeline, "retrieve_bilingual", _fake_retrieve_bilingual)
    items = evidence_verifier._local_rag_items(
        input_q="q", english_q="q", language="en", region=None, rag_mode="dual_rag",
        filter_target=True, query_strategy="both",
    )
    assert len(items) == 1
    assert "<EOS>" not in items[0].passage


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_no_rag_mode_generates_zero_queries_and_zero_evidence()
    test_valid_verdict_labels_are_exactly_three()
    test_bilingual_query_formulation_returns_both_variants()
    test_clean_evidence_text_strips_eos_markers()
    test_clean_evidence_text_handles_empty_input()
    test_local_rag_items_passage_never_contains_eos(_FakeMonkeypatch())
    print("test_evidence_verification.py: ALL PASSED")
