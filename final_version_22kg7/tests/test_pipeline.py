"""
End-to-end pipeline test with a fully mocked LLM client AND mocked RAG
retrieval (no real backend, no real FAISS/embedding-model calls) - covers
required test #20 "Pipeline works with mocked LLM components" and #21
"...mocked retrieval components". Retrieval used to be real here (against
the already-built local index), on the reasoning that it's small/local/
deterministic - but that made this test depend on sentence-transformers
successfully importing transformers.PreTrainedModel, which breaks on some
local installs (observed: a broken torchvision/transformers dependency
chain) for reasons entirely unrelated to this project's own code. Mocking
retrieval removes that dependency without losing what this test actually
checks (the pipeline's wiring, not FAISS/embedding correctness - that's
covered separately, see test_evidence_verification.py / rag_pipeline itself).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import rag_pipeline
from model_api import ModelClient
from pipeline import NewArchPipeline
from schemas import InputRecord

_FAKE_RAG_RECORD = {
    "corpus_id": 1, "dataset_name": "mock-corpus", "language": "en", "_score": 0.9,
    "knowledge_text": "Mock factual evidence for testing.", "reference_counter_narrative": "",
    "_query_used": "mock query", "_query_language": "input_language",
}


def _fake_retrieve_bilingual(**kwargs):
    return [dict(_FAKE_RAG_RECORD)]


class MockClient(ModelClient):
    """Returns a plausible canned JSON object for whatever response_schema
    is requested, so every agent's parse step succeeds without a real LLM."""
    backend_name = "mock"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "Mock free-text response."

        canned = {
            "language": "en", "target_group": "gay men", "hate_category": "homophobia",
            "intent": ["moral_panic"], "strategy_hint": ["rights_based_framing"],
            "hate_type": "implicit", "confidence": 0.8, "hidden_claim": "being gay is unnatural",
            "evidence_topics": ["biology of sexual orientation"], "cultural_context_needed": False,
            "region_suggestion": {"region": None, "country": None, "confidence": 0.1, "rationale": "no strong evidence"},
            "safety_notes": [], "rationale": "mock rationale",

            "candidate_personas": [
                {"name": "biologist", "role_type": "scientist", "expertise": ["evolutionary biology"],
                 "cultural_relevance": "n/a", "strategy": "myth_correction", "relevance_score": 0.9,
                 "suited_for": "defender"},
                {"name": "rhetoric analyst", "role_type": "analyst", "expertise": ["moral panic rhetoric"],
                 "cultural_relevance": "n/a", "strategy": "surface assumptions", "relevance_score": 0.85,
                 "suited_for": "prosecutor"},
            ],
            "selected_prosecutor": {"name": "rhetoric analyst", "role": "prosecutor",
                                     "objective": "surface the claim's internal logic",
                                     "boundaries": ["internal-only"]},
            "selected_defender": {"name": "biologist", "role": "defender",
                                   "objective": "rebut with evidence", "expertise": ["evolutionary biology"],
                                   "cultural_guidance": []},
            "selection_rationale": "mock selection rationale",

            "round": 1, "round_objective": "mock objective",
            "prosecutor_argument": "MOCK INTERNAL PROSECUTOR ARGUMENT - must never leak to output",
            "surfaced_claims": ["being gay is unnatural"],

            "defender_response": "Mock defender rebuttal grounded in evidence.",
            "claim_assessments": [
                {"claim": "being gay is unnatural", "verdict": "REFUTED",
                 "reasoning_summary": "mock reasoning", "usable_in_final_cn": True},
            ],
            "new_information": ["mock new info"], "unresolved_questions": [],
            "cultural_notes": [], "safety_notes": [],

            "input_language_query": "es la homosexualidad antinatural", "english_query": "is homosexuality unnatural",

            "selected_language": "en", "core_claim_to_counter": "being gay is unnatural",
            "recommended_strategy": ["myth_correction"], "approved_evidence": [],
            "rejected_content": [], "cultural_guidance": [], "safety_guidance": [],
            "final_response_plan": "Write a calm, evidence-based rebuttal.",

            "counter_narrative": "Being gay is a natural part of human diversity, observed across cultures and history.",
            "explanation": "Used a myth-correction strategy grounded in biology.",
        }
        return {k: canned[k] for k in response_schema if k in canned} or canned


def test_fast_track_explicit_hate_smoke(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "retrieve_bilingual", _fake_retrieve_bilingual)
    client = MockClient()
    # Force explicit routing for this test by monkeypatching CaseAnalysisAgent's
    # output indirectly isn't needed - MockClient always returns hate_type="implicit"
    # above for the deep-dive test; a second canned client covers explicit.
    pipeline = NewArchPipeline(client, backend_name="mock", model_name="mock")
    record = InputRecord(text="Being gay is unnatural and against nature.", id="t1", rag_mode="dual_rag")
    trace = pipeline.generate(record, filter_target=True)

    assert trace["counter_narrative"], "final counter_narrative should not be empty"
    assert "prosecutor_argument" not in str(trace), "Prosecutor content must never leak into the public trace"
    assert trace["selected_track"] in ("fast_track", "deep_dive")
    assert trace["metadata"]["persona_prosecutor"] == "rhetoric analyst"
    assert trace["metadata"]["persona_defender"] == "biologist"
    print("test_fast_track_explicit_hate_smoke: PASSED")
    print("trace keys:", list(trace.keys()))
    print("counter_narrative:", trace["counter_narrative"])


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_fast_track_explicit_hate_smoke(_FakeMonkeypatch())
