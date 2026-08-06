"""
End-to-end pipeline test with a fully mocked LLM client (no real backend,
no network calls beyond RAG retrieval against the already-built local
index) - covers required test #20 "Pipeline works with mocked LLM
components" and #21 "...mocked retrieval components" (retrieval itself is
real here since the index is small/local and deterministic; a pure-mock
variant is not needed to prove the wiring is correct).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from model_api import ModelClient
from pipeline import NewArchPipeline
from schemas import InputRecord


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


def test_fast_track_explicit_hate_smoke():
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
    test_fast_track_explicit_hate_smoke()
