"""Smoke tests for the v22kg7 ablation-study switches.

These tests keep the full path available while verifying the three thesis
ablation modes: no_kg, no_debate, and no_persona. All LLM/RAG calls are mocked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import prompts
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


class AblationMockClient(ModelClient):
    backend_name = "mock"

    def __init__(self):
        self.final_prompts = []

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "Mock free-text response."
        if response_schema and "fidelity" in response_schema:
            return {"fidelity": "on_target", "reason": "stub - not under test"}
        if response_schema and set(response_schema) == {"counter_narrative", "explanation"}:
            self.final_prompts.append(messages)

        canned = {
            "language": "en", "target_group": "gay people", "hate_category": "homophobia",
            "intent": ["dehumanisation"], "strategy_hint": ["dignity_based_rebuttal"],
            "hate_type": "explicit", "confidence": 0.9, "hidden_claim": "gay people lack dignity",
            "evidence_topics": ["dignity and respect"], "cultural_context_needed": False,
            "region_suggestion": {"region": None, "country": None, "confidence": 0.1, "rationale": "none"},
            "safety_notes": [], "rationale": "mock case analysis",

            "candidate_personas": [
                {"name": "rhetoric analyst", "role_type": "analyst", "expertise": ["harmful rhetoric"],
                 "cultural_relevance": "n/a", "strategy": "surface assumptions", "relevance_score": 0.9,
                 "suited_for": "prosecutor"},
                {"name": "community educator", "role_type": "educator", "expertise": ["inclusive language"],
                 "cultural_relevance": "n/a", "strategy": "dignity rebuttal", "relevance_score": 0.9,
                 "suited_for": "defender"},
            ],
            "selected_prosecutor": {"name": "rhetoric analyst", "role": "prosecutor",
                                     "objective": "identify unsafe assumptions", "boundaries": ["internal-only"]},
            "selected_defender": {"name": "community educator", "role": "defender",
                                   "objective": "rebut respectfully", "expertise": ["inclusive language"],
                                   "cultural_guidance": []},
            "selection_rationale": "mock selection",

            "round": 1, "round_objective": "mock objective",
            "prosecutor_argument": "MOCK INTERNAL PROSECUTOR ARGUMENT - must not leak",
            "surfaced_claims": ["gay people lack dignity"],
            "defender_response": "Everyone deserves dignity and respect.",
            "claim_assessments": [
                {"claim": "gay people lack dignity", "verdict": "REFUTED",
                 "reasoning_summary": "mock reasoning", "usable_in_final_cn": True},
            ],
            "new_information": [], "unresolved_questions": [], "cultural_notes": [], "safety_notes": [],
            "input_language_query": "gay people dignity", "english_query": "gay people dignity",

            "selected_language": "en", "core_claim_to_counter": "gay people lack dignity",
            "recommended_strategy": ["dignity_based_rebuttal"], "approved_evidence": [],
            "rejected_content": [], "cultural_guidance": [], "safety_guidance": [],
            "final_response_plan": "Directly reject the harmful claim and affirm dignity and respect.",

            "counter_narrative": "Gay people deserve dignity and respect like anyone else.",
            "explanation": "The response directly rejects the harmful claim and emphasizes dignity and respect.",
        }
        return {k: canned[k] for k in response_schema if k in canned} or canned


def _run_mode(monkeypatch, mode):
    old_style = prompts.FINAL_CN_STYLE
    old_mode = config.ablation_mode
    try:
        prompts.FINAL_CN_STYLE = "v22kg7"
        config.set_ablation_mode(mode)
        monkeypatch.setattr(rag_pipeline, "retrieve_bilingual", _fake_retrieve_bilingual)
        client = AblationMockClient()
        pipeline = NewArchPipeline(client, backend_name="mock", model_name="mock")
        trace = pipeline.generate(InputRecord(text="Gay people should not be respected.", id="ab1", language_hint="en"))
        return trace, client
    finally:
        prompts.FINAL_CN_STYLE = old_style
        config.set_ablation_mode(old_mode)


def test_no_kg_removes_kg_from_prompt_and_trace(monkeypatch):
    trace, client = _run_mode(monkeypatch, "no_kg")
    assert trace["ablation_mode"] == "no_kg"
    assert trace["cn_knowledge_graph"]["disabled"] is True
    assert trace["kg_consistency"]["disabled"] is True
    assert trace["kg_safety_fallback"]["disabled"] is True
    assert client.final_prompts, "final CN prompt should have been generated"
    assert "COUNTER-NARRATIVE KNOWLEDGE GRAPH" not in client.final_prompts[0]
    assert "NO-GRAPH ABLATION MODE" in client.final_prompts[0]


def test_full_mode_keeps_kg_prompt_and_trace(monkeypatch):
    trace, client = _run_mode(monkeypatch, "full")
    assert trace["ablation_mode"] == "full"
    assert isinstance(trace["cn_knowledge_graph"], dict)
    assert trace["cn_knowledge_graph"].get("nodes")
    assert "COUNTER-NARRATIVE KNOWLEDGE GRAPH" in client.final_prompts[0]


def test_no_debate_disables_debate_only(monkeypatch):
    trace, _ = _run_mode(monkeypatch, "no_debate")
    assert trace["ablation_mode"] == "no_debate"
    assert trace["debate_disabled"] is True
    assert trace["debate_rounds"] == []
    assert trace["debate_outputs"] == []
    assert trace["cn_knowledge_graph"].get("nodes")


def test_no_persona_uses_fixed_generic_personas(monkeypatch):
    trace, _ = _run_mode(monkeypatch, "no_persona")
    assert trace["ablation_mode"] == "no_persona"
    assert trace["persona_selection_disabled"] is True
    assert trace["metadata"]["persona_prosecutor"] == "Critical Discourse Analyst"
    assert trace["metadata"]["persona_defender"] == "Human Rights Advocate"
    assert trace["personas"]["selection_rationale"].startswith("Ablation no_persona")
