"""
Coverage for the v22kg Counter-Narrative Knowledge Graph work: the schema/
builder/validator in counter_narrative_kg.py, the v22kg prompt style in
prompts.py, and the deliberate, minimal, gated wiring into
final_cn_agent.py/pipeline.py (see tests/test_generation_path_frozen.py's
docstring for why those two files' manifest hashes changed).

Directly motivated by a real, repeated finding from live Colab runs (V4_67,
Tamil): the Judge's own final_response_plan invented an unsupported claim
("religion has principles of accepting everyone" / "மதம் அனைத்து
மனிதர்களையும் ஏற்றுக்கொள்ளும் கொள்கைகளை கொண்டுள்ளது") from evidence that
only supported "empathy, fairness, and respect" - free-text bans in the
v22b6 prompt alone kept missing new rewordings of the same unsupported
claim. Several tests below reproduce that exact real case.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import main
import prompts
import rag_pipeline
from counter_narrative_kg import (
    EDGE_TYPES, NODE_TYPES, build_counter_narrative_kg, validate_kg_consistency,
)
from final_cn_agent import FinalCNAgent
from model_api import ModelClient
from pipeline import NewArchPipeline
from schemas import InputRecord, JudgePlan

_V4_67_APPROVED_EVIDENCE = [
    {"source_id": "e1", "title": "", "passage": (
        "Recognizing LGBTQ+ rights is an act of justice, not defiance. True spiritual values "
        "center on empathy, fairness, and respect for all."
     ), "retrieval_score": 0.8, "source_type": "cultural"},
    {"source_id": "e2", "title": "", "passage": "பாலின அடையாளத்தை அவமானப்படுத்துவது பாகுபாடு.",
     "retrieval_score": 0.7, "source_type": "cultural"},
]
_V4_67_FINAL_RESPONSE_PLAN = (
    "LGBT+ அடையாளங்களை கடவுளின் சாபமாக கருதுவது தவறான கருத்தாகும். மதம் அனைத்து மனிதர்களையும் "
    "ஏற்றுக்கொள்ளும் கொள்கைகளை கொண்டுள்ளது. சமூக நீதி மற்றும் மனித உரிமைகள் அனைத்து மக்களுக்கும் "
    "சமநிலைக்கு வழிவகுக்கின்றன."
)


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------
def test_kg_is_json_serializable():
    kg = build_counter_narrative_kg("some hate comment", "en")
    json.dumps(kg)  # must not raise


def test_kg_has_required_top_level_keys():
    kg = build_counter_narrative_kg("comment", "en")
    assert set(kg.keys()) == {"graph_id", "language", "region_context", "nodes", "edges", "kg_summary"}


def test_kg_node_types_all_in_allowed_vocabulary():
    kg = build_counter_narrative_kg("Gay people are criminals and should be silenced.", "en")
    assert kg["nodes"], "builder must always produce at least the base nodes"
    for node in kg["nodes"]:
        assert node["type"] in NODE_TYPES, f"unexpected node type: {node['type']}"
        assert set(node.keys()) >= {"id", "type", "text", "language", "source", "source_type",
                                     "evidence_ids", "confidence"}


def test_kg_edge_types_all_in_allowed_vocabulary():
    kg = build_counter_narrative_kg("Gay people are criminals and should be silenced.", "en")
    assert kg["edges"], "builder must always produce at least the base edges"
    for edge in kg["edges"]:
        assert edge["type"] in EDGE_TYPES, f"unexpected edge type: {edge['type']}"


def test_kg_builder_never_raises_on_missing_optional_inputs():
    kg = build_counter_narrative_kg("comment", "ta", case_analysis=None, judge_plan=None,
                                     approved_evidence=None, region_context=None)
    assert kg["kg_summary"]["target_group"] == "unclear"
    assert kg["region_context"] == "unknown"


# ---------------------------------------------------------------------------
# 2. Evidence separation - EXACT source_type match only
# ---------------------------------------------------------------------------
def test_fact_source_type_becomes_factual_evidence_node():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "A real fact.", "source_type": "fact"},
    ])
    types = [n["type"] for n in kg["nodes"] if n["text"] == "A real fact."]
    assert types == ["approved_factual_evidence"]


def test_web_source_type_becomes_factual_evidence_node():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "A web-sourced fact.", "source_type": "web"},
    ])
    types = [n["type"] for n in kg["nodes"] if n["text"] == "A web-sourced fact."]
    assert types == ["approved_factual_evidence"]


def test_cultural_source_type_becomes_cultural_context_node():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Cultural framing only.", "source_type": "cultural"},
    ])
    types = [n["type"] for n in kg["nodes"] if n["text"] == "Cultural framing only."]
    assert types == ["approved_cultural_context"]


def test_missing_unknown_or_translated_source_type_falls_back_to_cultural_context():
    for source_type in ["culturale", "", None, "unknown", "translated", "garbage_value"]:
        evidence = [{"source_id": "1", "passage": f"passage for {source_type!r}", "source_type": source_type}]
        kg = build_counter_narrative_kg("comment", "en", approved_evidence=evidence)
        node = next(n for n in kg["nodes"] if n["text"] == f"passage for {source_type!r}")
        assert node["type"] == "approved_cultural_context", f"source_type={source_type!r} must fall back to cultural"


# ---------------------------------------------------------------------------
# 3. Forbidden claims - country/region, science/research, legal
# ---------------------------------------------------------------------------
def test_country_region_forbidden_claim_present_when_no_country_named_in_factual_evidence():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Many countries recognize LGBTQ+ rights.", "source_type": "fact"},
    ])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_country_claim" in forbidden
    assert "unsupported_region_claim" in forbidden
    assert "unsupported_local_culture_claim" in forbidden


def test_country_region_forbidden_claim_absent_when_factual_evidence_names_a_country():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Italy has laws protecting LGBTQ+ people.", "source_type": "fact"},
    ])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_country_claim" not in forbidden


def test_science_research_forbidden_claim_present_without_factual_support():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert {"unsupported_science_claim", "unsupported_research_claim", "unsupported_statistics_claim"} <= forbidden


def test_science_research_forbidden_claim_absent_when_factual_evidence_supports_it():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Research shows sexual orientation is not a choice.", "source_type": "fact"},
    ])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_research_claim" not in forbidden


def test_legal_forbidden_claim_present_without_factual_support():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_legal_claim" in forbidden


def test_legal_forbidden_claim_absent_when_factual_evidence_supports_it():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "The law protects LGBTQ+ people from discrimination.", "source_type": "fact"},
    ])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_legal_claim" not in forbidden


# ---------------------------------------------------------------------------
# 4. Religious/theological invention - the real V4_67 case. Deliberately
# conservative: cultural-only "empathy/fairness/respect" evidence must NEVER
# lift the forbidden node, only an explicit institutional-religious marker
# in FACT/WEB evidence may.
# ---------------------------------------------------------------------------
def test_theological_forbidden_claim_present_for_real_v4_67_style_cultural_only_evidence():
    kg = build_counter_narrative_kg(
        "LGBT+ identity being called a curse from God is wrong.", "ta",
        approved_evidence=_V4_67_APPROVED_EVIDENCE,
    )
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_theological_claim" in forbidden
    assert "unsupported_religion_generalization" in forbidden


def test_theological_forbidden_claim_gets_values_level_style_constraint_when_evidence_mentions_empathy():
    kg = build_counter_narrative_kg("comment", "ta", approved_evidence=_V4_67_APPROVED_EVIDENCE)
    style_nodes = [n["text"] for n in kg["nodes"] if n["type"] == "style_constraint"]
    assert any("values-level only" in t for t in style_nodes)


def test_theological_forbidden_claim_absent_when_institutional_fact_evidence_supports_it():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Official church doctrine states all people deserve dignity.",
         "source_type": "fact"},
    ])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_theological_claim" not in forbidden
    assert "unsupported_religion_generalization" not in forbidden


def test_theological_forbidden_claim_not_lifted_by_cultural_tagged_institutional_wording():
    """Even if a CULTURAL passage's own text uses an institutional-sounding word
    like "doctrine", it must not lift the forbidden node - only fact/web can."""
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Official church doctrine states all people deserve dignity.",
         "source_type": "cultural"},
    ])
    forbidden = {n["text"] for n in kg["nodes"] if n["type"] == "forbidden_claim"}
    assert "unsupported_theological_claim" in forbidden


# ---------------------------------------------------------------------------
# 5. should_not_strengthen_to edges - paraphrase-drift guarding
# ---------------------------------------------------------------------------
def test_should_not_strengthen_edge_natural_diversity_to_not_a_choice():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Being gay is a natural expression of human diversity.",
         "source_type": "fact"},
    ])
    evidence_node = next(n for n in kg["nodes"] if n["text"] == "Being gay is a natural expression of human diversity.")
    strengthen_edges = [e for e in kg["edges"] if e["type"] == "should_not_strengthen_to"
                         and e["from_node"] == evidence_node["id"]]
    assert strengthen_edges, "must flag this exact narrowing risk"
    target = next(n for n in kg["nodes"] if n["id"] == strengthen_edges[0]["to_node"])
    assert target["text"] == "not_a_choice"


def test_should_not_strengthen_edge_many_countries_to_country_claim():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Many countries recognize LGBTQ+ rights.", "source_type": "fact"},
    ])
    evidence_node = next(n for n in kg["nodes"] if n["text"] == "Many countries recognize LGBTQ+ rights.")
    strengthen_edges = [e for e in kg["edges"] if e["type"] == "should_not_strengthen_to"
                         and e["from_node"] == evidence_node["id"]]
    assert strengthen_edges
    target = next(n for n in kg["nodes"] if n["id"] == strengthen_edges[0]["to_node"])
    assert target["text"] == "unsupported_country_claim"


# ---------------------------------------------------------------------------
# 6. Required rebuttal generation
# ---------------------------------------------------------------------------
def test_required_rebuttal_crime_generalization():
    kg = build_counter_narrative_kg("Gay people are all criminals.", "en")
    rebuttal = next(n["text"] for n in kg["nodes"] if n["type"] == "required_rebuttal")
    assert rebuttal == "Do not blame a whole group for crimes."
    assert kg["kg_summary"]["claim_type"] == "crime_generalization"


def test_required_rebuttal_silencing():
    kg = build_counter_narrative_kg("Their voices should not speak about this at all.", "en")
    rebuttal = next(n["text"] for n in kg["nodes"] if n["type"] == "required_rebuttal")
    assert rebuttal == "Silencing someone because of identity is unfair."
    assert kg["kg_summary"]["claim_type"] == "silencing"


def test_required_rebuttal_curse_or_religion_framing():
    kg = build_counter_narrative_kg("Being gay is a curse from God.", "en")
    rebuttal = next(n["text"] for n in kg["nodes"] if n["type"] == "required_rebuttal")
    assert rebuttal == "Calling someone's identity a curse is harmful."
    assert kg["kg_summary"]["claim_type"] == "curse_or_religion_framing"


def test_required_rebuttal_generic_fallback_when_no_claim_type_keyword_matches():
    kg = build_counter_narrative_kg("Something vague and unrelated to any known category.", "en")
    rebuttal = next(n["text"] for n in kg["nodes"] if n["type"] == "required_rebuttal")
    assert rebuttal == "Everyone deserves dignity and equal respect."
    assert kg["kg_summary"]["claim_type"] == "generic_dehumanization"


# ---------------------------------------------------------------------------
# 7. Non-invasive consistency validator - directly reproduces the real V4_67
# case: the Judge's own final_response_plan trips the theological forbidden
# node built from cultural-only evidence.
# ---------------------------------------------------------------------------
def test_validate_kg_consistency_flags_real_v4_67_final_response_plan():
    kg = build_counter_narrative_kg(
        "LGBT+ identity being called a curse from God is wrong.", "ta",
        approved_evidence=_V4_67_APPROVED_EVIDENCE,
    )
    judge_plan = {"final_response_plan": _V4_67_FINAL_RESPONSE_PLAN, "cultural_guidance": []}
    result = validate_kg_consistency(kg, judge_plan)
    assert "unsupported_religion_generalization" in result["forbidden_claims_triggered"]
    assert result["risk_level"] != "none"


def test_validate_kg_consistency_returns_none_risk_for_clean_plan():
    kg = build_counter_narrative_kg("Gay people are criminals.", "en", approved_evidence=[])
    judge_plan = {"final_response_plan": "Respond with dignity and respect.", "cultural_guidance": []}
    result = validate_kg_consistency(kg, judge_plan)
    assert result["risk_level"] == "none"
    assert result["forbidden_claims_triggered"] == []


def test_validate_kg_consistency_never_mutates_inputs():
    kg = build_counter_narrative_kg("Gay people are criminals.", "en", approved_evidence=[])
    judge_plan = {"final_response_plan": "science confirms this and the law protects it too",
                  "cultural_guidance": []}
    before = json.dumps(kg, sort_keys=True)
    validate_kg_consistency(kg, judge_plan)
    assert json.dumps(kg, sort_keys=True) == before, "validator must be read-only, never rewrite the KG"


# ---------------------------------------------------------------------------
# 8. v22kg prompt wiring
# ---------------------------------------------------------------------------
def _minimal_judge_plan_dict(final_response_plan="plan", approved_evidence=None, cultural_guidance=None):
    return {
        "selected_language": "ta", "core_claim_to_counter": "claim", "recommended_strategy": [],
        "approved_evidence": approved_evidence or [], "rejected_content": [],
        "cultural_guidance": cultural_guidance or [], "safety_guidance": [],
        "final_response_plan": final_response_plan,
    }


def test_v22kg_prompt_includes_kg_grounding_block():
    prompt = prompts.build_final_cn_prompt("comment", "ta", _minimal_judge_plan_dict(), style="v22kg")
    assert "COUNTER-NARRATIVE KNOWLEDGE GRAPH" in prompt
    assert "FORBIDDEN CLAIMS" in prompt
    assert "REQUIRED REBUTTAL" in prompt


def test_v22kg_prompt_states_final_response_plan_is_not_evidence():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg")
    assert "SUGGESTION" in prompt
    assert "never as evidence" in prompt


def test_v22kg_prompt_states_forbidden_overrides_plan():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg")
    assert "ignore that specific" in prompt
    assert "part of the final_response_plan entirely" in prompt


def test_v22kg_prompt_preserves_v22b6_tamil_style_anchors():
    prompt = prompts.build_final_cn_prompt("comment", "ta", _minimal_judge_plan_dict(), style="v22kg")
    assert "STYLE ANCHORS, not fixed templates" in prompt
    assert "For Tamil specifically" in prompt


def test_v22kg_prompt_uses_real_kg_nodes_not_placeholder_text():
    plan = _minimal_judge_plan_dict(final_response_plan=_V4_67_FINAL_RESPONSE_PLAN,
                                     approved_evidence=_V4_67_APPROVED_EVIDENCE)
    prompt = prompts.build_final_cn_prompt("curse comment", "ta", plan, style="v22kg")
    assert "unsupported_religion_generalization" in prompt


def test_v22kg_prompt_opening_line_prioritizes_kg_over_judge_plan():
    """v22b6's default opening line ("...following the Judge's plan.") frames the plan as
    authoritative, which conflicts with the KG's whole point (the plan may contain unsupported
    claims - see the real V4_67 case). v22kg must rewrite ONLY this opening line."""
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg")
    assert prompt.startswith(
        "Write the final counter-narrative response using the Counter-Narrative Knowledge Graph as "
        "the primary grounding contract. Use the Judge's plan only as a non-evidence suggestion when "
        "it does not conflict with the KG."
    )
    assert "following the Judge's plan." not in prompt


def test_v22b6_opening_line_unchanged_by_v22kg_addition():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22b6")
    assert prompt.startswith(
        "Write the final counter-narrative response to this comment, following the Judge's plan."
    )


def test_older_styles_unaffected_by_v22kg_addition():
    plan = _minimal_judge_plan_dict()
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "en", plan, style=style)
        assert "COUNTER-NARRATIVE KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg"


def test_case_analysis_argument_does_not_affect_prompt_for_old_styles():
    """Rule 2 (v22kg gating): final_cn_agent.py now always threads case_analysis through, for
    every style - but build_final_cn_prompt must only ever forward it to the v22kg branch. This
    proves it byte-for-byte, for every non-KG style, with a non-null case_analysis."""
    ca = {"target_group": "gay men", "hate_category": "homophobia", "language": "en"}
    plan = _minimal_judge_plan_dict()
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        without_ca = prompts.build_final_cn_prompt("comment", "en", plan, style=style, case_analysis=None)
        with_ca = prompts.build_final_cn_prompt("comment", "en", plan, style=style, case_analysis=ca)
        assert without_ca == with_ca, f"{style} prompt must be byte-identical regardless of case_analysis"


def test_filename_suffix_for_v22kg():
    assert main._tag_filename_for_final_cn_style("ta", "v22kg") == "ta-finalcnv22kg"


def test_v22kg_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg"])
    assert args.final_cn_style == "v22kg"


# ---------------------------------------------------------------------------
# 9. End-to-end trace wiring (pipeline.py) - gated strictly on
# prompts.FINAL_CN_STYLE == "v22kg"; every other style must leave the trace
# shape exactly as before.
# ---------------------------------------------------------------------------
_FAKE_RAG_RECORD = {
    "corpus_id": 1, "dataset_name": "mock-corpus", "language": "ta", "_score": 0.9,
    "knowledge_text": "Mock factual evidence for testing.", "reference_counter_narrative": "",
    "_query_used": "mock query", "_query_language": "input_language",
}


def _fake_retrieve_bilingual(**kwargs):
    return [dict(_FAKE_RAG_RECORD)]


class _MockClient(ModelClient):
    backend_name = "mock"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "Mock free-text response."
        canned = {
            "language": "ta", "target_group": "LGBTQ+ people", "hate_category": "homophobia",
            "intent": ["moral_panic"], "strategy_hint": ["rights_based_framing"],
            "hate_type": "implicit", "confidence": 0.8, "hidden_claim": "being gay is unnatural",
            "evidence_topics": ["dignity"], "cultural_context_needed": False,
            "region_suggestion": {"region": None, "country": None, "confidence": 0.1, "rationale": "n/a"},
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
                                     "objective": "surface the claim's internal logic", "boundaries": ["internal-only"]},
            "selected_defender": {"name": "biologist", "role": "defender", "objective": "rebut with evidence",
                                   "expertise": ["evolutionary biology"], "cultural_guidance": []},
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
            "input_language_query": "mock query", "english_query": "mock query en",
            "selected_language": "ta", "core_claim_to_counter": "being gay is unnatural",
            "recommended_strategy": ["myth_correction"], "approved_evidence": [],
            "rejected_content": [], "cultural_guidance": [], "safety_guidance": [],
            "final_response_plan": "Write a calm, evidence-based rebuttal.",
            "counter_narrative": "ஒவ்வொருவரின் அடையாளமும் மரியாதைக்குரியது.",
            "explanation": "Used a dignity-based framing grounded in evidence.",
        }
        return {k: canned[k] for k in response_schema if k in canned} or canned


class _StubFinalCNClient(ModelClient):
    """Minimal stub for FinalCNAgent.run() - always passes the echo/off-topic/fidelity guards
    (same comment/core_claim/counter_narrative combo already proven safe in test_pipeline.py's
    MockClient), so these tests are isolated to the one thing they're checking: whether
    cn_knowledge_graph/kg_consistency get attached to the return dict."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        if response_schema and "fidelity" in response_schema:
            return {"fidelity": "on_target", "reason": "stub - not under test here"}
        return {"counter_narrative": "Being gay is a natural part of human diversity, observed across cultures and history.",
                "explanation": "Used a myth-correction strategy grounded in biology."}


def _run_final_cn_agent_with_style(style: str) -> dict:
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = style
        plan = JudgePlan(selected_language="en", core_claim_to_counter="being gay is unnatural",
                          final_response_plan="plan")
        agent = FinalCNAgent(_StubFinalCNClient())
        return agent.run("Being gay is unnatural and against nature.", plan, region=None, case_analysis=None)
    finally:
        prompts.FINAL_CN_STYLE = original_style


def test_final_cn_agent_return_dict_excludes_kg_fields_for_every_old_style():
    """Rule 7: old styles' FINAL OUTPUT (not just the pipeline trace) must never gain
    cn_knowledge_graph/kg_consistency keys."""
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        result = _run_final_cn_agent_with_style(style)
        assert set(result.keys()) == {"counter_narrative", "explanation", "safety_flags"}, (
            f"{style} must return exactly the original 3 keys, got {set(result.keys())}"
        )


def test_final_cn_agent_return_dict_includes_kg_fields_for_v22kg_style():
    """Rule 8: the v22kg style's FINAL OUTPUT (not just the pipeline trace) must include both
    cn_knowledge_graph and kg_consistency, with a real, non-empty graph."""
    result = _run_final_cn_agent_with_style("v22kg")
    assert "cn_knowledge_graph" in result
    assert "kg_consistency" in result
    assert result["cn_knowledge_graph"]["nodes"], "attached KG must contain real nodes, not be empty"
    assert set(result["kg_consistency"].keys()) == {
        "final_response_plan_violations", "forbidden_claims_triggered", "risk_level",
    }


def test_pipeline_trace_includes_kg_fields_when_v22kg_style_active(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "retrieve_bilingual", _fake_retrieve_bilingual)
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v22kg"
        client = _MockClient()
        pipeline = NewArchPipeline(client, backend_name="mock", model_name="mock")
        record = InputRecord(text="Being gay is unnatural.", id="kg1", language_hint="ta", rag_mode="dual_rag")
        trace = pipeline.generate(record, filter_target=True)
    finally:
        prompts.FINAL_CN_STYLE = original_style

    assert "cn_knowledge_graph" in trace
    assert "kg_consistency" in trace
    assert "forbidden_claim_nodes" in trace
    assert "required_rebuttal_nodes" in trace
    assert "approved_factual_evidence_nodes" in trace
    assert "approved_cultural_context_nodes" in trace
    assert trace["cn_knowledge_graph"]["nodes"], "trace KG must contain real nodes, not be empty"
    assert "prosecutor_argument" not in str(trace), "Prosecutor content must never leak, even with KG attached"


def test_pipeline_trace_omits_kg_fields_for_v20_style(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "retrieve_bilingual", _fake_retrieve_bilingual)
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v20"
        client = _MockClient()
        pipeline = NewArchPipeline(client, backend_name="mock", model_name="mock")
        record = InputRecord(text="Being gay is unnatural.", id="kg2", language_hint="ta", rag_mode="dual_rag")
        trace = pipeline.generate(record, filter_target=True)
    finally:
        prompts.FINAL_CN_STYLE = original_style

    assert "cn_knowledge_graph" not in trace
    assert "kg_consistency" not in trace
    assert "forbidden_claim_nodes" not in trace


if __name__ == "__main__":
    import traceback

    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    failures = []
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            try:
                if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                    fn(_FakeMonkeypatch())
                else:
                    fn()
                print(f"{name}: PASSED")
            except Exception:
                failures.append(name)
                print(f"{name}: FAILED")
                traceback.print_exc()
    if failures:
        print(f"\n{len(failures)} FAILED: {failures}")
    else:
        print("\ntest_counter_narrative_kg.py: ALL PASSED")
