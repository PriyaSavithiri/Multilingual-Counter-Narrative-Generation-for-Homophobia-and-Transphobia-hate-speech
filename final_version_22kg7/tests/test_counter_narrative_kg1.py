"""
Coverage for v22kg1: KG obedience + final-output consistency + Tamil
one-sentence safety fix, built on top of v22kg (v22kg itself must stay
completely unchanged - see test_counter_narrative_kg.py for v22kg's own
coverage, still passing untouched).

Directly motivated by the real v22kg Tamil rerun (5 rows, reviewed this
session): V4_67's counter_narrative still generalized about "religions" in
new wording ("மதங்கள் அன்பு, மரியாதை, மற்றும் ஒப்புக்கொள்ளல் ...
ஊக்குவிக்கின்றன") that v22kg's narrower, judge-plan-only validator never
even had a chance to catch (it only ever looked at final_response_plan, not
the model's own output); V4_64's counter_narrative code-mixed a stray French
word ("diversité"); V4_668's explanation used disease/disorder framing
("குறிப்பிட்ட கோளாறு தொடர்பான...") for a claim_type that was never about
disease. v22kg1 addresses all three, plus tightens Tamil to exactly 1
sentence (matching typical reference length) instead of 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import prompts
import rag_pipeline
from counter_narrative_kg import (
    build_counter_narrative_kg, validate_kg_consistency_full, _detect_code_mixing,
)
from final_cn_agent import FinalCNAgent
from model_api import ModelClient
from pipeline import NewArchPipeline
from schemas import InputRecord, JudgePlan

# Real trace text from the v22kg Tamil rerun (V4_67), reviewed this session.
_V4_67_APPROVED_EVIDENCE = [
    {"source_id": "e1", "title": "", "passage": (
        "Recognizing LGBTQ+ rights is an act of justice, not defiance. True spiritual values "
        "center on empathy, fairness, and respect for all."
     ), "retrieval_score": 0.8, "source_type": "cultural"},
    {"source_id": "e2", "title": "", "passage": "பாலின அடையாளத்தை அவமானப்படுத்துவது பாகுபாடு.",
     "retrieval_score": 0.7, "source_type": "cultural"},
]

# The real leaked text from the v22kg Tamil rerun - v22kg's validator missed this entirely.
_V4_67_LEAKED_COUNTER_NARRATIVE = (
    "ஒருவரின் அடையாளத்தை சாபமாகச் சொல்லுவது அவர்களுக்கு மிகுந்த பாதிப்பு ஏற்படுத்தும். "
    "மதங்கள் அன்பு, மரியாதை, மற்றும் ஒப்புக்கொள்ளல் போன்ற மதிப்புகளை ஊக்குவிக்கின்றன."
)
_V4_67_LEAKED_EXPLANATION = (
    "இந்த பதிலில், அடையாளத்தை சாபமாகக் குறிப்பிடுவதன் பாதிப்பைக் காட்டி, மத மரபுகளின் "
    "உள்ளடக்கிய தன்மையை மேற்கோள் காட்டப்பட்டது."
)

# The real leaked explanation from V4_64 - crediting "science" for a cultural-only evidence item.
_V4_64_LEAKED_EXPLANATION = (
    "அறிவியல் மற்றும் பண்பாடு அடிப்படையில் ஆளுமைகளின் செறிவு மற்றும் முக்கியத்துவத்தை "
    "மேம்படுத்துவதற்கு முயன்றுள்ளது."
)

# The real code-mixed text from V4_64.
_V4_64_CODE_MIXED_CN = "இது மனித diversitéன் ஒரு இயற்கையான பகுதியாகும்."

# The real mismatched explanation from V4_668 (identity mockery, not disease).
_V4_668_DISEASE_MISMATCH_EXPLANATION = "குறிப்பிட்ட கோளாறு தொடர்பான தவறான கருத்துக்களை நேரடியாக எதிர்த்துள்ளது."


def _minimal_judge_plan_dict(final_response_plan="plan", approved_evidence=None, cultural_guidance=None):
    return {
        "selected_language": "ta", "core_claim_to_counter": "claim", "recommended_strategy": [],
        "approved_evidence": approved_evidence or [], "rejected_content": [],
        "cultural_guidance": cultural_guidance or [], "safety_guidance": [],
        "final_response_plan": final_response_plan,
    }


# ---------------------------------------------------------------------------
# 1-3. Style existence / isolation
# ---------------------------------------------------------------------------
def test_v22kg1_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg1"])
    assert args.final_cn_style == "v22kg1"


def test_v22kg_remains_unchanged_by_v22kg1_addition():
    plan = _minimal_judge_plan_dict()
    prompt = prompts.build_final_cn_prompt("comment", "ta", plan, style="v22kg")
    assert "PRIMARY GROUNDING CONTRACT" not in prompt, "v22kg must not gain v22kg1's stronger header"
    assert "exactly 1 short, simple, complete sentence" not in prompt, "v22kg must keep the 2-sentence Tamil rule"
    assert "exactly 2 short, simple, complete sentences" in prompt


def test_old_styles_v20_through_v22b6_remain_unchanged_by_v22kg1_addition():
    plan = _minimal_judge_plan_dict()
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "en", plan, style=style)
        assert "KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg1"


# ---------------------------------------------------------------------------
# 4-7. v22kg1 prompt content
# ---------------------------------------------------------------------------
def test_v22kg1_prompt_includes_kg_grounding_block():
    prompt = prompts.build_final_cn_prompt("comment", "ta", _minimal_judge_plan_dict(), style="v22kg1")
    assert "COUNTER-NARRATIVE KNOWLEDGE GRAPH" in prompt
    assert "FORBIDDEN CLAIMS" in prompt
    assert "REQUIRED REBUTTAL" in prompt


def test_v22kg1_prompt_states_kg_is_primary_grounding_contract():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg1")
    assert "PRIMARY GROUNDING CONTRACT" in prompt
    assert "PRIMARY grounding contract for this" in prompt


def test_v22kg1_prompt_states_final_response_plan_is_suggestion_not_evidence():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg1")
    assert "non-evidence SUGGESTION" in prompt
    assert "itself a source of facts" in prompt


def test_v22kg1_prompt_states_forbidden_claims_override_final_response_plan():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg1")
    assert "ignore that specific" in prompt
    assert "conflicting part entirely" in prompt


# ---------------------------------------------------------------------------
# 8-9. Tamil 1-sentence CN / 1-sentence explanation instruction
# ---------------------------------------------------------------------------
def test_v22kg1_tamil_prompt_requires_exactly_one_sentence_counter_narrative():
    prompt = prompts.build_final_cn_prompt("comment", "ta", _minimal_judge_plan_dict(), style="v22kg1")
    assert "exactly 1 short, simple, complete sentence" in prompt
    assert "exactly 2 short, simple, complete sentences" not in prompt
    assert "counter_narrative MUST be exactly 1 short" in prompt


def test_v22kg1_tamil_prompt_requires_exactly_one_sentence_explanation():
    prompt = prompts.build_final_cn_prompt("comment", "ta", _minimal_judge_plan_dict(), style="v22kg1")
    assert "explanation must be exactly 1 short Tamil sentence, method-level only" in prompt


def test_v22kg1_non_tamil_language_keeps_v22b6_sentence_count():
    """The 1-sentence rule is Tamil-only per the task spec - other languages keep v22b6's
    normal 2-3 sentence structure."""
    prompt = prompts.build_final_cn_prompt("comment", "it", _minimal_judge_plan_dict(), style="v22kg1")
    assert "2 to 3 complete sentences" in prompt


# ---------------------------------------------------------------------------
# 10-12, 22-24. kg_consistency checks all three text sources + risk level rules
# ---------------------------------------------------------------------------
def test_kg_consistency_full_flags_final_response_plan_only_as_low_risk_when_output_avoids_it():
    kg = build_counter_narrative_kg("curse comment", "ta", approved_evidence=_V4_67_APPROVED_EVIDENCE)
    plan = {"final_response_plan": "religion accepts everyone", "cultural_guidance": []}
    result = validate_kg_consistency_full(kg, plan, counter_narrative="தெளிவான உரை",
                                           explanation="தெளிவான விளக்கம்")
    assert result["final_response_plan_violations"]
    assert result["counter_narrative_violations"] == []
    assert result["explanation_violations"] == []
    assert result["risk_level"] == "low"


def test_kg_consistency_full_flags_counter_narrative_violation_as_high_risk():
    kg = build_counter_narrative_kg("curse comment", "ta", approved_evidence=_V4_67_APPROVED_EVIDENCE)
    plan = {"final_response_plan": "clean plan", "cultural_guidance": []}
    result = validate_kg_consistency_full(
        kg, plan, counter_narrative=_V4_67_LEAKED_COUNTER_NARRATIVE, explanation="தெளிவான விளக்கம்",
    )
    assert result["counter_narrative_violations"]
    assert "unsupported_religion_generalization" in result["forbidden_claims_triggered"]
    assert result["risk_level"] == "high"


def test_kg_consistency_full_flags_explanation_violation_as_medium_risk():
    kg = build_counter_narrative_kg("curse comment", "ta", approved_evidence=_V4_67_APPROVED_EVIDENCE)
    plan = {"final_response_plan": "clean plan", "cultural_guidance": []}
    result = validate_kg_consistency_full(
        kg, plan, counter_narrative="தெளிவான உரை", explanation=_V4_67_LEAKED_EXPLANATION,
    )
    assert result["explanation_violations"]
    assert result["counter_narrative_violations"] == []
    assert result["risk_level"] == "medium"


def test_kg_consistency_full_returns_none_risk_when_everything_is_clean():
    kg = build_counter_narrative_kg("crime comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "respond with dignity", "cultural_guidance": []}
    result = validate_kg_consistency_full(kg, plan, counter_narrative="clean text", explanation="clean explanation")
    assert result["risk_level"] == "none"
    assert result["final_response_plan_violations"] == []
    assert result["counter_narrative_violations"] == []
    assert result["explanation_violations"] == []


# ---------------------------------------------------------------------------
# 13-15, 17. Broader Tamil religion/theology/science triggers
# ---------------------------------------------------------------------------
def test_tamil_matham_obbukkollal_triggers_unsupported_religion_generalization():
    kg = build_counter_narrative_kg("curse comment", "ta", approved_evidence=_V4_67_APPROVED_EVIDENCE)
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_full(
        kg, plan, counter_narrative="மதங்கள் அன்பு மற்றும் ஒப்புக்கொள்ளல் ஊக்குவிக்கின்றன.", explanation="",
    )
    assert "unsupported_religion_generalization" in result["forbidden_claims_triggered"]


def test_tamil_matha_maraibugalin_ulladakkiya_thanmai_triggers_unsupported_religion_generalization():
    kg = build_counter_narrative_kg("curse comment", "ta", approved_evidence=_V4_67_APPROVED_EVIDENCE)
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_full(
        kg, plan, counter_narrative="", explanation="மத மரபுகளின் உள்ளடக்கிய தன்மையை மேற்கோள் காட்டப்பட்டது.",
    )
    assert "unsupported_religion_generalization" in result["forbidden_claims_triggered"]


def test_tamil_kadavul_triggers_unsupported_theological_claim_when_unsupported():
    kg = build_counter_narrative_kg("curse comment", "ta", approved_evidence=_V4_67_APPROVED_EVIDENCE)
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_full(kg, plan, counter_narrative="கடவுள் அனைவரையும் நேசிக்கிறார்.", explanation="")
    assert "unsupported_theological_claim" in result["forbidden_claims_triggered"]


def test_tamil_science_and_culture_basis_phrase_triggers_unsupported_science_claim():
    kg = build_counter_narrative_kg("comment", "ta", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_full(kg, plan, counter_narrative="", explanation=_V4_64_LEAKED_EXPLANATION)
    assert "unsupported_science_claim" in result["forbidden_claims_triggered"]


# ---------------------------------------------------------------------------
# 16. V4_67 safe fallback anchor is clean
# ---------------------------------------------------------------------------
def test_v4_67_safe_fallback_anchor_contains_no_religion_or_god_wording():
    anchor = prompts._V22KG1_TAMIL_ONE_SENTENCE_ANCHORS["curse_or_religion_framing"]
    for banned in ["மதம்", "மதங்கள்", "கடவுள்", "religion", "God"]:
        assert banned not in anchor, f"safe fallback anchor must not contain {banned!r}"


# ---------------------------------------------------------------------------
# 18-19. Code-mixing detection
# ---------------------------------------------------------------------------
def test_diversite_suffixed_is_detected_as_code_mixing():
    found = _detect_code_mixing("இது ஒரு diversitéன் பகுதியாகும்.")
    assert any(tok.lower() == "diversité" for tok in found)


def test_manidha_diversite_suffixed_is_detected_as_code_mixing():
    found = _detect_code_mixing(_V4_64_CODE_MIXED_CN)
    assert any(tok.lower() == "diversité" for tok in found)


def test_lgbtq_plus_is_whitelisted_not_flagged_as_code_mixing():
    found = _detect_code_mixing("LGBTQ+ மக்களுக்கு மரியாதை தேவை.")
    assert found == []


def test_v22kg1_kg_consistency_flags_code_mixing_in_counter_narrative():
    kg = build_counter_narrative_kg("silencing comment", "ta", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_full(kg, plan, counter_narrative=_V4_64_CODE_MIXED_CN, explanation="")
    code_mix_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "code_mixing"]
    assert code_mix_hits
    assert result["risk_level"] == "high"


# ---------------------------------------------------------------------------
# 20-21. Disease/disorder wording vs claim_type mismatch (V4_668)
# ---------------------------------------------------------------------------
def test_disease_wording_blocked_for_identity_mockery_claim_type():
    kg = build_counter_narrative_kg("Mocking someone's identity.", "ta", approved_evidence=[])
    assert kg["kg_summary"]["claim_type"] == "identity_mockery"
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_full(
        kg, plan, counter_narrative="", explanation=_V4_668_DISEASE_MISMATCH_EXPLANATION,
    )
    mismatch_hits = [v for v in result["explanation_violations"] if v["forbidden_claim"] == "disease_claim_type_mismatch"]
    assert mismatch_hits


def test_disease_wording_allowed_for_disease_or_pathology_claim_type():
    kg = build_counter_narrative_kg("comment", "ta", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_full(
        kg, plan, counter_narrative="", explanation="இது ஒரு நோய் அல்ல.", claim_type="disease_or_pathology",
    )
    mismatch_hits = [v for v in result["explanation_violations"] if v["forbidden_claim"] == "disease_claim_type_mismatch"]
    assert mismatch_hits == []


# ---------------------------------------------------------------------------
# 25-26. Trace wiring - v22kg1 gets cn_knowledge_graph + the 3-way kg_consistency
# ---------------------------------------------------------------------------
class _StubFinalCNClient(ModelClient):
    """Same safe-output combo already proven not to trip echo/off-topic/fidelity guards."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        if response_schema and "fidelity" in response_schema:
            return {"fidelity": "on_target", "reason": "stub - not under test here"}
        return {"counter_narrative": "Being gay is a natural part of human diversity, observed across cultures and history.",
                "explanation": "Used a myth-correction strategy grounded in biology."}


def test_final_cn_agent_v22kg1_return_dict_includes_kg_and_extended_consistency():
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v22kg1"
        plan = JudgePlan(selected_language="en", core_claim_to_counter="being gay is unnatural",
                          final_response_plan="plan")
        agent = FinalCNAgent(_StubFinalCNClient())
        result = agent.run("Being gay is unnatural and against nature.", plan, region=None, case_analysis=None)
    finally:
        prompts.FINAL_CN_STYLE = original_style

    assert "cn_knowledge_graph" in result
    assert "kg_consistency" in result
    assert set(result["kg_consistency"].keys()) == {
        "final_response_plan_violations", "counter_narrative_violations",
        "explanation_violations", "forbidden_claims_triggered", "risk_level",
    }


_FAKE_RAG_RECORD = {
    "corpus_id": 1, "dataset_name": "mock-corpus", "language": "ta", "_score": 0.9,
    "knowledge_text": "Mock factual evidence for testing.", "reference_counter_narrative": "",
    "_query_used": "mock query", "_query_language": "input_language",
}


def _fake_retrieve_bilingual(**kwargs):
    return [dict(_FAKE_RAG_RECORD)]


class _MockPipelineClient(ModelClient):
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


def test_pipeline_trace_includes_extended_kg_consistency_for_v22kg1(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "retrieve_bilingual", _fake_retrieve_bilingual)
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v22kg1"
        client = _MockPipelineClient()
        pipeline = NewArchPipeline(client, backend_name="mock", model_name="mock")
        record = InputRecord(text="Being gay is unnatural.", id="kg1_1", language_hint="ta", rag_mode="dual_rag")
        trace = pipeline.generate(record, filter_target=True)
    finally:
        prompts.FINAL_CN_STYLE = original_style

    assert "cn_knowledge_graph" in trace
    assert "kg_consistency" in trace
    assert "final_response_plan_violations" in trace["kg_consistency"]
    assert "counter_narrative_violations" in trace["kg_consistency"]
    assert "explanation_violations" in trace["kg_consistency"]
    assert "prosecutor_argument" not in str(trace)


# ---------------------------------------------------------------------------
# 27. Mistral tokenizer fix remains unchanged
# ---------------------------------------------------------------------------
def test_mistral_tokenizer_fix_unchanged_by_v22kg1_work():
    from model_api import _hf_transformers_tokenizer_kwargs, _hf_transformers_model_auth_kwargs
    tok_kwargs = _hf_transformers_tokenizer_kwargs("gghfez/Mistral-Small-3.2-24B-Instruct-hf", hf_token="")
    assert tok_kwargs.get("fix_mistral_regex") is True
    model_kwargs = _hf_transformers_model_auth_kwargs(hf_token="tok123")
    assert "fix_mistral_regex" not in model_kwargs


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
        print("\ntest_counter_narrative_kg1.py: ALL PASSED")
