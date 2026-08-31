"""
Coverage for v22kg5: the final KG-grounded concise multilingual style, on
top of v22kg4 (v22kg4 itself must stay completely unchanged - see
test_counter_narrative_kg4.py for its own coverage, still passing
untouched).

Motivated by: Spanish/Italian/English v22kg4 outputs were mostly safe and
fluent but too verbose (3-5 sentences, reading like mini-explanations
rather than concise counter-narratives), and English specifically sometimes
drifted into unsupported biology/brain/hormone framing, "choose to love"
wording that can imply orientation is a choice, and unnecessary broadening
from sexual orientation to gender identity. Tamil (v22kg1) and Basque
(v22kg4) are explicitly NOT re-iterated - their prompt text is duplicated
unchanged, per the instruction that both are already the best versions so
far.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import prompts
from counter_narrative_kg import build_counter_narrative_kg, validate_kg_consistency_v4, has_supported_claim_family
from model_api import ModelClient
from schemas import JudgePlan
from final_cn_agent import FinalCNAgent

_SAMPLE_IDS = ["EU125", "EU130", "EU133", "V4_67", "V4_64", "V4_534", "V4_668", "EN944"]


def _minimal_judge_plan_dict(language="en", final_response_plan="plan", approved_evidence=None, cultural_guidance=None):
    return {
        "selected_language": language, "core_claim_to_counter": "claim", "recommended_strategy": [],
        "approved_evidence": approved_evidence or [], "rejected_content": [],
        "cultural_guidance": cultural_guidance or [], "safety_guidance": [],
        "final_response_plan": final_response_plan,
    }


# ---------------------------------------------------------------------------
# 1-4. Style existence / isolation
# ---------------------------------------------------------------------------
def test_v22kg5_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg5"])
    assert args.final_cn_style == "v22kg5"


def test_v22kg4_remains_unchanged_by_v22kg5_addition():
    plan = _minimal_judge_plan_dict("eu")
    prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style="v22kg4")
    assert "must stay 1-3 sentences" in prompt, "v22kg4 must keep v22b6's original explanation-length text"
    plan_en = _minimal_judge_plan_dict("en")
    prompt_en = prompts.build_final_cn_prompt("comment", "en", plan_en, style="v22kg4")
    idx = prompt_en.find("STRUCTURE")
    assert "2 to 3 complete sentences" in prompt_en[idx:idx + 60]
    assert "AVOID biology, brain-development" not in prompt_en


def test_v22kg1_v22kg2_v22kg3_remain_unchanged_by_v22kg5_addition():
    plan_ta = _minimal_judge_plan_dict("ta")
    p1 = prompts.build_final_cn_prompt("comment", "ta", plan_ta, style="v22kg1")
    assert "IT125-style case" in p1
    plan_eu = _minimal_judge_plan_dict("eu")
    p2 = prompts.build_final_cn_prompt("comment", "eu", plan_eu, style="v22kg2")
    assert "IT125-style case" in p2
    p3 = prompts.build_final_cn_prompt("comment", "eu", plan_eu, style="v22kg3")
    assert "FALLBACK (WHEN UNSURE)" not in p3


def test_old_styles_v20_through_v22b6_remain_unchanged_by_v22kg5_addition():
    plan = _minimal_judge_plan_dict("en")
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "en", plan, style=style)
        assert "KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg5"


# ---------------------------------------------------------------------------
# 5-7. No sample IDs; style anchors, not templates
# ---------------------------------------------------------------------------
def test_v22kg5_prompt_contains_no_sample_ids():
    for language in ["en", "es", "it", "ta", "eu"]:
        plan = _minimal_judge_plan_dict(language)
        prompt = prompts.build_final_cn_prompt("comment", language, plan, style="v22kg5")
        for sample_id in _SAMPLE_IDS:
            assert sample_id not in prompt, f"v22kg5 prompt ({language}) must not contain sample ID {sample_id}"
        assert "IT125-style" not in prompt
        assert "IT133-style" not in prompt


def test_v22kg5_prompt_says_style_anchors_not_fixed_templates():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert "STYLE ANCHORS ONLY" in prompt
    assert "NOT fixed templates" in prompt
    assert "Do not hardcode a sample-specific answer" in prompt


def test_v22kg5_prompt_says_not_to_copy_blindly():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert "Do not copy an example verbatim unless it genuinely and exactly fits" in prompt
    assert "Do not force every output into the same sentence pattern" in prompt


# ---------------------------------------------------------------------------
# 8-10. KG obedience / evidence-conditional statements
# ---------------------------------------------------------------------------
def test_v22kg5_prompt_says_kg_is_primary_grounding_contract():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert "PRIMARY GROUNDING CONTRACT" in prompt


def test_v22kg5_prompt_says_judge_plan_is_suggestion_not_evidence():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert "non-evidence SUGGESTION" in prompt
    assert "itself a source of facts" in prompt


def test_v22kg5_prompt_says_claims_allowed_only_with_approved_evidence():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert "ARE NOT BANNED" in prompt
    assert "APPROVED FACTUAL/WEB EVIDENCE nodes are the ONLY basis for factual, legal, scientific" in prompt


# ---------------------------------------------------------------------------
# 11-16. Length rules
# ---------------------------------------------------------------------------
def test_tamil_v22kg5_requires_exactly_one_cn_sentence():
    prompt = prompts.build_final_cn_prompt("comment", "ta", _minimal_judge_plan_dict("ta"), style="v22kg5")
    assert "exactly 1 short, simple, complete sentence" in prompt


def test_basque_v22kg5_requires_exactly_one_cn_sentence():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict("eu"), style="v22kg5")
    assert "counter_narrative MUST be exactly 1 short" in prompt


def test_spanish_v22kg5_allows_max_two_short_sentences():
    prompt = prompts.build_final_cn_prompt("comment", "es", _minimal_judge_plan_dict("es"), style="v22kg5")
    idx = prompt.find("STRUCTURE")
    assert "1 to 2 short, concise sentences (2 at the absolute most" in prompt[idx:idx + 100]


def test_italian_v22kg5_allows_max_two_short_sentences():
    prompt = prompts.build_final_cn_prompt("comment", "it", _minimal_judge_plan_dict("it"), style="v22kg5")
    idx = prompt.find("STRUCTURE")
    assert "1 to 2 short, concise sentences (2 at the absolute most" in prompt[idx:idx + 100]


def test_english_v22kg5_allows_max_two_short_sentences():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    idx = prompt.find("STRUCTURE")
    assert "1 to 2 short, concise sentences (2 at the absolute most" in prompt[idx:idx + 100]


def test_explanation_requires_exactly_one_sentence_for_all_languages():
    for language in ["en", "es", "it", "ta", "eu"]:
        prompt = prompts.build_final_cn_prompt("comment", language, _minimal_judge_plan_dict(language), style="v22kg5")
        assert "must stay exactly 1 short, method-level sentence" in prompt, f"{language} missing explanation-length rule"


# ---------------------------------------------------------------------------
# 17-19. English-specific guardrails
# ---------------------------------------------------------------------------
def test_english_prompt_discourages_biology_hormone_science_framing():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert "AVOID biology, brain-development, hormone-response" in prompt
    assert "unless an APPROVED\n  FACTUAL/WEB EVIDENCE item explicitly supports that exact claim" in prompt or \
           "unless an APPROVED FACTUAL/WEB EVIDENCE item explicitly supports that exact claim" in prompt.replace("\n  ", " ")


def test_english_prompt_discourages_choose_to_love_wording():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert '"choose to love"' in prompt
    assert "could imply sexual orientation itself is a\n  choice" in prompt or "could imply sexual orientation itself is a" in prompt


def test_english_prompt_discourages_target_broadening():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg5")
    assert "Do NOT broaden the target from sexual orientation to gender identity" in prompt


# ---------------------------------------------------------------------------
# 20-21. Basque "dio" / Italian "Dio" (unchanged from v3, reused by v4)
# ---------------------------------------------------------------------------
def test_basque_dio_is_not_flagged_as_theology():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="Berak dio hori garrantzitsua dela.",
                                         explanation="")
    theology_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_theological_claim"]
    assert theology_hits == []


def test_italian_dio_can_still_be_flagged_as_theology_when_unsupported():
    kg = build_counter_narrative_kg("comment", "it", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="Dio ama tutti secondo la tradizione.",
                                         explanation="")
    theology_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_theological_claim"]
    assert theology_hits


# ---------------------------------------------------------------------------
# 22. English negated phrase not falsely flagged
# ---------------------------------------------------------------------------
def test_english_avoids_unsupported_scientific_claims_not_falsely_flagged():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(
        kg, plan, counter_narrative="", explanation="The response avoids unsupported scientific claims.",
    )
    science_hits = [v for v in result["explanation_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert science_hits == []


# ---------------------------------------------------------------------------
# 23-24. Evidence-conditional logic remains intact
# ---------------------------------------------------------------------------
def test_evidence_supported_science_claim_is_not_flagged():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "Scientific evidence shows X.", "source_type": "fact"},
    ])
    assert has_supported_claim_family(kg, "science_research") is True
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="Science shows this is true.", explanation="")
    assert result["counter_narrative_violations"] == []


def test_cultural_only_science_wording_not_treated_as_factual_proof():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[
        {"source_id": "1", "passage": "supported by science and psychology.", "source_type": "cultural"},
    ])
    assert has_supported_claim_family(kg, "science_research") is False
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="This is supported by science.", explanation="")
    science_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert science_hits


# ---------------------------------------------------------------------------
# 25. kg_consistency checks all 3 fields
# ---------------------------------------------------------------------------
def test_v4_validator_checks_final_response_plan_counter_narrative_and_explanation():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "science shows this", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="text", explanation="text")
    assert set(result.keys()) == {
        "final_response_plan_violations", "counter_narrative_violations",
        "explanation_violations", "forbidden_claims_triggered", "risk_level",
    }


# ---------------------------------------------------------------------------
# 26. Mistral tokenizer fix remains unchanged
# ---------------------------------------------------------------------------
def test_mistral_tokenizer_fix_unchanged_by_v22kg5_work():
    from model_api import _hf_transformers_tokenizer_kwargs, _hf_transformers_model_auth_kwargs
    tok_kwargs = _hf_transformers_tokenizer_kwargs("gghfez/Mistral-Small-3.2-24B-Instruct-hf", hf_token="")
    assert tok_kwargs.get("fix_mistral_regex") is True
    model_kwargs = _hf_transformers_model_auth_kwargs(hf_token="tok123")
    assert "fix_mistral_regex" not in model_kwargs


# ---------------------------------------------------------------------------
# Trace wiring sanity check (FinalCNAgent-level)
# ---------------------------------------------------------------------------
class _StubFinalCNClient(ModelClient):
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        if response_schema and "fidelity" in response_schema:
            return {"fidelity": "on_target", "reason": "stub"}
        return {"counter_narrative": "Being gay is a natural part of human diversity, observed across cultures and history.",
                "explanation": "Used a myth-correction strategy grounded in biology."}


def test_final_cn_agent_v22kg5_return_dict_includes_kg_and_v4_consistency():
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v22kg5"
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


if __name__ == "__main__":
    import traceback

    failures = []
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            try:
                fn()
                print(f"{name}: PASSED")
            except Exception:
                failures.append(name)
                print(f"{name}: FAILED")
                traceback.print_exc()
    if failures:
        print(f"\n{len(failures)} FAILED: {failures}")
    else:
        print("\ntest_counter_narrative_kg5.py: ALL PASSED")
