"""
Coverage for v22kg6: a tiny, prompt-only English cleanup on top of v22kg5
(v22kg5 itself must stay completely unchanged - see test_counter_narrative_kg5.py
for its own coverage, still passing untouched). Spanish and Italian already
passed sanity validation and are explicitly out of scope; Tamil (v22kg1)
and Basque (v22kg4) are explicitly not re-iterated.

No new validator - validate_kg_consistency_v4 is reused unchanged, confirmed
sufficient before implementing (already doesn't flag "avoids unsupported
scientific claims" or "without using science/legal/religious/country
claims", already fixes the Basque "dio" false positive, still flags real
unsupported claims).

Motivated by three real v22kg5 English findings: (1) "choose to love"
wording appeared, which can imply orientation is a choice; (2) mental-
disorder/science facts were used as a default rebuttal even for claim types
that were not actually disease/pathology framing; (3) explanations
sometimes used science-coded words ("pseudoscientific assertion") that
could trip kg_consistency even when the counter_narrative itself was clean.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import prompts
from counter_narrative_kg import validate_kg_consistency_v4, build_counter_narrative_kg
from model_api import ModelClient
from schemas import JudgePlan
from final_cn_agent import FinalCNAgent

_SAMPLE_IDS = [
    "EU125", "EU130", "EU133", "V4_67", "V4_64", "V4_534", "V4_668", "EN944",
    "EN125", "EN130", "EN133", "ML-MTCONAN", "Codabench",
]


def _minimal_judge_plan_dict(language="en", final_response_plan="plan", approved_evidence=None, cultural_guidance=None):
    return {
        "selected_language": language, "core_claim_to_counter": "claim", "recommended_strategy": [],
        "approved_evidence": approved_evidence or [], "rejected_content": [],
        "cultural_guidance": cultural_guidance or [], "safety_guidance": [],
        "final_response_plan": final_response_plan,
    }


# ---------------------------------------------------------------------------
# 1. v22kg6 exists and is selectable
# ---------------------------------------------------------------------------
def test_v22kg6_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg6"])
    assert args.final_cn_style == "v22kg6"


# ---------------------------------------------------------------------------
# 2-5. Old styles unchanged
# ---------------------------------------------------------------------------
def test_v22kg5_remains_unchanged_by_v22kg6_addition():
    plan = _minimal_judge_plan_dict("en")
    prompt = prompts.build_final_cn_prompt("comment", "en", plan, style="v22kg5")
    assert "HARD BAN" not in prompt
    assert "EXPLANATION DISCIPLINE" not in prompt
    idx = prompt.find("STRUCTURE")
    assert "1 to 2 short, concise sentences" in prompt[idx:idx + 90]


def test_v22kg4_remains_unchanged_by_v22kg6_addition():
    plan = _minimal_judge_plan_dict("eu")
    prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style="v22kg4")
    assert "must stay 1-3 sentences" in prompt


def test_older_kg_styles_remain_unchanged_by_v22kg6_addition():
    plan_ta = _minimal_judge_plan_dict("ta")
    p1 = prompts.build_final_cn_prompt("comment", "ta", plan_ta, style="v22kg1")
    assert "IT125-style case" in p1
    plan_eu = _minimal_judge_plan_dict("eu")
    p2 = prompts.build_final_cn_prompt("comment", "eu", plan_eu, style="v22kg2")
    assert "IT125-style case" in p2
    p3 = prompts.build_final_cn_prompt("comment", "eu", plan_eu, style="v22kg3")
    assert "FALLBACK (WHEN UNSURE)" not in p3


def test_old_non_kg_styles_remain_unchanged_by_v22kg6_addition():
    plan = _minimal_judge_plan_dict("en")
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "en", plan, style=style)
        assert "KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg6"


# ---------------------------------------------------------------------------
# 6-7. KG obedience preserved
# ---------------------------------------------------------------------------
def test_v22kg6_uses_kg_as_primary_grounding_contract():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "PRIMARY GROUNDING CONTRACT" in prompt


def test_v22kg6_treats_judge_plan_as_suggestion_not_evidence():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "non-evidence SUGGESTION" in prompt
    assert "itself a source of facts" in prompt


# ---------------------------------------------------------------------------
# 8-10. No sample IDs; style anchors, not templates
# ---------------------------------------------------------------------------
def test_v22kg6_prompt_contains_no_sample_ids():
    for language in ["en", "es", "it", "ta", "eu"]:
        plan = _minimal_judge_plan_dict(language)
        prompt = prompts.build_final_cn_prompt("comment", language, plan, style="v22kg6")
        for sample_id in _SAMPLE_IDS:
            assert sample_id not in prompt, f"v22kg6 prompt ({language}) must not contain {sample_id}"
        assert "IT125-style" not in prompt
        assert "IT133-style" not in prompt


def test_v22kg6_examples_marked_as_style_anchors_only():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "STYLE ANCHORS ONLY" in prompt


def test_v22kg6_prompt_says_examples_not_fixed_templates():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "NOT fixed templates" in prompt
    assert "Do not hardcode a sample-specific answer" in prompt


# ---------------------------------------------------------------------------
# 11-16. English content rules
# ---------------------------------------------------------------------------
def test_v22kg6_bans_choose_to_love_and_close_variants():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "HARD BAN" in prompt
    for phrase in ["choose to love", "chooses to love", "chosen love",
                   "who they choose to love", "whom they choose to love"]:
        assert phrase in prompt, f"missing banned phrase {phrase!r}"


def test_v22kg6_english_discourages_mental_disorder_facts_for_child_family_rejection():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "CLAIM-TYPE CONDITIONAL" in prompt
    assert "child/family rejection" in prompt
    assert "Do NOT default to mental-disorder facts for" in prompt


def test_v22kg6_english_has_child_dignity_anchor_for_abortion_family_rejection():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "A child's sexual orientation is never a reason to reject them; every child deserves love and respect." in prompt


def test_v22kg6_english_has_conversion_change_framing_guidance():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "Homosexuality is not a condition to be freed from; every person deserves dignity and respect." in prompt
    assert "No one should be pressured to change who they are; everyone deserves dignity and respect." in prompt


def test_v22kg6_english_has_relationship_stigma_guidance_without_choose_to_love():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "Same-sex relationships are not wrong or shameful; every relationship deserves respect when built on care and consent." in prompt
    # the anchor itself must not contain the banned phrase
    anchor = "Same-sex relationships are not wrong or shameful; every relationship deserves respect when built on care and consent."
    assert "choose to love" not in anchor


def test_v22kg6_english_discourages_target_broadening():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "TARGET SPECIFICITY" in prompt
    assert "do not broaden the target from sexual orientation to gender identity" in prompt


# ---------------------------------------------------------------------------
# 17-18. Explanation discipline
# ---------------------------------------------------------------------------
def test_v22kg6_explanation_rule_prefers_generic_method_level_explanation():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    assert "EXPLANATION DISCIPLINE" in prompt
    assert "The response directly rejects the harmful claim and emphasizes dignity and respect." in prompt


def test_v22kg6_explanation_rule_avoids_science_legal_religious_cultural_words():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg6")
    for word in ["scientific", "pseudoscientific", "research", "biological", "medical",
                 "legal", "religious", "cultural", "country-specific"]:
        assert f'"{word}"' in prompt, f"explanation avoid-list missing {word!r}"


# ---------------------------------------------------------------------------
# 19-22. Length rules preserved
# ---------------------------------------------------------------------------
def test_tamil_v22kg6_still_requires_exactly_one_cn_sentence():
    prompt = prompts.build_final_cn_prompt("comment", "ta", _minimal_judge_plan_dict("ta"), style="v22kg6")
    assert "exactly 1 short, simple, complete sentence" in prompt


def test_basque_v22kg6_still_requires_exactly_one_cn_sentence():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict("eu"), style="v22kg6")
    assert "counter_narrative MUST be exactly 1" in prompt
    assert "everyday Basque sentence" in prompt


def test_spanish_italian_english_v22kg6_still_require_max_two_short_sentences():
    for language in ["es", "it", "en"]:
        prompt = prompts.build_final_cn_prompt("comment", language, _minimal_judge_plan_dict(language), style="v22kg6")
        idx = prompt.find("STRUCTURE")
        assert "1 to 2 short, concise sentences (2 at the absolute most" in prompt[idx:idx + 100]


def test_explanation_exactly_one_sentence_for_all_languages():
    for language in ["en", "es", "it", "ta", "eu"]:
        prompt = prompts.build_final_cn_prompt("comment", language, _minimal_judge_plan_dict(language), style="v22kg6")
        assert "must stay exactly 1 short, method-level sentence" in prompt


# ---------------------------------------------------------------------------
# 23-27. Validator reuse and behavior
# ---------------------------------------------------------------------------
def test_basque_dio_remains_not_flagged_as_theology():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="Berak dio hori garrantzitsua dela.",
                                         explanation="")
    theology_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_theological_claim"]
    assert theology_hits == []


def test_italian_dio_remains_flaggable_as_theology_when_unsupported():
    kg = build_counter_narrative_kg("comment", "it", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="Dio ama tutti secondo la tradizione.",
                                         explanation="")
    theology_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_theological_claim"]
    assert theology_hits


def test_avoids_unsupported_scientific_claims_remains_not_falsely_flagged():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(
        kg, plan, counter_narrative="", explanation="The response avoids unsupported scientific claims.",
    )
    science_hits = [v for v in result["explanation_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert science_hits == []


def test_real_unsupported_science_claim_remains_flagged():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(
        kg, plan, counter_narrative="Science shows this is a real fact.", explanation="",
    )
    science_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert science_hits


def test_final_cn_agent_v22kg6_uses_v4_validator():
    class _StubFinalCNClient(ModelClient):
        backend_name = "stub"

        def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
            if response_schema is None:
                return "n/a"
            if response_schema and "fidelity" in response_schema:
                return {"fidelity": "on_target", "reason": "stub"}
            return {"counter_narrative": "Being gay is a natural part of human diversity, observed across cultures and history.",
                    "explanation": "Used a myth-correction strategy grounded in biology."}

    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v22kg6"
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


# ---------------------------------------------------------------------------
# 28. Mistral tokenizer fix remains unchanged
# ---------------------------------------------------------------------------
def test_mistral_tokenizer_fix_unchanged_by_v22kg6_work():
    from model_api import _hf_transformers_tokenizer_kwargs, _hf_transformers_model_auth_kwargs
    tok_kwargs = _hf_transformers_tokenizer_kwargs("gghfez/Mistral-Small-3.2-24B-Instruct-hf", hf_token="")
    assert tok_kwargs.get("fix_mistral_regex") is True
    model_kwargs = _hf_transformers_model_auth_kwargs(hf_token="tok123")
    assert "fix_mistral_regex" not in model_kwargs


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
        print("\ntest_counter_narrative_kg6.py: ALL PASSED")
