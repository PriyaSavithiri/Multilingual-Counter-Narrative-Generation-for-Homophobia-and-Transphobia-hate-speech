"""
Coverage for v22kg7: v22kg6 + a small, deterministic, no-LLM-call KG safety
fallback (English only for now). v22kg6 itself must stay completely
unchanged - see test_counter_narrative_kg6.py for its own coverage, still
passing untouched. The v22kg7 prompt is byte-identical to v22kg6's (reused
directly via the same function, no new prompt code).

Motivated by 4 real v22kg6 English findings: cultural-only brain/biology/
hormone wording still leaked through (v22kg6's validator never had those
triggers); a child/abortion case used mental-disorder evidence instead of a
direct child-dignity rebuttal; unsupported "not supported by science"
wording appeared; and a genuinely relevant mental-disorder rebuttal (hate
comment said "insane", which the shared claim_type classifier doesn't
recognize as disease_or_pathology) was wrongly flagged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import prompts
from counter_narrative_kg import (
    build_counter_narrative_kg, validate_kg_consistency_v3, validate_kg_consistency_v4,
    validate_kg_consistency_v5, apply_kg_safety_fallback, _select_english_kg_safety_fallback_category,
)
from model_api import ModelClient
from schemas import JudgePlan
from final_cn_agent import FinalCNAgent

_SAMPLE_IDS = [
    "EN944", "EN125", "EN130", "EN133", "ML-MTCONAN", "Codabench",
    "EU125", "EU130", "EU133", "V4_67", "V4_64", "V4_534", "V4_668",
]


def _minimal_judge_plan_dict(language="en", final_response_plan="plan", approved_evidence=None, cultural_guidance=None):
    return {
        "selected_language": language, "core_claim_to_counter": "claim", "recommended_strategy": [],
        "approved_evidence": approved_evidence or [], "rejected_content": [],
        "cultural_guidance": cultural_guidance or [], "safety_guidance": [],
        "final_response_plan": final_response_plan,
    }


# ---------------------------------------------------------------------------
# 1. v22kg7 exists and is selectable
# ---------------------------------------------------------------------------
def test_v22kg7_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg7"])
    assert args.final_cn_style == "v22kg7"


# ---------------------------------------------------------------------------
# 2-4. Old styles unchanged
# ---------------------------------------------------------------------------
def test_v22kg6_remains_unchanged_by_v22kg7_addition():
    plan = _minimal_judge_plan_dict("en")
    p6 = prompts.build_final_cn_prompt("comment", "en", plan, style="v22kg6")
    p7 = prompts.build_final_cn_prompt("comment", "en", plan, style="v22kg7")
    assert p6 == p7, "v22kg7 prompt must be byte-identical to v22kg6's"


def test_v22kg5_remains_unchanged_by_v22kg7_addition():
    plan = _minimal_judge_plan_dict("en")
    prompt = prompts.build_final_cn_prompt("comment", "en", plan, style="v22kg5")
    assert "HARD BAN" not in prompt


def test_v22kg4_and_older_styles_remain_unchanged_by_v22kg7_addition():
    plan = _minimal_judge_plan_dict("en")
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "en", plan, style=style)
        assert "KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg7"
    plan_eu = _minimal_judge_plan_dict("eu")
    p_kg3 = prompts.build_final_cn_prompt("comment", "eu", plan_eu, style="v22kg3")
    assert "FALLBACK (WHEN UNSURE)" not in p_kg3  # that's v22kg4's own Basque fallback section, unrelated


# ---------------------------------------------------------------------------
# 5-6. KG primary contract, no sample IDs
# ---------------------------------------------------------------------------
def test_v22kg7_uses_kg_as_primary_grounding_contract():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict("en"), style="v22kg7")
    assert "PRIMARY GROUNDING CONTRACT" in prompt


def test_v22kg7_prompt_contains_no_sample_ids():
    for language in ["en", "es", "it", "ta", "eu"]:
        plan = _minimal_judge_plan_dict(language)
        prompt = prompts.build_final_cn_prompt("comment", language, plan, style="v22kg7")
        for sample_id in _SAMPLE_IDS:
            assert sample_id not in prompt, f"v22kg7 prompt ({language}) must not contain {sample_id}"
        assert "IT125-style" not in prompt
        assert "IT133-style" not in prompt


# ---------------------------------------------------------------------------
# 7. v22kg7 reuses v22kg6 prompt behaviour
# ---------------------------------------------------------------------------
def test_v22kg7_reuses_v22kg6_prompt_behaviour():
    for language in ["en", "es", "it", "ta", "eu"]:
        plan = _minimal_judge_plan_dict(language)
        p6 = prompts.build_final_cn_prompt("comment", language, plan, style="v22kg6")
        p7 = prompts.build_final_cn_prompt("comment", language, plan, style="v22kg7")
        assert p6 == p7, f"{language} prompt must match v22kg6 exactly"


# ---------------------------------------------------------------------------
# 8-10. Validator routing
# ---------------------------------------------------------------------------
def test_v22kg7_routes_to_validate_kg_consistency_v5():
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
        prompts.FINAL_CN_STYLE = "v22kg7"
        plan = JudgePlan(selected_language="en", core_claim_to_counter="being gay is unnatural",
                          final_response_plan="plan")
        agent = FinalCNAgent(_StubFinalCNClient())
        result = agent.run("Being gay is unnatural and against nature.", plan, region=None, case_analysis=None)
    finally:
        prompts.FINAL_CN_STYLE = original_style

    assert "cn_knowledge_graph" in result
    assert "kg_consistency" in result
    assert "kg_safety_fallback" in result
    assert set(result["kg_consistency"].keys()) == {
        "final_response_plan_violations", "counter_narrative_violations",
        "explanation_violations", "forbidden_claims_triggered", "risk_level",
    }


def test_v22kg3_v22kg4_still_route_to_v3():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    # v3 must NOT flag bare brain/hormone wording (v5-only extension)
    result = validate_kg_consistency_v3(kg, plan, counter_narrative="caused by brain structure", explanation="")
    hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert hits == []


def test_v22kg5_v22kg6_still_route_to_v4():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v4(kg, plan, counter_narrative="caused by brain structure", explanation="")
    hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert hits == [], "v4 must remain unchanged - no brain/biology triggers"


# ---------------------------------------------------------------------------
# 11-14. v5 validator behavior
# ---------------------------------------------------------------------------
def test_v5_flags_unsupported_brain_biology_hormone_wording():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v5(
        kg, plan, counter_narrative="This is caused by brain structure and hormone response, not choice.",
        explanation="",
    )
    hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert hits


def test_v5_does_not_flag_negated_explanation_phrases():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v5(
        kg, plan, counter_narrative="", explanation="The response avoids unsupported biological claims.",
    )
    hits = [v for v in result["explanation_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert hits == []


def test_v5_allows_mental_disorder_rebuttal_when_hate_comment_says_insane():
    kg = build_counter_narrative_kg("Gays are insane and mentally unfit.", "en", approved_evidence=[])
    assert kg["kg_summary"]["claim_type"] != "disease_or_pathology"  # confirms the real classifier gap
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v5(
        kg, plan, counter_narrative="Homosexuality is not a mental disorder; every person deserves dignity and respect.",
        explanation="",
    )
    mismatch_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "disease_claim_type_mismatch"]
    assert mismatch_hits == []


def test_v5_still_flags_mental_disorder_wording_in_unrelated_child_family_case():
    kg = build_counter_narrative_kg("I would reject my child if they were gay.", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v5(
        kg, plan, counter_narrative="Homosexuality is not a mental disorder; every person deserves dignity and respect.",
        explanation="",
    )
    mismatch_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "disease_claim_type_mismatch"]
    assert mismatch_hits


# ---------------------------------------------------------------------------
# 15-17. Fallback application logic
# ---------------------------------------------------------------------------
def test_fallback_applies_when_cn_has_high_risk_violation():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="This is caused by brain structure and hormone response.",
        explanation="", comment="comment",
    )
    assert result["kg_safety_fallback"]["applied"] is True
    assert result["counter_narrative"] != "This is caused by brain structure and hormone response."


def test_fallback_does_not_apply_when_only_plan_has_low_risk_violation():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "science shows this is true", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="Everyone deserves dignity and respect.",
        explanation="The response directly rejects the harmful claim and emphasizes dignity and respect.",
        comment="comment",
    )
    assert result["kg_safety_fallback"]["applied"] is False
    assert result["counter_narrative"] == "Everyone deserves dignity and respect."


def test_explanation_only_violation_replaces_explanation_but_keeps_cn():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    original_cn = "Everyone deserves dignity and respect regardless of orientation."
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative=original_cn,
        explanation="This is a pseudoscientific and biological claim.", comment="comment",
    )
    assert result["kg_safety_fallback"]["applied"] is True
    assert result["kg_safety_fallback"]["fallback_type"] == "explanation_only"
    assert result["counter_narrative"] == original_cn
    assert result["explanation"] != "This is a pseudoscientific and biological claim."


# ---------------------------------------------------------------------------
# 18-22. Claim-type fallback selection
# ---------------------------------------------------------------------------
def test_child_abortion_fallback_returns_child_love_respect_cn():
    kg = build_counter_narrative_kg("I would consider abortion if my child turned out gay.", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="Science shows sexual orientation is not chosen.", explanation="",
        comment="I would consider abortion if my child turned out gay.",
    )
    assert result["kg_safety_fallback"]["fallback_type"] == "child_family_rejection"
    assert "every child deserves love and respect" in result["counter_narrative"]


def test_conversion_fallback_returns_not_a_condition_to_be_freed_from_cn():
    kg = build_counter_narrative_kg("Gay people should be cured and freed from homosexuality.", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="Brain and hormone research supports change therapy being harmful.",
        explanation="", comment="Gay people should be cured and freed from homosexuality.",
    )
    assert result["kg_safety_fallback"]["fallback_type"] == "conversion_change_framing"
    assert "not a condition to be freed from" in result["counter_narrative"]


def test_same_sex_relationship_fallback_avoids_choose_to_love():
    kg = build_counter_narrative_kg("Is it okay to be in a same-sex relationship? Very questionable marriage.", "en",
                                     approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="This is not supported by science.", explanation="",
        comment="Is it okay to be in a same-sex relationship? Very questionable marriage.",
    )
    assert result["kg_safety_fallback"]["fallback_type"] == "same_sex_relationship_stigma"
    assert "choose to love" not in result["counter_narrative"]


def test_trans_biology_fallback_avoids_brain_hormone_biology_science_wording():
    kg = build_counter_narrative_kg("Transgender people have brain and hormone abnormalities.", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="This is caused by brain structure and hormone levels.", explanation="",
        comment="Transgender people have brain and hormone abnormalities.",
    )
    assert result["kg_safety_fallback"]["fallback_type"] == "trans_biology_pseudoscience"
    for banned in ["brain", "hormone", "biological", "neurodevelopment", "scientific", "pseudoscientific"]:
        assert banned not in result["counter_narrative"].lower()


def test_disease_pathology_fallback_is_evidence_conditional():
    # Tests the selection heuristic directly (not the full apply_kg_safety_fallback flow) - once
    # fact evidence supports the science family, a science-worded CN stops being a violation at
    # all, so no fallback would trigger in the first place; the category-selection logic itself
    # is what must be evidence-conditional, and that's what this test isolates.
    kg_unsupported = build_counter_narrative_kg("Gays are insane.", "en", approved_evidence=[])
    assert _select_english_kg_safety_fallback_category(kg_unsupported, "Gays are insane.") == "disease_pathology_unsupported"

    kg_supported = build_counter_narrative_kg("Gays are insane.", "en", approved_evidence=[
        {"source_id": "1", "passage": "Scientific research confirms homosexuality is not a mental disorder.",
         "source_type": "fact"},
    ])
    assert _select_english_kg_safety_fallback_category(kg_supported, "Gays are insane.") == "disease_pathology_supported"


# ---------------------------------------------------------------------------
# 23-25. Trace/revalidation
# ---------------------------------------------------------------------------
def test_fallback_result_is_revalidated():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="This is caused by brain structure and hormone response.",
        explanation="", comment="comment",
    )
    assert result["kg_safety_fallback"]["post_fallback_kg_consistency"]["risk_level"] == "none"


def test_trace_stores_original_cn_and_explanation_when_fallback_applied():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    original_cn = "This is caused by brain structure and hormone response."
    original_exp = "This is a scientific claim."
    result = apply_kg_safety_fallback(kg, plan, counter_narrative=original_cn, explanation=original_exp, comment="comment")
    assert result["kg_safety_fallback"]["original_counter_narrative"] == original_cn
    assert result["kg_safety_fallback"]["original_explanation"] == original_exp


def test_trace_stores_fallback_reason_and_type():
    kg = build_counter_narrative_kg("comment", "en", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = apply_kg_safety_fallback(
        kg, plan, counter_narrative="This is caused by brain structure.", explanation="", comment="comment",
    )
    assert result["kg_safety_fallback"]["reason"]
    assert result["kg_safety_fallback"]["fallback_type"] is not None


# ---------------------------------------------------------------------------
# 26. Mistral tokenizer fix remains unchanged
# ---------------------------------------------------------------------------
def test_mistral_tokenizer_fix_unchanged_by_v22kg7_work():
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
        print("\ntest_counter_narrative_kg7.py: ALL PASSED")
