"""
Coverage for v22kg3: Basque one-sentence fluency fix + no-template-overfitting
cleanup, built on top of v22kg2 (v22kg2 itself must stay completely
unchanged - see test_counter_narrative_kg2.py for v22kg2's own coverage,
still passing untouched).

Two motivations, both from real findings this session:

1. Design audit (explicitly requested before writing any v22kg3 code): v22b6's
   shared EXPLANATION-FIELD RULES section - reused unchanged by every KG
   style - names two rules after specific validation-sample IDs ("IT125-style
   case", "IT133-style case"), baked directly into the actual runtime prompt
   text (confirmed via `prompts.build_final_cn_prompt(..., style="v22b6")`,
   not just a comment). v22kg3's own prompt replaces both with generic
   claim-type descriptions; v20-v22b6/v22kg/v22kg1/v22kg2 keep the sample-ID
   wording, per the explicit "old styles unchanged" constraint.
2. Real v22kg2 Basque rerun: fluency problems persisted (unnatural
   "Pertsonak guztiak"/"Pertsoak"/"sexuen independentziak" wording), and the
   validator itself had a false positive - Basque "dio" (an ordinary
   auxiliary verb) was matched against the Italian religion trigger
   "dio"=God, because v22kg2 scanned every language's trigger phrases at once
   regardless of the case's own language.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import prompts
from counter_narrative_kg import build_counter_narrative_kg, validate_kg_consistency_v3, has_supported_claim_family
from model_api import ModelClient
from schemas import JudgePlan
from final_cn_agent import FinalCNAgent

_SAMPLE_IDS = ["EU125", "EU130", "EU133", "V4_67", "V4_64", "V4_534", "V4_668"]


def _minimal_judge_plan_dict(final_response_plan="plan", approved_evidence=None, cultural_guidance=None):
    return {
        "selected_language": "eu", "core_claim_to_counter": "claim", "recommended_strategy": [],
        "approved_evidence": approved_evidence or [], "rejected_content": [],
        "cultural_guidance": cultural_guidance or [], "safety_guidance": [],
        "final_response_plan": final_response_plan,
    }


# ---------------------------------------------------------------------------
# 1-4. Style existence / isolation
# ---------------------------------------------------------------------------
def test_v22kg3_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg3"])
    assert args.final_cn_style == "v22kg3"


def test_v22kg2_remains_unchanged_by_v22kg3_addition():
    plan = _minimal_judge_plan_dict()
    prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style="v22kg2")
    # v22kg2's own sample-specific IT125-style/IT133-style text is untouched (v22b6 base reused
    # unmodified) - only v22kg3's own prompt replaces it.
    assert "IT125-style case" in prompt
    assert "exactly 2 short sentences wherever possible" in prompt
    assert '"Pertsoak"' not in prompt, "v22kg2 must not gain v22kg3's new Basque avoid-list"


def test_v22kg1_remains_unchanged_by_v22kg3_addition():
    plan = dict(_minimal_judge_plan_dict(), selected_language="ta")
    prompt = prompts.build_final_cn_prompt("comment", "ta", plan, style="v22kg1")
    assert "IT125-style case" in prompt
    assert '"Pertsoak"' not in prompt


def test_old_styles_v20_through_v22b6_remain_unchanged_by_v22kg3_addition():
    plan = _minimal_judge_plan_dict()
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style=style)
        assert "KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg3"


# ---------------------------------------------------------------------------
# 5-7. No sample IDs in the v22kg3 runtime prompt; style-anchor discipline stated
# ---------------------------------------------------------------------------
def test_v22kg3_prompt_contains_no_sample_ids():
    plan = _minimal_judge_plan_dict()
    for language in ["eu", "ta", "it", "en"]:
        prompt = prompts.build_final_cn_prompt("comment", language, plan, style="v22kg3")
        for sample_id in _SAMPLE_IDS:
            assert sample_id not in prompt, f"v22kg3 prompt ({language}) must not contain sample ID {sample_id}"
        assert "IT125-style" not in prompt
        assert "IT133-style" not in prompt


def test_v22kg3_prompt_says_examples_are_style_anchors_not_templates():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    assert "STYLE ANCHOR DISCIPLINE" in prompt
    assert "STYLE ANCHORS ONLY" in prompt
    assert "NOT fixed templates" in prompt


def test_v22kg3_prompt_says_not_to_copy_examples_blindly():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    assert "Do not copy an example verbatim unless it genuinely and exactly fits" in prompt
    assert "Do not force every output into the same sentence pattern" in prompt


# ---------------------------------------------------------------------------
# 8-9. Basque one-sentence CN + explanation
# ---------------------------------------------------------------------------
def test_basque_v22kg3_requires_exactly_one_counter_narrative_sentence():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    assert "exactly 1 short, simple, complete sentence" in prompt
    assert "exactly 2 short" not in prompt
    assert "counter_narrative MUST be exactly 1 short" in prompt


def test_basque_v22kg3_requires_exactly_one_explanation_sentence():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    assert "explanation must be exactly 1 short Basque sentence, method-level only" in prompt


# ---------------------------------------------------------------------------
# 10-12. Basque avoid-list / style anchors
# ---------------------------------------------------------------------------
def test_basque_avoid_list_contains_required_phrases():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    for phrase in ["Pertsoak", "Pertsonak guztiak", "sexuen independentziak",
                   "horrek ez dio ezer okerrari dagokio"]:
        assert phrase in prompt, f"avoid-list must contain {phrase!r}"


def test_basque_style_anchors_are_claim_type_labelled_not_sample_id_labelled():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    for label in ["identity mockery / hiding identity", "exclusion based on sexual orientation",
                  "child/family rejection based on sexual orientation", "conversion/change framing",
                  "same-sex relationship stigma", "slur/name-calling", "generic dignity/dehumanization"]:
        assert label in prompt
    for sample_id in _SAMPLE_IDS:
        assert sample_id not in prompt


def test_basque_same_sex_relationship_anchor_uses_sexu_bereko_harremanak():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    assert "Sexu bereko harremanak" in prompt


# ---------------------------------------------------------------------------
# 13-14. Validator language-awareness fix ("dio")
# ---------------------------------------------------------------------------
def test_basque_dio_is_not_flagged_as_theology():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v3(kg, plan, counter_narrative="Berak dio hori garrantzitsua dela.",
                                         explanation="")
    theology_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_theological_claim"]
    assert theology_hits == []


def test_italian_dio_can_still_be_flagged_as_theology_when_unsupported():
    kg = build_counter_narrative_kg("comment", "it", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v3(kg, plan, counter_narrative="Dio ama tutti secondo la tradizione.",
                                         explanation="")
    theology_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_theological_claim"]
    assert theology_hits


# ---------------------------------------------------------------------------
# 15. Basque explanation avoids zientzia/ikerketa/froga unless supported
# ---------------------------------------------------------------------------
def test_basque_explanation_avoid_words_named_in_prompt():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg3")
    for word in ["zientzia", "ikerketa", "froga", "legea", "kultura", "herrialdea", "erlijioa", "Jainkoa"]:
        assert word in prompt


def test_basque_negated_science_mention_not_flagged():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v3(
        kg, plan, counter_narrative="",
        explanation="Zientzia edo ikerketa zehatzik gabe, erantzuna zuzena da.",
    )
    science_hits = [v for v in result["explanation_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert science_hits == []


# ---------------------------------------------------------------------------
# 16-19. kg_consistency checks all 3 fields + risk level rules
# ---------------------------------------------------------------------------
def test_v3_validator_checks_final_response_plan_counter_narrative_and_explanation():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "science shows this", "cultural_guidance": []}
    result = validate_kg_consistency_v3(kg, plan, counter_narrative="text", explanation="text")
    assert set(result.keys()) == {
        "final_response_plan_violations", "counter_narrative_violations",
        "explanation_violations", "forbidden_claims_triggered", "risk_level",
    }


def test_final_response_plan_only_violation_gives_low_risk_if_output_avoids_it():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "according to science, this is true", "cultural_guidance": []}
    result = validate_kg_consistency_v3(kg, plan, counter_narrative="Pertsona guztiek duintasuna merezi dute.",
                                         explanation="Erantzunak duintasuna azpimarratzen du.")
    assert result["final_response_plan_violations"]
    assert result["counter_narrative_violations"] == []
    assert result["explanation_violations"] == []
    assert result["risk_level"] == "low"


def test_explanation_violation_gives_medium_risk():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "clean plan", "cultural_guidance": []}
    result = validate_kg_consistency_v3(kg, plan, counter_narrative="Pertsona guztiek duintasuna merezi dute.",
                                         explanation="Erantzuna zientziaren arabera idatzi da.")
    assert result["explanation_violations"]
    assert result["counter_narrative_violations"] == []
    assert result["risk_level"] == "medium"


def test_counter_narrative_violation_gives_high_risk():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[
        {"source_id": "1", "passage": "supported by science and psychology.", "source_type": "cultural"},
    ])
    plan = {"final_response_plan": "clean plan", "cultural_guidance": []}
    result = validate_kg_consistency_v3(
        kg, plan, counter_narrative="Zientziaren eta psikologiaren arabera onartuta dago.",
        explanation="Erantzunak duintasuna azpimarratzen du.",
    )
    assert result["counter_narrative_violations"]
    assert result["risk_level"] == "high"


# ---------------------------------------------------------------------------
# 20. Evidence-conditional logic from v22kg2 remains intact
# ---------------------------------------------------------------------------
def test_evidence_conditional_logic_intact_science_supported_not_flagged():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[
        {"source_id": "1", "passage": "Scientific evidence shows X.", "source_type": "fact"},
    ])
    assert has_supported_claim_family(kg, "science_research") is True
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v3(kg, plan, counter_narrative="zientziaren arabera onartuta dago.",
                                         explanation="")
    assert result["counter_narrative_violations"] == []


# ---------------------------------------------------------------------------
# 21. Mistral tokenizer fix remains unchanged
# ---------------------------------------------------------------------------
def test_mistral_tokenizer_fix_unchanged_by_v22kg3_work():
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
            return {"fidelity": "on_target", "reason": "stub - not under test here"}
        return {"counter_narrative": "Being gay is a natural part of human diversity, observed across cultures and history.",
                "explanation": "Used a myth-correction strategy grounded in biology."}


def test_final_cn_agent_v22kg3_return_dict_includes_kg_and_v3_consistency():
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v22kg3"
        plan = JudgePlan(selected_language="eu", core_claim_to_counter="being gay is unnatural",
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
        print("\ntest_counter_narrative_kg3.py: ALL PASSED")
