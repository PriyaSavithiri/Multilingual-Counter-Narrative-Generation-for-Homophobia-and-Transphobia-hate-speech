"""
Coverage for v22kg4: a tiny, prompt-only Basque fluency fallback on top of
v22kg3 (unchanged). No KG architecture, validator, RAG, Judge, Defender,
tokenizer, model, or evaluation changes - v22kg4 reuses v22kg3's own
validate_kg_consistency_v3() unchanged (see final_cn_agent.py).

Motivated by: Basque v22kg3 passed the structural KG gates (1-sentence
length, no unsupported-claim leaks, the "dio" false positive fixed) but real
review still found awkward/complex Basque wording. v22kg4 adds a small set
of claim-type FALLBACK style anchors the model can lean on when unsure,
explicitly framed as non-mandatory fallbacks (not fixed templates, not
sample-specific), plus an expanded avoid-list for the specific awkward
constructions found in review.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import prompts
from counter_narrative_kg import validate_kg_consistency_v3
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
# v22kg4 exists/selectable
# ---------------------------------------------------------------------------
def test_v22kg4_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg4"])
    assert args.final_cn_style == "v22kg4"


# ---------------------------------------------------------------------------
# No sample IDs in runtime prompt
# ---------------------------------------------------------------------------
def test_v22kg4_prompt_contains_no_sample_ids():
    plan = _minimal_judge_plan_dict()
    for language in ["eu", "ta", "it", "en"]:
        prompt = prompts.build_final_cn_prompt("comment", language, plan, style="v22kg4")
        for sample_id in _SAMPLE_IDS:
            assert sample_id not in prompt, f"v22kg4 prompt ({language}) must not contain sample ID {sample_id}"
        assert "IT125-style" not in prompt
        assert "IT133-style" not in prompt


def test_v22kg4_fallback_anchors_are_claim_type_labelled_not_sample_specific():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg4")
    assert "not mandatory fixed" in prompt
    assert "not tied to any specific sample" in prompt


# ---------------------------------------------------------------------------
# Avoid-list includes the failed phrases from v22kg3 review
# ---------------------------------------------------------------------------
def test_avoid_list_includes_phrases_found_in_v22kg3_review():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg4")
    for phrase in [
        "Gizon-emakumezkoak diren edo ez",
        "orientazio sexualak",
        "merezi dituzte",
        "naturaleko maitasuna",
        "iritziki gaizkileak",
        "harremanen arteko aniztasuna",
    ]:
        assert phrase in prompt, f"avoid-list must mention {phrase!r}"
    # carried forward from v22kg3's own avoid-list too
    for phrase in ["Pertsoak", "Pertsonak guztiak", "dignitatea"]:
        assert phrase in prompt


# ---------------------------------------------------------------------------
# Basque explanation fallback is present
# ---------------------------------------------------------------------------
def test_basque_explanation_fallback_present():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg4")
    assert "Erantzunak iruzkin kaltegarria zuzenean ukatu eta duintasuna eta errespetua azpimarratzen ditu." in prompt


def test_basque_fallback_claim_type_anchors_present():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg4")
    assert "FALLBACK (WHEN UNSURE)" in prompt
    for text in [
        "Inor ez da bere sexu-orientazioagatik baztertu behar",
        "Haurraren sexu-orientazioa ez da baztertzeko arrazoia",
        "Norberaren sexu-orientazioa ez da konpondu beharreko arazoa",
        "Sexu bereko harremanak ez dira lotsagarriak edo kaltegarriak",
        "Pertsona guztiek duintasuna eta errespetua merezi dute, haien sexu-orientazioa edozein",
    ]:
        assert text in prompt


def test_basque_one_sentence_rule_still_intact():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg4")
    assert "exactly 1 short, simple, complete sentence" in prompt
    assert "exactly 2 short" not in prompt


# ---------------------------------------------------------------------------
# Old styles unchanged
# ---------------------------------------------------------------------------
def test_v22kg3_remains_unchanged_by_v22kg4_addition():
    plan = _minimal_judge_plan_dict()
    prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style="v22kg3")
    assert "FALLBACK (WHEN UNSURE)" not in prompt
    assert "Gizon-emakumezkoak diren edo ez" not in prompt


def test_v22kg2_v22kg1_remain_unchanged_by_v22kg4_addition():
    plan = _minimal_judge_plan_dict()
    p2 = prompts.build_final_cn_prompt("comment", "eu", plan, style="v22kg2")
    assert "IT125-style case" in p2
    assert "FALLBACK (WHEN UNSURE)" not in p2
    plan_ta = dict(plan, selected_language="ta")
    p1 = prompts.build_final_cn_prompt("comment", "ta", plan_ta, style="v22kg1")
    assert "IT125-style case" in p1


def test_old_styles_v20_through_v22b6_unchanged_by_v22kg4_addition():
    plan = _minimal_judge_plan_dict()
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style=style)
        assert "KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg4"


# ---------------------------------------------------------------------------
# Validator unchanged - v22kg4 reuses validate_kg_consistency_v3 exactly (no new validator)
# ---------------------------------------------------------------------------
def test_final_cn_agent_v22kg4_uses_v3_validator_unchanged():
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
        prompts.FINAL_CN_STYLE = "v22kg4"
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


def test_mistral_tokenizer_fix_unchanged_by_v22kg4_work():
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
        print("\ntest_counter_narrative_kg4.py: ALL PASSED")
