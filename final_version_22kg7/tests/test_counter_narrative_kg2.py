"""
Coverage for v22kg2: language-independent evidence-conditional KG consistency
+ Basque fluency fix, built on top of v22kg1 (v22kg1 itself must stay
completely unchanged - see test_counter_narrative_kg1.py for v22kg1's own
coverage, still passing untouched).

Core correction from v22kg1: science/legal/religious/country/cultural TERMS
must never be blindly banned - a term is only a problem when the
corresponding claim is UNSUPPORTED by approved factual/web evidence. Directly
motivated by two real leaks found in the v22kg1 Basque rerun (5 rows,
reviewed this session): EU125's counter_narrative stated "zientziaren eta
psikologiaren arabera" (according to science and psychology) from evidence
that was all source_type "cultural" - v22kg1's Tamil/English-only trigger
list had no way to catch Basque wording. EU130 stated as flat fact "Ez da
posible sexualitatea aldatzea" (it's not possible to change sexual
orientation) - an immutability/conversion claim not covered by any of the
4 default KG-builder categories at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import prompts
from counter_narrative_kg import build_counter_narrative_kg, validate_kg_consistency_v2, has_supported_claim_family
from model_api import ModelClient
from schemas import JudgePlan
from final_cn_agent import FinalCNAgent

# Real trace text from the v22kg1 Basque rerun (EU125), reviewed this session.
_EU125_CULTURAL_ONLY_EVIDENCE = [
    {"source_id": "1178", "passage": (
        "homosexuality is not wrong. it's a natural expression of human diversity and is "
        "supported by science and psychology."
     ), "source_type": "cultural"},
    {"source_id": "144", "passage": "Scientific evidence shows that LGBTQ+ identities are normal variations, not defects.",
     "source_type": "cultural"},
]
_EU125_FACT_SUPPORTED_EVIDENCE = [
    {"source_id": "1178", "passage": "Scientific evidence shows homosexuality is a natural variation, not a defect.",
     "source_type": "fact"},
]
_EU125_LEAKED_CN = (
    "Homoseksualitatea ez da gaizki edo madarikazio bat; gizakiaren aniztasun natural baten "
    "adierazpen bat da, eta zientziaren eta psikologiaren arabera onartuta dago."
)
_EU130_LEAKED_CN = (
    "Guztiok merezitako duintasuna eta errespetu bera dute. Ez da posible sexualitatea "
    "aldatzea, eta hori saiatu nahi izatea ere kaltegarria izango litzateke."
)


def _minimal_judge_plan_dict(final_response_plan="plan", approved_evidence=None, cultural_guidance=None):
    return {
        "selected_language": "eu", "core_claim_to_counter": "claim", "recommended_strategy": [],
        "approved_evidence": approved_evidence or [], "rejected_content": [],
        "cultural_guidance": cultural_guidance or [], "safety_guidance": [],
        "final_response_plan": final_response_plan,
    }


# ---------------------------------------------------------------------------
# 1-3. Style existence / isolation
# ---------------------------------------------------------------------------
def test_v22kg2_is_a_valid_cli_choice():
    args = main.build_arg_parser().parse_args(["generate", "--text", "x", "--final-cn-style", "v22kg2"])
    assert args.final_cn_style == "v22kg2"


def test_v22kg1_remains_unchanged_by_v22kg2_addition():
    plan = _minimal_judge_plan_dict()
    prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style="v22kg1")
    assert "EVIDENCE-CONDITIONAL RULES" not in prompt
    assert "ARE NOT BANNED" not in prompt
    assert "dignitatea" not in prompt, "v22kg1 must not gain v22kg2's new Basque fluency note"


def test_old_styles_remain_unchanged_by_v22kg2_addition():
    # v20-v22b6 must never contain KG text at all; v22kg legitimately has its own KG block
    # (that's expected, unrelated to v22kg2) - checked separately by
    # test_v22kg1_remains_unchanged_by_v22kg2_addition and the v22kg1-vs-v22kg2 header check below.
    plan = _minimal_judge_plan_dict()
    for style in ["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6"]:
        prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style=style)
        assert "KNOWLEDGE GRAPH" not in prompt, f"{style} must be unaffected by v22kg2"

    v22kg_prompt = prompts.build_final_cn_prompt("comment", "eu", plan, style="v22kg")
    assert "PRIMARY GROUNDING CONTRACT" not in v22kg_prompt, "v22kg must not gain v22kg2's stronger header"
    assert "ARE NOT BANNED" not in v22kg_prompt


# ---------------------------------------------------------------------------
# 4-6. v22kg2 prompt content - evidence-conditional framing
# ---------------------------------------------------------------------------
def test_v22kg2_prompt_contains_evidence_conditional_claim_control_language():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(final_response_plan="plan",
                                            approved_evidence=[]), style="v22kg2")
    assert "EVIDENCE-CONDITIONAL RULES" in prompt
    assert "ARE NOT BANNED" in prompt
    assert "ALLOWED whenever" in prompt


def test_v22kg2_prompt_says_claims_allowed_only_with_approved_factual_web_evidence():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg2")
    assert "APPROVED FACTUAL/WEB EVIDENCE nodes are the ONLY basis for factual, legal, scientific" in prompt


def test_v22kg2_prompt_says_cultural_context_is_tone_framing_only():
    prompt = prompts.build_final_cn_prompt("comment", "en", _minimal_judge_plan_dict(), style="v22kg2")
    assert "APPROVED CULTURAL CONTEXT nodes are tone/framing only" in prompt


# ---------------------------------------------------------------------------
# 7-9. Basque evidence-conditional validator behavior (real EU125/EU130 cases)
# ---------------------------------------------------------------------------
def test_basque_science_phrase_triggers_unsupported_science_claim_when_unsupported():
    kg = build_counter_narrative_kg("curse comment", "eu", approved_evidence=_EU125_CULTURAL_ONLY_EVIDENCE)
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative=_EU125_LEAKED_CN, explanation="")
    science_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert science_hits
    assert result["risk_level"] == "high"


def test_basque_science_phrase_not_flagged_when_fact_evidence_supports_it():
    kg = build_counter_narrative_kg("curse comment", "eu", approved_evidence=_EU125_FACT_SUPPORTED_EVIDENCE)
    assert has_supported_claim_family(kg, "science_research") is True
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative=_EU125_LEAKED_CN, explanation="")
    science_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_science_claim"]
    assert science_hits == [], "must NOT flag science wording when fact/web evidence explicitly supports it"


def test_basque_immutability_claim_flagged_unless_factually_supported():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[
        {"source_id": "152", "passage": "Sexual orientation and gender identity are natural variations, not errors.",
         "source_type": "cultural"},
    ])
    assert has_supported_claim_family(kg, "orientation_immutability") is False
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative=_EU130_LEAKED_CN, explanation="")
    immutability_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_immutability_claim"]
    assert immutability_hits
    assert result["risk_level"] == "high"


def test_basque_immutability_claim_not_flagged_when_factually_supported():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[
        {"source_id": "1", "passage": "Sexual orientation is innate and cannot be changed.", "source_type": "fact"},
    ])
    assert has_supported_claim_family(kg, "orientation_immutability") is True
    plan = {"final_response_plan": "", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative=_EU130_LEAKED_CN, explanation="")
    immutability_hits = [v for v in result["counter_narrative_violations"] if v["forbidden_claim"] == "unsupported_immutability_claim"]
    assert immutability_hits == []


# ---------------------------------------------------------------------------
# 10-12. Basque fluency prompt instructions
# ---------------------------------------------------------------------------
def test_v22kg2_basque_prompt_avoids_dignitatea_prefers_duintasuna():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg2")
    assert '"dignitatea"' in prompt  # named in the AVOID list
    assert "duintasuna" in prompt    # named in the PREFER list


def test_v22kg2_basque_prompt_avoids_berdintasunik_gabeko_arreta():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg2")
    assert "berdintasunik gabeko arreta" in prompt
    idx = prompt.index("berdintasunik gabeko arreta")
    assert "AVOID" in prompt[:idx][-400:]


def test_v22kg2_basque_prompt_avoids_harreman_homofobikoak():
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg2")
    assert "harreman homofobikoak" in prompt
    idx = prompt.index("harreman homofobikoak")
    assert "AVOID" in prompt[:idx][-400:]


def test_v22kg2_basque_prompt_keeps_two_sentence_structure_not_one():
    """v22kg2 must NOT force Tamil's 1-sentence rule onto Basque."""
    prompt = prompts.build_final_cn_prompt("comment", "eu", _minimal_judge_plan_dict(), style="v22kg2")
    assert "exactly 2 short sentences wherever possible" in prompt
    assert "exactly 1 short, simple, complete sentence" not in prompt


# ---------------------------------------------------------------------------
# 13-16. kg_consistency checks all 3 fields + risk level rules
# ---------------------------------------------------------------------------
def test_v2_validator_checks_final_response_plan_counter_narrative_and_explanation():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "science shows this", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative="text", explanation="text")
    assert set(result.keys()) == {
        "final_response_plan_violations", "counter_narrative_violations",
        "explanation_violations", "forbidden_claims_triggered", "risk_level",
    }


def test_final_response_plan_only_violation_gives_low_risk_if_output_avoids_it():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "according to science, this is true", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative="Pertsona guztiek duintasuna merezi dute.",
                                         explanation="Erantzunak duintasuna azpimarratzen du.")
    assert result["final_response_plan_violations"]
    assert result["counter_narrative_violations"] == []
    assert result["explanation_violations"] == []
    assert result["risk_level"] == "low"


def test_explanation_violation_gives_medium_risk():
    kg = build_counter_narrative_kg("comment", "eu", approved_evidence=[])
    plan = {"final_response_plan": "clean plan", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative="Pertsona guztiek duintasuna merezi dute.",
                                         explanation="Erantzuna zientziaren arabera idatzi da.")
    assert result["explanation_violations"]
    assert result["counter_narrative_violations"] == []
    assert result["risk_level"] == "medium"


def test_counter_narrative_violation_gives_high_risk():
    kg = build_counter_narrative_kg("curse comment", "eu", approved_evidence=_EU125_CULTURAL_ONLY_EVIDENCE)
    plan = {"final_response_plan": "clean plan", "cultural_guidance": []}
    result = validate_kg_consistency_v2(kg, plan, counter_narrative=_EU125_LEAKED_CN,
                                         explanation="Erantzunak duintasuna azpimarratzen du.")
    assert result["counter_narrative_violations"]
    assert result["risk_level"] == "high"


# ---------------------------------------------------------------------------
# 17. Mistral tokenizer fix remains unchanged
# ---------------------------------------------------------------------------
def test_mistral_tokenizer_fix_unchanged_by_v22kg2_work():
    from model_api import _hf_transformers_tokenizer_kwargs, _hf_transformers_model_auth_kwargs
    tok_kwargs = _hf_transformers_tokenizer_kwargs("gghfez/Mistral-Small-3.2-24B-Instruct-hf", hf_token="")
    assert tok_kwargs.get("fix_mistral_regex") is True
    model_kwargs = _hf_transformers_model_auth_kwargs(hf_token="tok123")
    assert "fix_mistral_regex" not in model_kwargs


# ---------------------------------------------------------------------------
# Trace wiring sanity check (FinalCNAgent-level, mirroring v22kg1's own test)
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


def test_final_cn_agent_v22kg2_return_dict_includes_kg_and_v2_consistency():
    original_style = prompts.FINAL_CN_STYLE
    try:
        prompts.FINAL_CN_STYLE = "v22kg2"
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
        print("\ntest_counter_narrative_kg2.py: ALL PASSED")
