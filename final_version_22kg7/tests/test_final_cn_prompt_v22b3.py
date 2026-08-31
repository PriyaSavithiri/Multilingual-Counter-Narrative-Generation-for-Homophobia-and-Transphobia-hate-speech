"""
Covers v22b.3 (prompts.py section 7): an explanation-safe, source-type-
strict hotfix on top of v22b2. Motivated by a real, trace-verified gap:
v22b2's Italian rerun produced a fully clean IT125 counter_narrative, but
its explanation field said "l'approccio culturale italiano valorizza la
dignita umana..." - traced to the Judge's own cultural_guidance field
("...valori centrali nella cultura italiana"), which approved_evidence
never supported. v22b2's rules governed counter_narrative primarily; the
explanation field only got one pointer sentence. v22b3 gives explanation
its own detailed rule set instead.

v22b3 does NOT replace v20/v22b/v22b1/v22b2 - all five remain independently
selectable. These tests check the prompt TEXT asks for the right things;
live model verification (the actual IT125 rerun) needs a real model - see
the Colab instructions in docs/07_EVALUATION_STRATEGY.md's "v22b.3" section.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts
from main import _tag_filename_for_final_cn_style

_SAMPLE_JUDGE_PLAN = {
    "selected_language": "it", "core_claim_to_counter": "gay people are sick",
    "recommended_strategy": ["factual_correction"],
    "approved_evidence": [{"source_id": "s1", "title": "t", "passage": "Same-sex attraction is natural.",
                            "retrieval_score": 0.9, "source_type": "cultural", "language": "it"}],
    "rejected_content": [], "cultural_guidance": ["Italian culture values family and dignity."],
    "safety_guidance": [], "final_response_plan": "Rebut using the approved evidence.",
}


def _v22b3(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b3")


def _v22b2(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b2")


def _v22b1(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b1")


def _v22b(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b")


def _v20(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v20")


def _normalized(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 1-3: style selection / availability / filename tagging
# ---------------------------------------------------------------------------
def test_v22b3_is_selectable():
    text = _v22b3()
    assert "EXPLANATION-FIELD RULES" in text


def test_v20_v22b_v22b1_v22b2_remain_available_and_unchanged():
    v20, v22b, v22b1, v22b2, v22b3 = _v20(), _v22b(), _v22b1(), _v22b2(), _v22b3()
    assert len({v20, v22b, v22b1, v22b2, v22b3}) == 5
    for older in (v20, v22b, v22b1, v22b2):
        assert "EXPLANATION-FIELD RULES" not in older
    assert "EXACT-EVIDENCE-SCOPE RULES" in v22b2  # v22b2's own body, still intact
    assert prompts.build_final_cn_prompt("test comment", "it", _SAMPLE_JUDGE_PLAN, "unknown") == v20


def test_filename_suffix_for_v22b3_is_distinct():
    names = {s: _tag_filename_for_final_cn_style("it", s) for s in ("v20", "v22b", "v22b1", "v22b2", "v22b3")}
    assert names["v20"] == "it"
    assert names["v22b3"] == "it-finalcnv22b3"
    assert len(set(names.values())) == 5


# ---------------------------------------------------------------------------
# 4: explanation must be method-level only
# ---------------------------------------------------------------------------
def test_explanation_must_be_method_level_only():
    text = _normalized(_v22b3()).lower()
    assert "method-level summary only, not a second" in text
    assert "may describe method only" in text


# ---------------------------------------------------------------------------
# 5-6: IT125-specific forbidden explanation phrases
# ---------------------------------------------------------------------------
def test_explanation_forbids_italian_cultural_approach_phrase():
    text = _normalized(_v22b3())
    assert "approccio culturale italiano" in text


def test_explanation_forbids_italian_culture_values_in_italy_phrases():
    text = _normalized(_v22b3())
    for phrase in ("cultura italiana", "valori italiani", "Italian approach", "Italian values",
                   "Italian culture", "in Italia", '"Italy"'):
        assert phrase in text, f"missing forbidden explanation phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# 7-8: explanation cannot copy unsupported claims from cultural_guidance/
# final_response_plan
# ---------------------------------------------------------------------------
def test_explanation_forbids_copying_unsupported_judge_guidance():
    text = _normalized(_v22b3()).lower()
    assert "cultural_guidance or final_response_plan contains a claim like" in text
    assert "ignore that specific part when writing the explanation" in text
    assert "do not repeat it, paraphrase it, or reference it" in text


# ---------------------------------------------------------------------------
# 9: IT133-style science/history gating
# ---------------------------------------------------------------------------
def test_explanation_forbids_science_and_history_unless_both_supported():
    text = _normalized(_v22b3())
    assert "scienza e storia" in text
    assert "science and history" in text
    assert "BOTH the scientific claim AND the historical claim" in text


# ---------------------------------------------------------------------------
# 10-11: source_type EXACT match requirement
# ---------------------------------------------------------------------------
def test_source_type_requires_exact_fact_or_web_match():
    text = _normalized(_v22b3())
    assert "EXACTLY the string \"fact\" or EXACTLY the string \"web\"" in text
    assert "an exact match, not an approximation" in text


def test_source_type_lists_culturale_and_missing_as_tone_only():
    text = _normalized(_v22b3())
    assert '"culturale"' in text
    assert "a missing value, an unrecognized value, or a malformed value" in text
    assert "must be treated as TONE/FRAMING ONLY" in text


# ---------------------------------------------------------------------------
# 12: counter_narrative exact-evidence rules still apply
# ---------------------------------------------------------------------------
def test_counter_narrative_rules_still_present():
    text = _v22b3()
    assert "COUNTRY/REGION-NAME RULE" in text
    assert "NO NARROWING" in text
    assert "HEDGING PRESERVATION" in text
    assert "WEAK/GENERAL EVIDENCE FALLBACK" in text


# ---------------------------------------------------------------------------
# 13: cultural_guidance limitation remains explicit
# ---------------------------------------------------------------------------
def test_cultural_guidance_limitation_still_explicit():
    text = _normalized(_v22b3()).lower()
    assert "cultural_guidance field is guidance about tone only" in text
    assert "never be treated as evidence" in text


# ---------------------------------------------------------------------------
# 14: language-specific rules preserved
# ---------------------------------------------------------------------------
def test_italian_tamil_basque_language_rules_preserved():
    it_text = _v22b3(language="it")
    assert "maximum 3 sentences" in it_text
    assert "tra cui l'Italia" in it_text

    ta_text = _v22b3(language="ta")
    assert "exactly 2 short, simple, complete sentences" in ta_text
    assert "everyday Tamil" in ta_text

    eu_text = _v22b3(language="eu")
    assert "exactly 2 short, simple, complete sentences" in eu_text
    assert "between men and women" in eu_text


if __name__ == "__main__":
    test_v22b3_is_selectable()
    test_v20_v22b_v22b1_v22b2_remain_available_and_unchanged()
    test_filename_suffix_for_v22b3_is_distinct()
    test_explanation_must_be_method_level_only()
    test_explanation_forbids_italian_cultural_approach_phrase()
    test_explanation_forbids_italian_culture_values_in_italy_phrases()
    test_explanation_forbids_copying_unsupported_judge_guidance()
    test_explanation_forbids_science_and_history_unless_both_supported()
    test_source_type_requires_exact_fact_or_web_match()
    test_source_type_lists_culturale_and_missing_as_tone_only()
    test_counter_narrative_rules_still_present()
    test_cultural_guidance_limitation_still_explicit()
    test_italian_tamil_basque_language_rules_preserved()
    print("test_final_cn_prompt_v22b3.py: ALL PASSED")
