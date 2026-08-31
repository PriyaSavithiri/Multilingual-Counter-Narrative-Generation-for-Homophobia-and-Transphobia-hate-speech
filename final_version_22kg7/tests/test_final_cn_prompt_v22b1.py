"""
Covers v22b.1 (prompts.py section 7): a targeted tightening of v22b's final
counter-narrative prompt, added after trace-verified evidence-control
problems were found in v22b's 5-sample Colab run (IT125 overgeneralization,
EN_IN-163 fabrication - see docs/07_EVALUATION_STRATEGY.md's "v22b.1"
section and prompts.py's own comment above _build_final_cn_prompt_v22b1).

v22b1 does NOT replace v22b - both remain independently selectable via
style="v22b"/"v22b1" (or --final-cn-style on the CLI) so a 3-way v20/v22b/
v22b1 comparison stays possible. These tests check the prompt TEXT asks for
the right things; live model output needs a real model (see the Colab
instructions in the same docs section).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts
from main import _tag_filename_for_final_cn_style

_SAMPLE_JUDGE_PLAN = {
    "selected_language": "en", "core_claim_to_counter": "gay people are sick",
    "recommended_strategy": ["factual_correction"],
    "approved_evidence": [{"source_id": "s1", "title": "t", "passage": "Being gay is not an illness.",
                            "retrieval_score": 0.9, "source_type": "fact", "language": "en"}],
    "rejected_content": [], "cultural_guidance": [], "safety_guidance": [],
    "final_response_plan": "Cite the approved evidence that this is not a disease.",
}


def _v22b1(language="en", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b1")


def _v22b(language="en", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b")


def _v20(language="en", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v20")


def _normalized(text: str) -> str:
    """Collapses all whitespace runs (including line-wrap newlines +
    indentation inside the prompt's f-string) into single spaces, so
    substring checks aren't accidentally broken by where the prompt text
    happens to wrap - only the actual wording matters here, not layout."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 1-4: style selection / availability / filename tagging
# ---------------------------------------------------------------------------
def test_v20_style_unchanged_after_adding_v22b1():
    assert prompts.FINAL_CN_STYLE == "v20"
    default_call = prompts.build_final_cn_prompt("test comment", "en", _SAMPLE_JUDGE_PLAN, "unknown")
    assert default_call == _v20()


def test_v22b_still_available_and_unchanged_alongside_v22b1():
    text = _v22b()
    # Pinned substring from v22b's original structure instruction - if this
    # ever fails, v22b's own body was accidentally edited while adding v22b1.
    assert "using the Judge's approved_evidence/final_response_plan/" in text
    assert "STRICT EVIDENCE-CONTROL RULES" not in text  # that's v22b1-only


def test_v22b1_is_selectable_and_distinct_from_v20_and_v22b():
    v20, v22b, v22b1 = _v20(), _v22b(), _v22b1()
    assert len({v20, v22b, v22b1}) == 3


def test_filename_tagging_handles_v22b1():
    assert _tag_filename_for_final_cn_style("it", "v20") == "it"
    assert _tag_filename_for_final_cn_style("it", "v22b") == "it-finalcnv22b"
    assert _tag_filename_for_final_cn_style("it", "v22b1") == "it-finalcnv22b1"
    # all three must be mutually distinct - no resume/output collision
    names = {_tag_filename_for_final_cn_style("it", s) for s in ("v20", "v22b", "v22b1")}
    assert len(names) == 3


# ---------------------------------------------------------------------------
# 5-6: citation-sounding phrase gating
# ---------------------------------------------------------------------------
def test_prompt_gates_studies_and_research_show_phrases():
    text = _v22b1().lower()
    assert "studies show" in text
    assert "research shows" in text
    assert "unless" in text  # gated, not a flat ban with no exception clause


def test_prompt_gates_science_evidence_data_statistics_phrases():
    text = _v22b1().lower()
    for phrase in ("science shows", "evidence proves", "data shows", "statistics show", "experts say"):
        assert phrase in text, f"missing gated phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# 7-8: legal/country-specific claim gating + no-narrowing rule
# ---------------------------------------------------------------------------
def test_prompt_gates_legal_and_country_specific_claims():
    text = _normalized(_v22b1()).lower()
    assert "laws guarantee" in text
    assert "protected by law" in text
    assert "in italy" in text and "in europe" in text and "in india" in text
    assert "explicitly names that exact country/region" in text


def test_prompt_forbids_narrowing_many_countries_into_named_region():
    text = _normalized(_v22b1()).lower()
    assert '"many countries"' in text
    assert "do not rewrite that as" in text or "not rewrite that as" in text
    assert "narrowing or strengthening" in text


# ---------------------------------------------------------------------------
# 9: cultural_guidance is tone-only, never factual evidence
# ---------------------------------------------------------------------------
def test_prompt_limits_cultural_guidance_to_tone_not_facts():
    text = _v22b1().lower()
    assert "cultural_guidance" in text
    assert "tone" in text
    assert "never" in text and "factual evidence" in text
    assert "approved_evidence, never from cultural_guidance" in text.replace("\n", " ")


# ---------------------------------------------------------------------------
# 10: weak/general evidence fallback to plain dignity rebuttal
# ---------------------------------------------------------------------------
def test_prompt_includes_weak_evidence_dignity_fallback():
    text = _normalized(_v22b1()).lower()
    assert "weak" in text and "general" in text
    assert "a person's identity is not a reason to reject or harm them" in text


# ---------------------------------------------------------------------------
# 11: Italian-specific tightening
# ---------------------------------------------------------------------------
def test_italian_prompt_forbids_unsupported_legal_demographic_claims():
    text = _v22b1(language="it")
    assert "In Italia..." in text
    assert "In Europa..." in text
    assert "le leggi garantiscono..." in text
    assert "maximum 3 sentences" in text


# ---------------------------------------------------------------------------
# 12-14: Basque/Tamil sentence-count and structure tightening
# ---------------------------------------------------------------------------
def test_basque_prompt_asks_for_exactly_two_short_sentences():
    text = _v22b1(language="eu")
    assert "exactly 2 short, simple, complete sentences" in text
    assert "exactly 2 short sentences wherever possible" in text


def test_tamil_prompt_asks_for_exactly_two_short_sentences():
    text = _v22b1(language="ta")
    assert "exactly 2 short, simple, complete sentences" in text
    assert "use exactly 2 short sentences" in text


def test_tamil_prompt_bans_rhetorical_questions_and_long_clauses():
    text = _v22b1(language="ta").lower()
    assert "rhetorical question" in text
    assert "long multi-clause sentences" in text


# ---------------------------------------------------------------------------
# 15-16: identity-term invention + poetic openings (carried over from v22b,
# must still be present in v22b1)
# ---------------------------------------------------------------------------
def test_prompt_still_discourages_inventing_identity_terms():
    text = _v22b1().lower()
    assert "invent" in text
    assert "transliterate" in text
    assert "lgbtq+ people" in text or "diverse sexual orientations" in text


def test_prompt_still_discourages_poetic_or_metaphorical_openings():
    text = _v22b1().lower()
    assert "poetic" in text
    assert "metaphorical" in text
    assert "clarity over style" in text


if __name__ == "__main__":
    test_v20_style_unchanged_after_adding_v22b1()
    test_v22b_still_available_and_unchanged_alongside_v22b1()
    test_v22b1_is_selectable_and_distinct_from_v20_and_v22b()
    test_filename_tagging_handles_v22b1()
    test_prompt_gates_studies_and_research_show_phrases()
    test_prompt_gates_science_evidence_data_statistics_phrases()
    test_prompt_gates_legal_and_country_specific_claims()
    test_prompt_forbids_narrowing_many_countries_into_named_region()
    test_prompt_limits_cultural_guidance_to_tone_not_facts()
    test_prompt_includes_weak_evidence_dignity_fallback()
    test_italian_prompt_forbids_unsupported_legal_demographic_claims()
    test_basque_prompt_asks_for_exactly_two_short_sentences()
    test_tamil_prompt_asks_for_exactly_two_short_sentences()
    test_tamil_prompt_bans_rhetorical_questions_and_long_clauses()
    test_prompt_still_discourages_inventing_identity_terms()
    test_prompt_still_discourages_poetic_or_metaphorical_openings()
    print("test_final_cn_prompt_v22b1.py: ALL PASSED")
