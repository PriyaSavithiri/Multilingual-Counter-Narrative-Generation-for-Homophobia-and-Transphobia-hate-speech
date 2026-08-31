"""
Covers v22b (prompts.py section 7): the experimental final counter-narrative
prompt aimed at Mistral full_pipeline quality, on top of the frozen v22a
evaluation-cleanup checkpoint. Scope is deliberately narrow - ONLY
prompts.build_final_cn_prompt()'s text content changes; these tests check
the prompt TEXT asks for the right things, not live model output (that
needs a real model - see docs/07_EVALUATION_STRATEGY.md's "v22b" section
for the Colab qualitative-run instructions).

Also confirms style="v20" (and the FINAL_CN_STYLE default) is byte-for-byte
IDENTICAL to the original v20 prompt - the whole point of the module-level
toggle (see prompts.py's section-7 docstring) is that nothing downstream
changes unless v22b is explicitly selected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts

_SAMPLE_JUDGE_PLAN = {
    "selected_language": "en", "core_claim_to_counter": "gay people are sick",
    "recommended_strategy": ["factual_correction"],
    "approved_evidence": [{"source_id": "s1", "title": "t", "passage": "Being gay is not an illness.",
                            "retrieval_score": 0.9, "source_type": "fact", "language": "en"}],
    "rejected_content": [], "cultural_guidance": [], "safety_guidance": [],
    "final_response_plan": "Cite the approved evidence that this is not a disease.",
}


def _v22b(language="en", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b")


def _v20(language="en", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v20")


# ---------------------------------------------------------------------------
# v20 unchanged / style-selection contract
# ---------------------------------------------------------------------------
def test_default_style_is_v20_unchanged():
    assert prompts.FINAL_CN_STYLE == "v20"
    default_call = prompts.build_final_cn_prompt("test comment", "en", _SAMPLE_JUDGE_PLAN, "unknown")
    assert default_call == _v20()


def test_v22b_is_selectable_and_differs_from_v20():
    assert _v22b() != _v20()


# ---------------------------------------------------------------------------
# 1. 2-3 complete sentences
# ---------------------------------------------------------------------------
def test_prompt_asks_for_two_to_three_sentences():
    assert "2 to 3 complete sentences" in _v22b()


# ---------------------------------------------------------------------------
# 2. empathy/dignity + direct rebuttal + respectful closing structure
# ---------------------------------------------------------------------------
def test_prompt_asks_for_dignity_empathy_opening():
    text = _v22b()
    assert "dignity" in text.lower()
    assert "empathy" in text.lower()


def test_prompt_asks_for_direct_rebuttal_of_specific_claim():
    text = _v22b()
    assert "directly rebut" in text.lower()
    assert "core_claim_to_counter" in text


def test_prompt_asks_for_respectful_closing():
    text = _v22b().lower()
    assert any(word in text for word in ("coexistence", "inclusion", "safer dialogue", "equality"))


# ---------------------------------------------------------------------------
# 3. use Judge-approved reasoning/evidence
# ---------------------------------------------------------------------------
def test_prompt_asks_to_use_judge_approved_evidence():
    text = _v22b()
    assert "approved_evidence" in text
    assert "final_response_plan" in text
    assert "recommended_strategy" in text


# ---------------------------------------------------------------------------
# 4. all thesis target languages representable (eu/es/it/ta directly,
#    en_EUR/en_IN via language="en" + region resolution)
# ---------------------------------------------------------------------------
def test_prompt_covers_basque_spanish_italian_tamil_directly():
    for language in ("eu", "es", "it", "ta"):
        text = _v22b(language=language)
        assert f"({language})" in text, f"language code {language!r} must appear in the rendered prompt"


def test_prompt_resolves_en_eur_region_code():
    text = _v22b(language="en", region_context="European")
    assert "en_EUR" in text


def test_prompt_resolves_en_in_region_code():
    text = _v22b(language="en", region_context="Indian")
    assert "en_IN" in text


def test_prompt_falls_back_to_plain_en_when_region_unknown():
    text = _v22b(language="en", region_context="unknown")
    assert "en_EUR" not in text
    assert "en_IN" not in text


# ---------------------------------------------------------------------------
# 5. simple-language rules for Basque and Tamil specifically
# ---------------------------------------------------------------------------
def test_prompt_adds_simple_wording_rule_for_basque_and_tamil():
    for language in ("eu", "ta"):
        text = _v22b(language=language).lower()
        assert "simple" in text and "clear" in text


def test_prompt_does_not_add_simple_wording_rule_for_spanish_italian():
    for language in ("es", "it"):
        text = _v22b(language=language)
        assert "greater" not in text.lower() or "risk" not in text.lower()
        assert "literal" in text.lower()  # gets the "avoid over-literal translation" rule instead


# ---------------------------------------------------------------------------
# 6. discourage invented/listed LGBTQ+ identity terms
# ---------------------------------------------------------------------------
def test_prompt_discourages_inventing_identity_terms():
    text = _v22b().lower()
    assert "invent" in text
    assert "transliterate" in text
    assert "lgbtq+ people" in text or "diverse sexual orientations" in text


def test_prompt_discourages_listing_multiple_identity_groups():
    text = _v22b().lower()
    assert "list" in text and "identity group" in text


# ---------------------------------------------------------------------------
# 7. discourage poetic/metaphorical openings
# ---------------------------------------------------------------------------
def test_prompt_discourages_poetic_or_metaphorical_openings():
    text = _v22b().lower()
    assert "poetic" in text
    assert "metaphorical" in text
    assert "clarity over style" in text


# ---------------------------------------------------------------------------
# 8. avoid unsupported cultural claims
# ---------------------------------------------------------------------------
def test_prompt_avoids_unsupported_cultural_claims():
    text = _v22b().lower()
    assert "our culture values respect" in text  # the exact bad-example phrase, quoted as what NOT to do
    assert "overgeneralize" in text


# ---------------------------------------------------------------------------
# 9. does not ask the model to copy the reference CN
# ---------------------------------------------------------------------------
def test_prompt_forbids_copying_reference_cn():
    text = _v22b().lower()
    assert "do not copy" in text
    assert "reference" in text


if __name__ == "__main__":
    test_default_style_is_v20_unchanged()
    test_v22b_is_selectable_and_differs_from_v20()
    test_prompt_asks_for_two_to_three_sentences()
    test_prompt_asks_for_dignity_empathy_opening()
    test_prompt_asks_for_direct_rebuttal_of_specific_claim()
    test_prompt_asks_for_respectful_closing()
    test_prompt_asks_to_use_judge_approved_evidence()
    test_prompt_covers_basque_spanish_italian_tamil_directly()
    test_prompt_resolves_en_eur_region_code()
    test_prompt_resolves_en_in_region_code()
    test_prompt_falls_back_to_plain_en_when_region_unknown()
    test_prompt_adds_simple_wording_rule_for_basque_and_tamil()
    test_prompt_does_not_add_simple_wording_rule_for_spanish_italian()
    test_prompt_discourages_inventing_identity_terms()
    test_prompt_discourages_listing_multiple_identity_groups()
    test_prompt_discourages_poetic_or_metaphorical_openings()
    test_prompt_avoids_unsupported_cultural_claims()
    test_prompt_forbids_copying_reference_cn()
    print("test_final_cn_prompt_v22b.py: ALL PASSED")
