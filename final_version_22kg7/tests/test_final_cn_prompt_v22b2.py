"""
Covers v22b.2 (prompts.py section 7): an exact-evidence-scope hotfix on top
of v22b1, added after a dual-RAG audit (see docs/07_EVALUATION_STRATEGY.md's
"v22b.2" section) found IT125's residual "including Italy" issue was
Final-CN-prompt-fixable - the per-item source_type ("fact"/"cultural"/"web")
was already present in every prior version's prompt, just never instructed
to be used. v22b2 makes the country/region-name rule mechanical (an
enumerated forbidden-phrase list) rather than a general principle, and adds
an explicit source_type-aware fact-vs-cultural usage rule.

v22b2 does NOT replace v20/v22b/v22b1 - all four remain independently
selectable so a genuine 4-way comparison stays possible. These tests check
the prompt TEXT asks for the right things; live model verification (the
actual IT125/ta/eu rerun) needs a real model - see the Colab instructions
in the same docs section.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts
from main import _tag_filename_for_final_cn_style

_SAMPLE_JUDGE_PLAN = {
    "selected_language": "it", "core_claim_to_counter": "gay people are sick",
    "recommended_strategy": ["factual_correction"],
    "approved_evidence": [{"source_id": "s1", "title": "t", "passage": "In many countries the law recognizes equal rights.",
                            "retrieval_score": 0.9, "source_type": "fact", "language": "it"}],
    "rejected_content": [], "cultural_guidance": ["Use warm, family-oriented framing."],
    "safety_guidance": [], "final_response_plan": "Rebut using the approved evidence.",
}


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
def test_v22b2_is_selectable():
    text = _v22b2()
    assert "EXACT-EVIDENCE-SCOPE RULES" in text


def test_v20_v22b_v22b1_remain_available_and_unchanged():
    v20, v22b, v22b1, v22b2 = _v20(), _v22b(), _v22b1(), _v22b2()
    assert len({v20, v22b, v22b1, v22b2}) == 4
    assert "EXACT-EVIDENCE-SCOPE RULES" not in v22b1
    assert "EXACT-EVIDENCE-SCOPE RULES" not in v22b
    assert "STRICT EVIDENCE-CONTROL RULES" in v22b1  # v22b1's own body, still intact
    assert prompts.build_final_cn_prompt("test comment", "it", _SAMPLE_JUDGE_PLAN, "unknown") == v20


def test_filename_suffix_for_v22b2_is_distinct():
    names = {s: _tag_filename_for_final_cn_style("it", s) for s in ("v20", "v22b", "v22b1", "v22b2")}
    assert names["v20"] == "it"
    assert names["v22b2"] == "it-finalcnv22b2"
    assert len(set(names.values())) == 4


def test_evidence_scope_rules_explicitly_apply_to_explanation_field_too():
    """Without this, a model could read the evidence-scope rules as
    governing only counter_narrative and let an unsupported country/legal
    claim slip into the separate explanation field instead."""
    text = _normalized(_v22b2())
    assert ("All exact-evidence-scope rules apply to both the counter_narrative "
            "and the explanation field.") in text


# ---------------------------------------------------------------------------
# 4-6: country/region-name gating (mechanical, enumerated list)
# ---------------------------------------------------------------------------
def test_prompt_forbids_including_italy_pattern():
    text = _normalized(_v22b2())
    assert "including Italy" in text


def test_prompt_forbids_named_regions_generally():
    text = _normalized(_v22b2())
    for phrase in ('"Italy"', '"Europe"', '"India"', '"Spain"', '"Basque Country"', '"Tamil Nadu"',
                   '"this country"', '"our country"'):
        assert phrase in text, f"missing forbidden-phrase entry: {phrase!r}"


def test_prompt_states_country_name_gated_on_exact_evidence_match():
    text = _normalized(_v22b2()).lower()
    assert "unless that exact name already" in text
    assert "appears in approved_evidence" in text


# ---------------------------------------------------------------------------
# 7-9: factual vs cultural evidence role separation
# ---------------------------------------------------------------------------
def test_prompt_explains_source_type_field_usage():
    text = _normalized(_v22b2())
    assert "source_type" in text
    assert '"fact"' in text and '"cultural"' in text and '"web"' in text


def test_prompt_forbids_cultural_guidance_as_factual_authorization():
    text = _normalized(_v22b2()).lower()
    assert "cultural_guidance field is guidance about tone only" in text
    assert "never be treated as evidence" in text


def test_prompt_describes_cultural_grounding_as_tone_not_invention():
    text = _normalized(_v22b2()).lower()
    assert "real cultural grounding means accurate tone and framing, not invented local detail" in text


# ---------------------------------------------------------------------------
# 10-11: hedging preservation + weak-evidence fallback
# ---------------------------------------------------------------------------
def test_prompt_requires_hedging_preservation():
    text = _normalized(_v22b2()).lower()
    assert "hedging preservation" in text
    assert "preserve that exact hedge or" in text


def test_prompt_includes_weak_evidence_fallback():
    text = _normalized(_v22b2()).lower()
    assert "sexual orientation is not a reason to reject or harm someone" in text


# ---------------------------------------------------------------------------
# 12: Italian-specific hard ban
# ---------------------------------------------------------------------------
def test_italian_prompt_bans_tra_cui_italia_unless_evidence_names_it():
    text = _v22b2(language="it")
    assert "tra cui l'Italia" in text
    assert "NEVER write" in text
    assert "molti Paesi" in text


# ---------------------------------------------------------------------------
# 13-14: Tamil / Basque tightening
# ---------------------------------------------------------------------------
def test_tamil_prompt_requires_exactly_two_simple_sentences():
    text = _v22b2(language="ta")
    assert "exactly 2 short, simple, complete sentences" in text
    assert "use exactly 2 short sentences" in text
    assert "everyday Tamil" in text


def test_basque_prompt_requires_two_sentences_and_bans_opposite_sex_wording():
    text = _v22b2(language="eu")
    assert "exactly 2 short, simple, complete sentences" in text
    assert "between men and women" in text
    assert "same-sex attraction or LGBTQ+ identity" in text


# ---------------------------------------------------------------------------
# 15: Spanish/English preserve v22b1 quality, add exact-scope rules
# ---------------------------------------------------------------------------
def test_spanish_prompt_preserves_quality_and_adds_scope_rule():
    text = _v22b2(language="es")
    assert "fluent Spanish speaker naturally would" in text  # carried over from v22b1
    assert "Spain/Europe/legal/social claims unless approved_evidence explicitly supports" in text


def test_english_prompt_preserves_quality_and_adds_scope_rule():
    text_in = _v22b2(language="en", region_context="Indian")
    assert "en_IN" in text_in
    assert "India/Europe/local-culture factual" in text_in
    text_eur = _v22b2(language="en", region_context="European")
    assert "en_EUR" in text_eur


if __name__ == "__main__":
    test_v22b2_is_selectable()
    test_v20_v22b_v22b1_remain_available_and_unchanged()
    test_filename_suffix_for_v22b2_is_distinct()
    test_evidence_scope_rules_explicitly_apply_to_explanation_field_too()
    test_prompt_forbids_including_italy_pattern()
    test_prompt_forbids_named_regions_generally()
    test_prompt_states_country_name_gated_on_exact_evidence_match()
    test_prompt_explains_source_type_field_usage()
    test_prompt_forbids_cultural_guidance_as_factual_authorization()
    test_prompt_describes_cultural_grounding_as_tone_not_invention()
    test_prompt_requires_hedging_preservation()
    test_prompt_includes_weak_evidence_fallback()
    test_italian_prompt_bans_tra_cui_italia_unless_evidence_names_it()
    test_tamil_prompt_requires_exactly_two_simple_sentences()
    test_basque_prompt_requires_two_sentences_and_bans_opposite_sex_wording()
    test_spanish_prompt_preserves_quality_and_adds_scope_rule()
    test_english_prompt_preserves_quality_and_adds_scope_rule()
    print("test_final_cn_prompt_v22b2.py: ALL PASSED")
