"""
Covers v22b.5 (prompts.py section 7): a Tamil native-fluency + direct-
rebuttal hotfix on top of v22b4. Motivated by native review of v22b4's real
Tamil rerun: structurally solid (5/5 exactly-2-sentences, no errors), but
(1) some phrasing read as unnatural/translated ("ஒவ்வொருவருமானும்", "மனித
வரைவுகள்", "உள்ளூர்மக்கள் எல்லோரும்"); (2) V4_534 (linking gay/lesbian
people to sexual crimes) got only generic human-rights wording instead of
directly rejecting the specific criminal-generalization claim; (3) V4_64's
explanation credited "அறிவியல் அடிப்படையில்" (scientific basis) when
approved_evidence was cultural-only - confirming the v22b3/v22b4
explanation-safety fix didn't generalize from its Italian-named examples to
Tamil's equivalent trigger; (4) V4_67 asserted a strong theological claim
("God does not hate anyone") only loosely grounded in the actual evidence.

v22b5 does NOT replace v20/v22b/v22b1/v22b2/v22b3/v22b4 - all seven remain
independently selectable, and v22b4's Italian rules/explanation rules are
reused unchanged. These tests check the prompt TEXT asks for the right
things; live model verification needs a real model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts
from main import _tag_filename_for_final_cn_style

_SAMPLE_JUDGE_PLAN = {
    "selected_language": "ta", "core_claim_to_counter": "gay people are criminals",
    "recommended_strategy": ["factual_correction"],
    "approved_evidence": [{"source_id": "s1", "title": "t", "passage": "homosexuality is natural.",
                            "retrieval_score": 0.9, "source_type": "cultural", "language": "ta"}],
    "rejected_content": [], "cultural_guidance": ["Use respectful framing."],
    "safety_guidance": [], "final_response_plan": "Rebut using the approved evidence.",
}


def _v22b5(language="ta", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b5")


def _v22b4(language="ta", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b4")


def _v20(language="ta", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v20")


def _normalized(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# v22b5 exists and is selectable / v22b4 remains unchanged / filename suffix
# ---------------------------------------------------------------------------
def test_v22b5_is_selectable():
    text = _v22b5()
    assert "DIRECTLY REBUT THE SPECIFIC HARMFUL CLAIM" in text


def test_v22b4_remains_unchanged():
    v22b4_text = _v22b4()
    assert "DIRECTLY REBUT THE SPECIFIC HARMFUL CLAIM" not in v22b4_text
    assert "STRICT EVIDENCE-CONTROL RULES" not in v22b4_text  # sanity: not v22b1's body either
    # v22b4's own Italian-rule content must still be present, untouched
    it_text = prompts.build_final_cn_prompt("c", "it", _SAMPLE_JUDGE_PLAN, "unknown", style="v22b4")
    assert "dal proprio orientamento sessuale" in it_text
    v20, v22b4, v22b5 = _v20(), _v22b4(), _v22b5()
    assert len({v20, v22b4, v22b5}) == 3
    assert prompts.build_final_cn_prompt("test comment", "ta", _SAMPLE_JUDGE_PLAN, "unknown") == v20


def test_filename_suffix_for_v22b5_is_distinct():
    names = {s: _tag_filename_for_final_cn_style("ta", s)
             for s in ("v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5")}
    assert names["v20"] == "ta"
    assert names["v22b5"] == "ta-finalcnv22b5"
    assert len(set(names.values())) == 7


# ---------------------------------------------------------------------------
# Tamil must directly rebut the specific hate claim (not just generic
# equality/human-rights wording)
# ---------------------------------------------------------------------------
def test_tamil_requires_direct_rebuttal_not_generic_wording():
    text = _normalized(_v22b5())
    assert "do not fall back to only generic equality/human-rights wording" in text
    assert "it links LGBTQ+ people to crime, say it is wrong to blame a whole group" in text
    assert "calling someone's identity a curse is harmful and disrespectful" in text
    assert "silencing people because of their identity is unfair" in text


# ---------------------------------------------------------------------------
# Tamil bans the specific awkward/translated-sounding phrases found
# ---------------------------------------------------------------------------
def test_tamil_bans_awkward_phrases_found_in_live_review():
    text = _v22b5()
    for phrase in ("ஒவ்வொருவருமானும்", "மனித வரைவுகள்", "உள்ளூர்மக்கள் எல்லோரும்",
                   "பல்வகைப் படைப்புகள்"):
        assert phrase in text, f"missing banned phrase: {phrase!r}"


def test_tamil_avoids_unsupported_identity_category_listing():
    text = _v22b5()
    assert "ஆண்பால், பெண்ண்பால், திருநங்கை/திருநம்பாள்" in text
    assert "LGBTQ+ மக்கள்" in text  # the preferred broader alternative


# ---------------------------------------------------------------------------
# Tamil explanation cannot mention science/religion/law unless exact
# fact/web evidence supports it
# ---------------------------------------------------------------------------
def test_tamil_explanation_gates_science_religion_law_claims():
    text = _normalized(_v22b5())
    assert "அறிவியல் அடிப்படையில்" in text
    assert "சமயப் படிப்புகள் கூறுகின்றன" in text
    assert "சட்டம் கூறுகிறது" in text
    assert 'source_type EXACTLY "fact" or EXACTLY "web" and explicitly supports that exact claim' in text


# ---------------------------------------------------------------------------
# Tamil religious fallback avoids strong unsupported theological claims
# ---------------------------------------------------------------------------
def test_tamil_religious_fallback_avoids_strong_theological_claim():
    text = _v22b5()
    assert "கடவுள் எவரையும் வெறுக்கவில்லை" in text  # cited as the risky example to avoid
    assert "ஒருவரின் அடையாளத்தை சாபமாகச் சொல்லுவது காயப்படுத்தும்" in text  # the safer alternative


if __name__ == "__main__":
    test_v22b5_is_selectable()
    test_v22b4_remains_unchanged()
    test_filename_suffix_for_v22b5_is_distinct()
    test_tamil_requires_direct_rebuttal_not_generic_wording()
    test_tamil_bans_awkward_phrases_found_in_live_review()
    test_tamil_avoids_unsupported_identity_category_listing()
    test_tamil_explanation_gates_science_religion_law_claims()
    test_tamil_religious_fallback_avoids_strong_theological_claim()
    print("test_final_cn_prompt_v22b5.py: ALL PASSED")
