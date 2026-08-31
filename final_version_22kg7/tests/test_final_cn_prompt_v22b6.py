"""
Covers v22b.6 (prompts.py section 7): a Tamil style-anchor native-fluency
hotfix on top of v22b5. Motivated by native review of v22b5's real Tamil
rerun - genuine progress (V4_534 addressed the crime claim more directly,
V4_64's explanation stopped inventing science/research support, all 5
stayed exactly 2 sentences) but not yet thesis-ready: more unnatural/
translated phrasing found; V4_64 still didn't directly name silencing/
suppression; V4_67 still used a near-theological claim only loosely
grounded in evidence; one explanation switched to English despite the
existing "same language as input" instruction (present since v20).

v22b6 does NOT replace v20/v22b/v22b1/v22b2/v22b3/v22b4/v22b5 - all eight
remain independently selectable, and v22b5's Italian rules/explanation
rules are reused unchanged. These tests check the prompt TEXT asks for the
right things; live model verification needs a real model.
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


def _v22b6(language="ta", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b6")


def _v22b5(language="ta", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b5")


def _v20(language="ta", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v20")


def _normalized(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 1-3: style selection / v22b5 unchanged / filename suffix
# ---------------------------------------------------------------------------
def test_v22b6_is_selectable():
    text = _v22b6()
    assert "STYLE ANCHORS" in text


def test_v22b5_remains_unchanged():
    v22b5_text = _v22b5()
    assert "STYLE ANCHORS" not in v22b5_text
    assert "DIRECTLY REBUT THE SPECIFIC HARMFUL CLAIM" in v22b5_text  # v22b5's own body intact
    v20, v22b5, v22b6 = _v20(), _v22b5(), _v22b6()
    assert len({v20, v22b5, v22b6}) == 3
    assert prompts.build_final_cn_prompt("test comment", "ta", _SAMPLE_JUDGE_PLAN, "unknown") == v20


def test_filename_suffix_for_v22b6_is_distinct():
    names = {s: _tag_filename_for_final_cn_style("ta", s)
             for s in ("v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6")}
    assert names["v20"] == "ta"
    assert names["v22b6"] == "ta-finalcnv22b6"
    assert len(set(names.values())) == 8


# ---------------------------------------------------------------------------
# 4-5: examples framed as style anchors, not mandatory templates
# ---------------------------------------------------------------------------
def test_tamil_prompt_frames_examples_as_style_anchors_not_templates():
    text = _normalized(_v22b6())
    assert "not fixed templates" in text
    assert "Do not force exact template copying" in text


def test_tamil_prompt_forbids_blind_copying():
    text = _normalized(_v22b6()).lower()
    assert "not text to copy verbatim" in text
    assert "only reuse an anchor's exact wording if it genuinely fits this exact case" in text
    assert "instance-specific and evidence-grounded" in text


# ---------------------------------------------------------------------------
# 6: direct rebuttal of the specific hate claim required
# ---------------------------------------------------------------------------
def test_tamil_requires_direct_rebuttal_of_specific_claim():
    text = _normalized(_v22b6())
    assert "DIRECTLY REBUT THE SPECIFIC HARMFUL CLAIM" in text
    assert "do not fall back to only generic equality/human-rights wording" in text


# ---------------------------------------------------------------------------
# 7-9: style anchors present for crime / silencing / curse-religion
# ---------------------------------------------------------------------------
def test_tamil_crime_generalization_style_anchor_present():
    text = _v22b6()
    assert "பொறுப்பாகக் காட்டுவது தவறு" in text
    assert "பாலியல் குற்றங்களுக்குப் பொறுப்பற்றார்கள்" in text  # named as the awkward form to avoid


def test_tamil_silencing_style_anchor_present_and_required():
    text = _normalized(_v22b6())
    assert "குரலை அடக்குவது நியாயமல்ல" in text
    assert "MUST include a natural equivalent of" in text


def test_tamil_curse_religion_style_anchor_avoids_strong_theology():
    text = _normalized(_v22b6())
    assert "சாபமாகச் சொல்லுவது காயப்படுத்தும்" in text
    assert "கடவுள் எவரையும் வெறுக்கவில்லை" in text  # named as risky
    assert "கடவுளின் அன்பும் மரியாதையும்" in text  # named as risky (this stage's new finding)
    assert "God loves" in text and "God hates no one" in text
    assert 'unless approved_evidence' in text and 'theological claim' in text


# ---------------------------------------------------------------------------
# 10: newly banned unnatural phrases present
# ---------------------------------------------------------------------------
def test_new_banned_tamil_phrases_present():
    text = _v22b6()
    for phrase in ("அனைத்து மக்களும் தனித்தன்மையைச் சேர்ந்தவர்கள்",
                   "அவர்களின் அடையாளம் அவர்களை சிறப்பாக்குகிறது",
                   "பாலியல் குற்றங்களுக்குப் பொறுப்பற்றார்கள்",
                   "கடவுளின் அன்பும் மரியாதையும் அனைத்து மக்களுக்கும் பொருந்துகிறது"):
        assert phrase in text, f"missing newly-banned phrase: {phrase!r}"


def test_v22b5_banned_phrases_still_present():
    """Carried-over bans from v22b5 must not have been dropped."""
    text = _v22b6()
    for phrase in ("ஒவ்வொருவருமானும்", "மனித வரைவுகள்", "உள்ளூர்மக்கள் எல்லோரும்"):
        assert phrase in text, f"missing carried-over banned phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# 11: Tamil explanation must stay in Tamil (target-language design intent,
# confirmed unchanged since v20's "same language as input" instruction)
# ---------------------------------------------------------------------------
def test_tamil_explanation_must_stay_in_tamil_not_english():
    text = _normalized(_v22b6())
    assert "NEVER switch the explanation to English" in text
    assert "Tamil explanation must ALSO stay written in Tamil - never switch to English" in text


# ---------------------------------------------------------------------------
# 12: Tamil explanation remains method-level and source-safe (v22b5's rules
# carried over unchanged)
# ---------------------------------------------------------------------------
def test_tamil_explanation_still_gates_science_religion_law():
    text = _normalized(_v22b6())
    assert "அறிவியல் அடிப்படையில்" in text
    assert "சமயப் படிப்புகள் கூறுகின்றன" in text
    assert "சட்டம் கூறுகிறது" in text
    assert "method-level summary only, not a second" in text.lower()


if __name__ == "__main__":
    test_v22b6_is_selectable()
    test_v22b5_remains_unchanged()
    test_filename_suffix_for_v22b6_is_distinct()
    test_tamil_prompt_frames_examples_as_style_anchors_not_templates()
    test_tamil_prompt_forbids_blind_copying()
    test_tamil_requires_direct_rebuttal_of_specific_claim()
    test_tamil_crime_generalization_style_anchor_present()
    test_tamil_silencing_style_anchor_present_and_required()
    test_tamil_curse_religion_style_anchor_avoids_strong_theology()
    test_new_banned_tamil_phrases_present()
    test_v22b5_banned_phrases_still_present()
    test_tamil_explanation_must_stay_in_tamil_not_english()
    test_tamil_explanation_still_gates_science_religion_law()
    print("test_final_cn_prompt_v22b6.py: ALL PASSED")
