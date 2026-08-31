import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_quality import evaluate_language_quality, _REQUIRED_FIELDS
from model_api import ModelClient

_EXPECTED_KEYS = {"language_quality_score", "fluency_score", "terminology_score", "clarity_score",
                  "target_language_match", "quality_flag", "quality_issues", "needs_rewrite",
                  "problem_spans", "confidence_score"}


class _CannedScoreClient(ModelClient):
    """Returns a fixed score regardless of prompt content - each test wires
    up the specific score/issues it wants to simulate, mirroring how a real
    model's judgment would come back for a genuinely fluent vs genuinely
    garbled text."""
    backend_name = "stub"

    def __init__(self, canned: dict):
        self._canned = canned

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema == _REQUIRED_FIELDS:
            return dict(self._canned)
        return "n/a"


def test_fluent_spanish_cn_passes():
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="La homosexualidad no es una enfermedad mental, según las principales organizaciones de salud.",
        language="es", comment="Los gays estan locos.", core_claim="being gay is a mental illness",
    )
    assert result["quality_flag"] == "ok"
    assert result["needs_rewrite"] is False
    assert result["language_quality_score"] == 2.0
    assert result["quality_issues"] == []
    assert result["problem_spans"] == []


def test_garbled_basque_cn_is_flagged():
    """Mirrors the real observed case: EU130-style output with non-words
    like 'hormaera' - on-topic but not real Basque vocabulary."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 1, "terminology_score": 0,
        "clarity_score": 1, "quality_issues": ["garbled_terminology"],
        "problem_spans": ["hormaera"], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="Justizia eta hormaera orden moralaren oinarria direla uste dut.",
        language="eu", comment="Homosexualak beren homosexualitatetik askatu behar ditugu.",
        core_claim="homosexuality needs to be corrected",
    )
    assert result["quality_flag"] == "needs_review"
    assert result["needs_rewrite"] is True
    assert "garbled_terminology" in result["quality_issues"]
    assert "hormaera" in result["problem_spans"]


def test_garbled_tamil_identity_terms_is_flagged():
    """Mirrors the real observed case: V4_64-style output with garbled
    transliterations of LGBTQ+ terms ('ஹேஸ்லிஸ்', 'பானிஸ்மேஸ்லிஸ்')."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 1, "terminology_score": 0,
        "clarity_score": 1, "quality_issues": ["garbled_terminology", "hallucinated_wording"],
        "problem_spans": ["ஹேஸ்லிஸ்", "பானிஸ்மேஸ்லிஸ்"], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="லஸ்ட், ஹேஸ்லிஸ், பானிஸ்மேஸ்லிஸ் மற்றும் திரிபால் தொடர்புடைய சமூகங்களுக்கு பாதுகாப்பு வழங்குகிறது.",
        language="ta", comment="gay lesbian sercakkaiyalargal palial kurrangalukku thaan peyar ponavargal.",
        core_claim="LGBTQ+ people are criminals",
    )
    assert result["quality_flag"] == "needs_review"
    assert result["needs_rewrite"] is True
    assert result["terminology_score"] == 0
    assert "garbled_terminology" in result["quality_issues"]
    assert set(result["problem_spans"]) == {"ஹேஸ்லிஸ்", "பானிஸ்மேஸ்லிஸ்"}


def test_scoring_does_not_modify_the_generated_cn():
    original_cn = "La homosexualidad no es una enfermedad mental."
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
    })
    result = evaluate_language_quality(client, counter_narrative=original_cn, language="es", comment="x")
    # The function has no mechanism to return altered text - confirm no such
    # key exists, and confirm the caller's original string object is untouched.
    assert "counter_narrative" not in result
    assert original_cn == "La homosexualidad no es una enfermedad mental."


def test_string_false_target_language_match_is_not_treated_as_true():
    """The actual bug this guards against: bool("false") is True in Python.
    A model returning the quoted JSON string "false" instead of a real JSON
    boolean must still be treated as a mismatch, not silently flipped."""
    client = _CannedScoreClient({
        "target_language_match": "false", "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
    })
    result = evaluate_language_quality(client, counter_narrative="Some text", language="eu", comment="x")
    assert result["target_language_match"] is False
    assert result["quality_flag"] == "needs_review"
    assert result["needs_rewrite"] is True


def test_single_low_dimension_flags_even_if_average_is_above_threshold():
    """fluency=1, terminology=2, clarity=2 averages to 1.67 (above the 1.5
    threshold) - an average-only check would miss this, but a single bad
    dimension should still flag it."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 1, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": ["ungrammatical"], "problem_spans": [], "confidence_score": 2,
    })
    result = evaluate_language_quality(client, counter_narrative="Some text", language="eu", comment="x")
    assert result["language_quality_score"] > 1.5  # confirms the average alone would NOT have flagged
    assert result["quality_flag"] == "needs_review"
    assert result["needs_rewrite"] is True


def test_zero_confidence_forces_flag_even_with_perfect_scores():
    """The judge saying "I'm not confident judging this language" must
    override otherwise-perfect scores - a low-confidence 2/2/2 is exactly
    the failure mode this field exists to catch (a judge that doesn't
    genuinely know the language but defaults to a high score anyway)."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 0,
    })
    result = evaluate_language_quality(client, counter_narrative="Some text", language="ta", comment="x")
    assert result["quality_flag"] == "needs_review"
    assert result["needs_rewrite"] is True


def test_missing_confidence_score_defaults_to_not_confident():
    """A judge response that omits confidence_score entirely (e.g. an older/
    non-compliant model) must not be silently treated as fully confident."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [],
        # confidence_score deliberately omitted
    })
    result = evaluate_language_quality(client, counter_narrative="Some text", language="ta", comment="x")
    assert result["confidence_score"] == 0.0
    assert result["quality_flag"] == "needs_review"


def test_result_shape_is_stable_regardless_of_input():
    """Same keys every time, whether scored cleanly, flagged, or degraded to
    the fail-open fallback - callers (pipeline.py, main.py's CSV row
    builder) can rely on this shape unconditionally."""
    ok_client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
    })
    ok_result = evaluate_language_quality(ok_client, "text", "en", "comment")
    assert set(ok_result.keys()) == _EXPECTED_KEYS

    empty_result = evaluate_language_quality(ok_client, "", "en", "comment")
    assert set(empty_result.keys()) == _EXPECTED_KEYS
    assert empty_result["quality_flag"] == "unscored"


# ---------------------------------------------------------------------------
# Calibration set - real text from the actual Aug 2026 thesis-review batch
# (es/eu/ta, Qwen2.5-32B vs Mistral-Small-3.2-24B), paired with the judgment
# a WELL-CALIBRATED reviewer should reach. These use canned client responses,
# not a real LLM call (no network access in unit tests) - they check that
# OUR flagging logic does the right thing given a correct judgment, not that
# a real judge model actually produces a correct judgment. Whether a real
# judge model agrees is checked separately, live, against actual v20 CSVs.
# ---------------------------------------------------------------------------
def test_calibration_clean_spanish_qwen_output_passes():
    """Real ES114/Qwen output - confirmed clean by manual review."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="La homosexualidad no se considera una enfermedad mental desde hace décadas, "
                           "según las principales organizaciones de salud mental.",
        language="es", comment="Los gays están locos. Deberían volver al armario.",
    )
    assert result["quality_flag"] == "ok"


def test_calibration_known_bad_basque_qwen_output_is_flagged():
    """Real EU114/Qwen output that manual review found garbled ('Eguzkiaren
    luzatik geroztik...' - an unrelated sun metaphor)."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 0, "terminology_score": 1,
        "clarity_score": 0, "quality_issues": ["hallucinated_wording", "off_topic"],
        "problem_spans": ["Eguzkiaren luzatik geroztik"], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="Eguzkiaren luzatik geroztik, bereizketariko errespetua eta berdintasuna "
                           "dira gure herrialdeen oinarriak.",
        language="eu", comment="Homosexualak erotuta daude. Armairura itzuli beharko lukete.",
    )
    assert result["quality_flag"] == "needs_review"


def test_calibration_known_bad_tamil_qwen_output_is_flagged():
    """Real V4_64/Qwen output with garbled LGBTQ+ term transliterations."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 1, "terminology_score": 0,
        "clarity_score": 1, "quality_issues": ["garbled_terminology"],
        "problem_spans": ["ஹேஸ்லிஸ்", "பானிஸ்மேஸ்லிஸ்", "திரிபால்"], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="ஐக்கிய மனித உரிமைகள் மற்றும் நீதிமுறை லஸ்ட், ஹேஸ்லிஸ், பானிஸ்மேஸ்லிஸ் "
                           "மற்றும் திரிபால் தொடர்புடைய சமூகங்களுக்கு பாதுகாப்பு வழங்குகிறது.",
        language="ta", comment="பகிரங்கமான தவறான நடத்தை என வர்ணித்து, அவர்கள் பேச்சை அழிக்க வேண்டும் என்று வாதிடுகிறார்கள்.",
    )
    assert result["quality_flag"] == "needs_review"


def test_calibration_clean_mistral_basque_output_passes():
    """Real EU114/Mistral output - confirmed clean by manual review, used to
    check the diagnostic doesn't over-flag a model that's actually fine."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="Gizon-emakumezkoen arteko maitasunak beste modu batzuk ere baditu bere "
                           "izendatzeko. Giza eskubideek baieztatzen dute pertsona guztiek dutela "
                           "bera izan ahal izateko eskubidea.",
        language="eu", comment="Homosexualak erotuta daude. Armairura itzuli beharko lukete.",
    )
    assert result["quality_flag"] == "ok"


def test_calibration_clean_mistral_tamil_output_passes():
    """Real V4_697/Mistral output - confirmed clean by manual review."""
    client = _CannedScoreClient({
        "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
        "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
    })
    result = evaluate_language_quality(
        client,
        counter_narrative="ஒவ்வொருவரும் அவரது/அவரது பாலின அடையாளத்தை வெளிப்படுத்தும் உரிமையை உடையவர். "
                           "இது சமூகத்தில் ஒற்றுமைக்கு முக்கியமானது.",
        language="ta", comment="பெண்ணா ஆணா தெரியாத மாதிரி பேசிக் கொண்டு மக்களை கேலி செய்யும் திருநங்கைகள் "
                                "கேவலமாக இருக்கின்றனர்.",
    )
    assert result["quality_flag"] == "ok"


if __name__ == "__main__":
    test_fluent_spanish_cn_passes()
    test_garbled_basque_cn_is_flagged()
    test_garbled_tamil_identity_terms_is_flagged()
    test_scoring_does_not_modify_the_generated_cn()
    test_string_false_target_language_match_is_not_treated_as_true()
    test_single_low_dimension_flags_even_if_average_is_above_threshold()
    test_zero_confidence_forces_flag_even_with_perfect_scores()
    test_missing_confidence_score_defaults_to_not_confident()
    test_result_shape_is_stable_regardless_of_input()
    test_calibration_clean_spanish_qwen_output_passes()
    test_calibration_known_bad_basque_qwen_output_is_flagged()
    test_calibration_known_bad_tamil_qwen_output_is_flagged()
    test_calibration_clean_mistral_basque_output_passes()
    test_calibration_clean_mistral_tamil_output_passes()
    print("test_language_quality.py: ALL PASSED")
