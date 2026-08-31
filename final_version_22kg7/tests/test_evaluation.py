"""
Covers evaluation.score_language_quality() - the v22 wiring that reuses
language_quality.py (calibrated diagnostic scorer) as an OPTIONAL,
evaluation-time-only diagnostic, gated behind `main.py evaluate
--language-quality`. Not coupled to pipeline.py's live generation path
(that would be a v21-style wiring, deliberately not present on the v20/v22
branch - see language_quality.py/prompts.py module docstrings).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_quality import _REQUIRED_FIELDS
from model_api import ModelClient
import evaluation
import utils


class _CannedScoreClient(ModelClient):
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema == _REQUIRED_FIELDS:
            return {
                "target_language_match": True, "fluency_score": 2, "terminology_score": 2,
                "clarity_score": 2, "quality_issues": [], "problem_spans": [], "confidence_score": 2,
            }
        return "n/a"


def _sample_df():
    return pd.DataFrame([
        {"hate_text": "Los gays estan locos.", "generated_counter_narrative": "La homosexualidad no es una enfermedad.",
         "language": "es"},
        {"hate_text": "Homosexualak erotuta daude.", "generated_counter_narrative": "Homosexualitatea ez da gaixotasuna.",
         "language": "eu"},
    ])


def test_score_language_quality_adds_lq_prefixed_columns():
    df = _sample_df()
    result = evaluation.score_language_quality(_CannedScoreClient(), df)
    for col in ("lq_language_quality_score", "lq_fluency_score", "lq_terminology_score",
                "lq_clarity_score", "lq_target_language_match", "lq_quality_flag",
                "lq_quality_issues", "lq_needs_rewrite", "lq_problem_spans", "lq_confidence_score"):
        assert col in result.columns, f"missing expected column {col}"


def test_score_language_quality_does_not_modify_original_columns():
    df = _sample_df()
    result = evaluation.score_language_quality(_CannedScoreClient(), df)
    assert result["generated_counter_narrative"].tolist() == df["generated_counter_narrative"].tolist()
    assert result["hate_text"].tolist() == df["hate_text"].tolist()
    assert result["language"].tolist() == df["language"].tolist()


def test_score_language_quality_preserves_row_count_and_order():
    df = _sample_df()
    result = evaluation.score_language_quality(_CannedScoreClient(), df)
    assert len(result) == len(df)
    assert result["lq_quality_flag"].tolist() == ["ok", "ok"]


# ---------------------------------------------------------------------------
# unsafe_keyword_rate must exclude non-evaluable-language rows (v22), not
# count them as "safe" - see utils.check_unsafe_keywords_with_context's
# Optional[bool]-style "evaluable" contract. NOTE: rag_pipeline._quality_filter
# is deliberately NOT language-aware (frozen to v20 behavior, evaluation-only
# scope) - see tests/test_unsafe_keywords.py's module docstring.
# ---------------------------------------------------------------------------
def _generations_df(language: str, texts: list) -> pd.DataFrame:
    return pd.DataFrame([
        {"generated_counter_narrative": t, "reference_counter_narrative": "", "language": language, "error": ""}
        for t in texts
    ])


def _isolate_empty_unsafe_patterns(tmp_path, monkeypatch, language: str):
    """Points utils' curated-patterns lookup at an isolated, empty file -
    used by tests asserting "non-evaluable language" behavior, so they don't
    depend on the ambient real data/unsafe_patterns_by_lang.json's current
    validation state (which legitimately changes over time - a hardcoded
    'es is non-evaluable' assumption broke during development the moment
    real Spanish candidates were validated and promoted in)."""
    import json
    path = tmp_path / "unsafe_patterns_by_lang.json"
    path.write_text(json.dumps({language: {"status": "not_yet_validated", "terms": []}}), encoding="utf-8")
    monkeypatch.setattr(utils, "_UNSAFE_PATTERNS_PATH", path)
    monkeypatch.setattr(utils, "_unsafe_patterns_cache", None)


def test_unsafe_keyword_rate_is_real_number_for_evaluable_language():
    df = _generations_df("en", ["A perfectly normal reply.", "That is a sin against nature."])
    summary = evaluation.summarize(df)
    assert summary["unsafe_keyword_rate"] == 0.5
    assert summary["unsafe_keyword_evaluable"] == 2
    assert summary["non_evaluable_language_count"] == 0


def test_unsafe_keyword_rate_is_none_for_non_evaluable_language(tmp_path, monkeypatch):
    """Spanish has no validated keyword list yet (isolated fixture, not the
    real shipped file) - the rate must be None, not 0.0 (which would
    misleadingly claim "checked, found nothing")."""
    _isolate_empty_unsafe_patterns(tmp_path, monkeypatch, "es")
    df = _generations_df("es", ["Todos merecen respeto sin discriminación.", "Una respuesta normal."])
    summary = evaluation.summarize(df)
    assert summary["unsafe_keyword_rate"] is None
    assert summary["unsafe_keyword_evaluable"] == 0
    assert summary["non_evaluable_language_count"] == 1


# ---------------------------------------------------------------------------
# Stage 8: compute_row_metrics/summarize expose the full structured
# unsafe-keyword field set, and requires_review is tracked separately from
# the auto-unsafe flag/rate.
# ---------------------------------------------------------------------------
def test_compute_row_metrics_exposes_structured_unsafe_keyword_fields():
    row = {"generated_counter_narrative": "That's not an abomination, it's a family.",
           "reference_counter_narrative": "", "language": "en"}
    metrics = evaluation.compute_row_metrics(row)
    assert metrics["unsafe_keyword_flag"] is False
    assert metrics["unsafe_keyword_requires_review"] is True
    assert "abomination" in metrics["unsafe_keyword_matches"]
    assert metrics["unsafe_keyword_evaluable"] is True
    assert metrics["unsafe_keyword_language_scope"] == "en"


def test_compute_row_metrics_non_evaluable_language_has_empty_structured_fields(tmp_path, monkeypatch):
    _isolate_empty_unsafe_patterns(tmp_path, monkeypatch, "es")
    row = {"generated_counter_narrative": "Una respuesta cualquiera.",
           "reference_counter_narrative": "", "language": "es"}
    metrics = evaluation.compute_row_metrics(row)
    assert metrics["unsafe_keyword_flag"] is None
    assert metrics["unsafe_keyword_evaluable"] is False
    assert metrics["unsafe_keyword_matches"] == []
    assert metrics["unsafe_keyword_requires_review"] is False


def test_summarize_requires_review_rate_does_not_affect_unsafe_rate():
    """A negated match must show up in unsafe_keyword_requires_review_rate
    but must NOT inflate unsafe_keyword_rate - those are separate signals."""
    df = _generations_df("en", ["That's not an abomination, it's a family.",
                                 "A perfectly normal reply."])
    summary = evaluation.summarize(df)
    assert summary["unsafe_keyword_rate"] == 0.0
    assert summary["unsafe_keyword_requires_review_rate"] == 0.5


if __name__ == "__main__":
    test_score_language_quality_adds_lq_prefixed_columns()
    test_score_language_quality_does_not_modify_original_columns()
    test_score_language_quality_preserves_row_count_and_order()
    test_unsafe_keyword_rate_is_real_number_for_evaluable_language()
    test_compute_row_metrics_exposes_structured_unsafe_keyword_fields()
    test_summarize_requires_review_rate_does_not_affect_unsafe_rate()
    print("test_evaluation.py: most tests passed (tmp_path/monkeypatch-dependent "
          "tests need pytest, run via `pytest tests/test_evaluation.py`)")
