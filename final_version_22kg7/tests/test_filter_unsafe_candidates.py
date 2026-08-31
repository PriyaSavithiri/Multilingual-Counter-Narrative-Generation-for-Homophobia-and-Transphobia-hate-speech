"""
Covers scripts/filter_unsafe_candidates.py's filtering rules with small
synthetic inputs. Filtering removes obvious non-useful candidates only -
this still produces input for Stage 5 validation, not final truth, so these
tests check that filtering is conservative (doesn't remove things it
shouldn't) as much as that it removes what it should.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from filter_unsafe_candidates import filter_candidates


def _row(term, ngram_size=1, hate_count=5, reference_count=0):
    return {"language": "en", "term": term, "ngram_size": ngram_size,
            "hate_count": hate_count, "reference_count": reference_count,
            "hate_association_score": 1.0, "example_hate_snippets": "example"}


def test_removes_very_short_unigrams():
    df = pd.DataFrame([_row("u"), _row("kill")])
    result = filter_candidates(df, "en")
    assert "u" not in result["term"].tolist()
    assert "kill" in result["term"].tolist()


def test_removes_pure_number_unigrams():
    df = pd.DataFrame([_row("2020"), _row("disgusting")])
    result = filter_candidates(df, "en")
    assert "2020" not in result["term"].tolist()
    assert "disgusting" in result["term"].tolist()


def test_removes_punctuation_only_terms():
    df = pd.DataFrame([_row("..."), _row("kill")])
    result = filter_candidates(df, "en")
    assert "..." not in result["term"].tolist()


def test_removes_english_stopwords():
    df = pd.DataFrame([_row("the"), _row("kill")])
    result = filter_candidates(df, "en")
    assert "the" not in result["term"].tolist()
    assert "kill" in result["term"].tolist()


def test_does_not_apply_stopword_filter_where_no_validated_list_exists():
    """Tamil has no validated stopword source (stopwordsiso doesn't cover
    it) - filtering must not silently invent one."""
    df = pd.DataFrame([{"language": "ta", "term": "இந்த", "ngram_size": 1,
                         "hate_count": 5, "reference_count": 0,
                         "hate_association_score": 1.0, "example_hate_snippets": "x"}])
    result = filter_candidates(df, "ta")
    assert "இந்த" in result["term"].tolist()


def test_removes_known_neutral_identity_terms():
    df = pd.DataFrame([_row("gay"), _row("lesbian"), _row("kill")])
    result = filter_candidates(df, "en")
    assert "gay" not in result["term"].tolist()
    assert "lesbian" not in result["term"].tolist()
    assert "kill" in result["term"].tolist()


def test_removes_neutral_identity_terms_from_non_english_buckets_too():
    """English loanwords ("gay") appearing in non-English text (e.g. code-
    switching in Tamil social media) must still be excluded - confirmed via
    live testing that this was NOT happening before the fix."""
    df = pd.DataFrame([{"language": "ta", "term": "gay", "ngram_size": 1,
                         "hate_count": 5, "reference_count": 0,
                         "hate_association_score": 1.0, "example_hate_snippets": "x"}])
    result = filter_candidates(df, "ta")
    assert "gay" not in result["term"].tolist()


def test_removes_bigrams_composed_entirely_of_neutral_terms():
    df = pd.DataFrame([_row("gay lesbian", ngram_size=2), _row("kill gay", ngram_size=2)])
    result = filter_candidates(df, "en")
    assert "gay lesbian" not in result["term"].tolist()
    # "kill gay" is NOT entirely neutral words - "kill" isn't a neutral term - must survive.
    assert "kill gay" in result["term"].tolist()


def test_removes_terms_balanced_between_hate_and_reference():
    df = pd.DataFrame([_row("people", hate_count=10, reference_count=8), _row("kill", hate_count=10, reference_count=0)])
    result = filter_candidates(df, "en")
    assert "people" not in result["term"].tolist()
    assert "kill" in result["term"].tolist()


def test_removes_terms_with_too_few_hate_occurrences():
    df = pd.DataFrame([_row("rare", hate_count=1), _row("kill", hate_count=5)])
    result = filter_candidates(df, "en", min_hate_count=2)
    assert "rare" not in result["term"].tolist()
    assert "kill" in result["term"].tolist()


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame(columns=["language", "term", "ngram_size", "hate_count",
                                "reference_count", "hate_association_score", "example_hate_snippets"])
    result = filter_candidates(df, "en")
    assert result.empty


if __name__ == "__main__":
    test_removes_very_short_unigrams()
    test_removes_pure_number_unigrams()
    test_removes_punctuation_only_terms()
    test_removes_english_stopwords()
    test_does_not_apply_stopword_filter_where_no_validated_list_exists()
    test_removes_known_neutral_identity_terms()
    test_removes_neutral_identity_terms_from_non_english_buckets_too()
    test_removes_bigrams_composed_entirely_of_neutral_terms()
    test_removes_terms_balanced_between_hate_and_reference()
    test_removes_terms_with_too_few_hate_occurrences()
    test_empty_dataframe_returns_empty()
    print("test_filter_unsafe_candidates.py: ALL PASSED")
