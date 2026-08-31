"""
Covers scripts/extract_unsafe_candidates.py's core logic with small
synthetic inputs - not the real dataset (that requires a network fetch of
ML_MTCONAN_KN and is exercised by actually running the script, not a unit
test). Candidates-only output: nothing here is a validated unsafe keyword.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_unsafe_candidates import _build_candidate_table, _ngrams_in_document, _resolve_language_bucket, _tokenize


def test_tokenize_keeps_tamil_syllables_intact():
    """The actual bug this guards against: a plain `\\w+` regex splits
    Tamil combining vowel signs (Unicode category Mark) away from their
    base consonant, producing meaningless single-mark fragments instead of
    whole words. Confirmed via live testing before this fix."""
    tokens = _tokenize("சிலர் திருநங்கை")
    assert tokens == ["சிலர்", "திருநங்கை"]
    assert all(len(t) > 1 for t in tokens), "no token should be a single combining mark"


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert _tokenize("Hello, World!") == ["hello", "world"]


def test_ngrams_in_document_returns_unigrams_bigrams_trigrams():
    terms = _ngrams_in_document(["a", "b", "c"])
    assert terms == {"a", "b", "c", "a b", "b c", "a b c"}


def test_ngrams_in_document_deduplicates_within_one_document():
    """A term appearing twice in the same document counts once for that
    document's contribution to hate_count/reference_count (document
    frequency, not raw token frequency)."""
    terms = _ngrams_in_document(["x", "x"])
    assert terms == {"x", "x x"}


def test_resolve_language_bucket_splits_english_by_region():
    assert _resolve_language_bucket({"language": "en", "region": "European"}) == "en_EUR"
    assert _resolve_language_bucket({"language": "en", "region": "Indian"}) == "en_IN"
    assert _resolve_language_bucket({"language": "en", "region": None}) == "en"


def test_resolve_language_bucket_leaves_other_languages_unsplit():
    for lang in ("es", "eu", "it", "ta"):
        assert _resolve_language_bucket({"language": lang, "region": "European"}) == lang


def test_build_candidate_table_association_score_favors_hate_only_terms():
    hate_texts = ["gays are disgusting", "gays are sick"]
    ref_texts = ["everyone deserves respect"]
    df = _build_candidate_table("en", hate_texts, ref_texts)
    disgusting_row = df[df["term"] == "disgusting"].iloc[0]
    respect_row = df[df["term"] == "respect"] if (df["term"] == "respect").any() else None
    assert disgusting_row["hate_count"] == 1
    assert disgusting_row["reference_count"] == 0
    assert disgusting_row["hate_association_score"] > 0
    # "respect" only appears in reference text, never hate_text - it must
    # not appear in the candidate table at all (candidates are drawn from
    # hate_text occurrences only, per _MIN_HATE_COUNT).
    assert respect_row is None or respect_row.empty


def test_build_candidate_table_includes_example_snippets():
    df = _build_candidate_table("en", ["gays are disgusting"], [])
    row = df[df["term"] == "disgusting"].iloc[0]
    assert "gays are disgusting" in row["example_hate_snippets"]


def test_build_candidate_table_is_sorted_by_score_descending():
    hate_texts = ["a b c", "a b c", "a b c", "d"]
    df = _build_candidate_table("en", hate_texts, [])
    scores = df["hate_association_score"].tolist()
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    test_tokenize_keeps_tamil_syllables_intact()
    test_tokenize_lowercases_and_splits_on_punctuation()
    test_ngrams_in_document_returns_unigrams_bigrams_trigrams()
    test_ngrams_in_document_deduplicates_within_one_document()
    test_resolve_language_bucket_splits_english_by_region()
    test_resolve_language_bucket_leaves_other_languages_unsplit()
    test_build_candidate_table_association_score_favors_hate_only_terms()
    test_build_candidate_table_includes_example_snippets()
    test_build_candidate_table_is_sorted_by_score_descending()
    print("test_extract_unsafe_candidates.py: ALL PASSED")
