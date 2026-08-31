"""
Regression coverage for the v22 unsafe-keyword redesign. The original
heuristic checked utils.UNSAFE_KEYWORDS_EN (an English-only list) against
text regardless of its actual language - Spanish "sin" means "without"
(e.g. "sin discriminación") and is one of the most common function words in
the language, so it was inflating evaluation.py's unsafe_keyword_rate on
Spanish output with a false safety signal.

The fix is not just "exclude the word sin" - it's a language-VALIDATION
model: contains_unsafe_keywords() now returns True/False only for languages
with an actual validated keyword list (currently just "en"), and None for
every other language, since applying an English list to Spanish/Italian/
Basque/Tamil was never validated to be safe to do at all, not just for the
one "sin" collision. None must never be treated as "checked and safe" by
any caller - see evaluation.py's unsafe_keyword_rate, which explicitly
excludes None rows rather than counting them as 0 (safe).

Scope: this fix is EVALUATION-ONLY, by explicit decision. It does not touch
rag_pipeline.py's corpus-build filter - that filter (_quality_filter) was
found to feed the FAISS retrieval corpus used at generation time, so
changing its behavior would make v22 non-generation-identical to v20. It
was reverted to inline v20's original unconditional/bare-substring check
directly (see rag_pipeline.py's own docstring), deliberately NOT reusing
this module's contains_unsafe_keywords() at all, so it stays frozen against
future changes here. That means the Spanish "sin" collision this file
documents is fixed for evaluation metrics, but NOT fixed at the RAG-corpus-
build layer - see test_quality_filter_still_drops_spanish_sin_by_design
below for that intentional, documented discrepancy.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_pipeline
from utils import contains_unsafe_keywords


def test_flags_real_english_sin():
    assert contains_unsafe_keywords("That is a sin against nature.", "en") is True


def test_english_keyword_list_is_not_applied_to_spanish():
    """Even an unambiguous English keyword like "kill" must NOT be checked
    against Spanish text - there is no validated Spanish list yet, so this
    must return None, not True and not False."""
    assert contains_unsafe_keywords("Deberían matar a todos.", "es") is None


def test_spanish_sin_is_not_flagged():
    assert contains_unsafe_keywords("Todos merecen respeto sin discriminación.", "es") is None


def test_italian_text_is_not_checked_using_english_keywords():
    assert contains_unsafe_keywords("Questo e disgusting davvero.", "it") is None


def test_basque_text_is_not_checked_using_english_keywords():
    assert contains_unsafe_keywords("Hau abomination bat da.", "eu") is None


def test_tamil_text_is_not_checked_using_english_keywords():
    assert contains_unsafe_keywords("இது kill செய்ய வேண்டும்.", "ta") is None


def test_returns_none_when_no_validated_list_exists():
    """Direct check of the contract itself, independent of any specific
    language or keyword - a language absent from
    utils.VALIDATED_UNSAFE_KEYWORDS must return None."""
    assert contains_unsafe_keywords("anything at all", "fr") is None


def test_language_parameter_is_required_not_defaulted():
    """v20 silently defaulted language="en" - exactly the bug this redesign
    fixes (English keywords applied to text by accident when a caller forgot
    to pass a language). language is now a required parameter."""
    with pytest.raises(TypeError):
        contains_unsafe_keywords("some text")


def test_word_boundary_not_bare_substring_even_for_english():
    """"sin" as a substring inside unrelated English words must not match -
    independent of the cross-language issue, a bare `kw in text` check was
    already a false-positive source for English-only text too."""
    assert contains_unsafe_keywords("I am using this cousin's laptop since Monday.", "en") is False


def test_neutral_identity_terms_are_not_flagged_in_english():
    for term in ["gay", "lesbian", "trans", "bisexual", "queer", "homosexual", "LGBTQ+"]:
        assert contains_unsafe_keywords(f"This is a story about {term} people.", "en") is False, (
            f"neutral identity term {term!r} must never be flagged as unsafe"
        )


def test_en_eur_falls_back_to_base_english_list():
    assert contains_unsafe_keywords("That is a sin.", "en_EUR") is True
    assert contains_unsafe_keywords("A perfectly normal sentence.", "en_EUR") is False


def test_en_in_falls_back_to_base_english_list():
    assert contains_unsafe_keywords("That is a sin.", "en_IN") is True
    assert contains_unsafe_keywords("A perfectly normal sentence.", "en_IN") is False


def test_quality_filter_still_drops_spanish_sin_by_design():
    """rag_pipeline._quality_filter is intentionally NOT language-aware -
    unlike contains_unsafe_keywords(), it inlines v20's original
    unconditional bare-substring check against every record regardless of
    language, so the RAG corpus (and therefore retrieval/generation) stays
    identical to v20. This means the "sin"/Spanish false-positive this file
    is otherwise about is deliberately still present here - see this
    module's docstring and rag_pipeline._quality_filter's own docstring for
    why. If this test ever starts failing because someone "fixed" the
    corpus filter to be language-aware, that's a generation-path behavior
    change and must be raised with the user, not silently accepted."""
    records = [
        {"reference_counter_narrative": "Todos merecen respeto sin discriminación por su identidad.",
         "knowledge_text": "", "language": "es"},
    ]
    kept = rag_pipeline._quality_filter(records)
    assert len(kept) == 0


def test_quality_filter_still_drops_genuinely_unsafe_english_records():
    records = [
        {"reference_counter_narrative": "They should just kill themselves.",
         "knowledge_text": "", "language": "en"},
    ]
    kept = rag_pipeline._quality_filter(records)
    assert len(kept) == 0


def test_quality_filter_matches_v20_bare_substring_behavior_exactly():
    """_quality_filter must reproduce v20's exact (buggy) bare-substring
    matching, not this module's newer word-boundary matching - "cousin"
    contains "sin" as a bare substring and v20 would have dropped it. This
    pins down that the corpus filter is frozen to v20, not accidentally
    sharing logic with contains_unsafe_keywords()."""
    records = [
        {"reference_counter_narrative": "Ask my cousin about it.",
         "knowledge_text": "", "language": "en"},
    ]
    kept = rag_pipeline._quality_filter(records)
    assert len(kept) == 0


if __name__ == "__main__":
    test_flags_real_english_sin()
    test_english_keyword_list_is_not_applied_to_spanish()
    test_spanish_sin_is_not_flagged()
    test_italian_text_is_not_checked_using_english_keywords()
    test_basque_text_is_not_checked_using_english_keywords()
    test_tamil_text_is_not_checked_using_english_keywords()
    test_returns_none_when_no_validated_list_exists()
    try:
        test_language_parameter_is_required_not_defaulted()
    except Exception:
        pass
    test_word_boundary_not_bare_substring_even_for_english()
    test_neutral_identity_terms_are_not_flagged_in_english()
    test_en_eur_falls_back_to_base_english_list()
    test_en_in_falls_back_to_base_english_list()
    test_quality_filter_still_drops_spanish_sin_by_design()
    test_quality_filter_still_drops_genuinely_unsafe_english_records()
    test_quality_filter_matches_v20_bare_substring_behavior_exactly()
    print("test_unsafe_keywords.py: ALL PASSED")
