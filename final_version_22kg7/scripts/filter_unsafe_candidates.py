"""
scripts/filter_unsafe_candidates.py - Stage 4 of the v22 unsafe-keyword
redesign. Reads the CANDIDATE tables Stage 3 (extract_unsafe_candidates.py)
wrote to outputs/unsafe_candidates_<bucket>.csv, and removes obvious
non-useful candidates - NOT a validation step. This still produces a
candidate list for Stage 5's LLM/human validation, not final truth. The
original, unfiltered Stage 3 CSVs are never modified - this writes separate
outputs/unsafe_candidates_<bucket>_filtered.csv files, so nothing is lost.

Deliberately conservative, per the spec: "Do not remove candidates too
aggressively." Every filter below only removes a candidate when there's a
clear, mechanical reason to (too short, pure punctuation/numbers, a known
stopword, a known neutral identity term, appears about as often in safe
reference text as in hate text, or too few occurrences to be statistically
meaningful) - never a judgment call about whether a term is "probably fine."
Judgment calls are Stage 5's job.

Filters applied:
  - very short unigrams (< min_term_length characters)
  - terms with no letter/number characters at all (punctuation-only)
  - pure-number unigrams
  - stopword unigrams, ONLY for languages with a validated stopword source
    (stopwordsiso covers en/es/it/eu; NOT ta - no safe Tamil stopword list
    is used here, since inventing one would be worse than having none)
  - known neutral identity terms (any n-gram size) - see NEUTRAL_IDENTITY_TERMS
    below; Basque and Tamil entries are flagged as LOWER CONFIDENCE and
    should be checked by a native/fluent speaker before being fully trusted,
    same caveat already applied to this project's Basque ground-truth data
  - terms appearing almost as often in reference_counter_narrative as in
    hate_text (not hate-specific, just a common shared topic word)
  - terms with too few hate_text occurrences (default: fewer than 2)
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config

# ---------------------------------------------------------------------------
# Neutral identity terms - never treated as unsafe by themselves, regardless
# of hate_association_score. Deliberately a SMALL, defensible list, not an
# attempt at an exhaustive multilingual LGBTQ+ terminology taxonomy - Stage
# 5's LLM/human validation is the backstop for anything this list misses
# (a genuinely neutral term not listed here should still get decision=
# "neutral_identity_term" from that step). English/Spanish/Italian entries
# are standard, common terms. Basque and Tamil entries are LOWER CONFIDENCE -
# flagged here for native/fluent-speaker verification before being fully
# relied on, the same caveat already applied elsewhere in this project to
# Basque ground-truth text.
# ---------------------------------------------------------------------------
NEUTRAL_IDENTITY_TERMS = {
    "en": {"gay", "lesbian", "trans", "transgender", "bisexual", "queer", "homosexual",
           "lgbtq", "lgbtq+", "lgbt", "lgbt+"},
    "es": {"gay", "lesbiana", "trans", "transgenero", "transgénero", "bisexual", "queer",
           "homosexual", "lgbtq", "lgbtq+", "lgbt", "lgbt+"},
    "it": {"gay", "lesbica", "trans", "transgender", "bisessuale", "queer", "omosessuale",
           "lgbtq", "lgbtq+", "lgbt", "lgbt+"},
    # LOWER CONFIDENCE - verify with a Basque speaker before relying on this fully.
    "eu": {"gay", "gaya", "lesbiana", "trans", "transexuala", "bisexuala", "queer",
           "homosexuala", "lgbtq", "lgbtq+", "lgbt", "lgbt+"},
    # LOWER CONFIDENCE - verify with a Tamil speaker before relying on this fully.
    # திருநங்கை (transgender person) is the one term directly confirmed via live
    # testing (Stage 3 output) as a neutral identity term that was otherwise
    # showing up as "hate-associated" purely due to corpus frequency imbalance.
    "ta": {"திருநங்கை", "ஓரினச்சேர்க்கையாளர்", "இருபாலின"},
}

_BASE_LANGUAGE = {"en_EUR": "en", "en_IN": "en"}


def _base_language(language: str) -> str:
    return _BASE_LANGUAGE.get(language, language)


def _stopwords_for_language(language: str) -> set:
    """Only returns a non-empty set for languages with an actual validated
    stopword source (stopwordsiso). Returns an empty set - not a guess -
    for anything else, most notably Tamil, which stopwordsiso does not cover."""
    base = _base_language(language)
    try:
        import stopwordsiso
    except ImportError:
        return set()
    if not stopwordsiso.has_lang(base):
        return set()
    return set(stopwordsiso.stopwords(base))


_PURE_NUMBER_RE = re.compile(r"^\d+$")
_HAS_LETTER_OR_NUMBER_RE = re.compile(r"[^\W\d_]|\d", re.UNICODE)


def filter_candidates(df: pd.DataFrame, language: str, min_term_length: int = 2,
                       min_hate_count: int = 2, balance_ratio_threshold: float = 0.5) -> pd.DataFrame:
    """Returns the subset of df that survives all filters. Does not mutate df."""
    if df.empty:
        return df.copy()

    terms = df["term"].astype(str)
    is_unigram = df["ngram_size"] == 1
    keep = pd.Series(True, index=df.index)

    # Punctuation-only (no letter or digit anywhere) - applies to any n-gram size.
    keep &= terms.apply(lambda t: bool(_HAS_LETTER_OR_NUMBER_RE.search(t)))

    # Very short unigrams, and pure-number unigrams - length/number-ness of a
    # single token is meaningful; a multi-word bigram/trigram's raw character
    # count isn't a useful signal the same way, so these two only apply to
    # ngram_size == 1.
    keep &= ~(is_unigram & (terms.str.len() < min_term_length))
    keep &= ~(is_unigram & terms.apply(lambda t: bool(_PURE_NUMBER_RE.match(t))))

    # Stopwords - unigrams only, only where a validated list exists for this language.
    stopwords = _stopwords_for_language(language)
    if stopwords:
        keep &= ~(is_unigram & terms.isin(stopwords))

    # Neutral identity terms - matches a candidate if it EITHER exactly
    # equals one neutral term OR is made up ENTIRELY of neutral terms
    # (e.g. "gay lesbian" as a bigram - two neutral labels named together
    # isn't inherently unsafe just because neither individual word matched
    # the whole bigram string). Always includes the English list too,
    # regardless of bucket - English LGBTQ+ terms ("gay", "lesbian", "trans",
    # ...) commonly appear as loanwords/code-switches inside non-English
    # text (confirmed via live testing: "gay" and "lesbian" showing up as
    # Tamil candidates, spelled in Latin script, inside otherwise-Tamil hate
    # comments), so checking only the bucket's own-script list would miss them.
    neutral_terms = NEUTRAL_IDENTITY_TERMS.get(_base_language(language), set()) | NEUTRAL_IDENTITY_TERMS["en"]
    if neutral_terms:
        is_all_neutral_words = terms.apply(lambda t: bool(t.split()) and all(w in neutral_terms for w in t.split()))
        keep &= ~is_all_neutral_words

    # Appears about as often in safe reference text as in hate text - not
    # hate-specific, just a common shared topic word. Only applies when
    # reference_count > 0 (a term with reference_count == 0 is, by
    # definition, not balanced - it never appears in safe text at all).
    keep &= ~((df["reference_count"] > 0) & (df["reference_count"] >= df["hate_count"] * balance_ratio_threshold))

    # Too few hate_text occurrences to be statistically meaningful.
    keep &= df["hate_count"] >= min_hate_count

    return df[keep].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Filter Stage 3's candidate tables - removes obvious "
                                                   "non-useful candidates, produces input for Stage 5 "
                                                   "validation, not final truth.")
    parser.add_argument("--min-term-length", type=int, default=2)
    parser.add_argument("--min-hate-count", type=int, default=2)
    parser.add_argument("--balance-ratio-threshold", type=float, default=0.5)
    args = parser.parse_args()

    input_files = sorted(config.OUTPUTS_DIR.glob("unsafe_candidates_*.csv"))
    input_files = [f for f in input_files if not f.stem.endswith("_filtered")]
    if not input_files:
        print(f"No unsafe_candidates_*.csv files found under {config.OUTPUTS_DIR} - "
              f"run scripts/extract_unsafe_candidates.py first.")
        return

    for in_path in input_files:
        language = in_path.stem.replace("unsafe_candidates_", "")
        df = pd.read_csv(in_path, encoding="utf-8")
        filtered = filter_candidates(df, language, args.min_term_length, args.min_hate_count,
                                      args.balance_ratio_threshold)
        out_path = config.OUTPUTS_DIR / f"unsafe_candidates_{language}_filtered.csv"
        filtered.to_csv(out_path, index=False)
        print(f"{language}: {len(df)} -> {len(filtered)} candidates after filtering -> {out_path}")


if __name__ == "__main__":
    main()
