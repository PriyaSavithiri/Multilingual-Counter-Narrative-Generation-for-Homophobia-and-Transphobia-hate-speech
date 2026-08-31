"""
scripts/extract_unsafe_candidates.py - Stage 3 of the v22 unsafe-keyword
redesign. Extracts CANDIDATE unsafe terms/phrases per language from the
project's own datasets, using a simple hate-vs-reference association score.
Outputs candidates only - nothing here is a validated unsafe keyword. See
scripts/validate_unsafe_candidates.py (Stage 5) for the next step, and
data/unsafe_patterns_by_lang.json (Stage 6) for where validated patterns
eventually live. This script does not touch generation, RAG, or any
pipeline agent - it is a standalone, offline analysis tool.

Method
------
Source data: dataset_loader.iter_corpus_records() - the same TRAIN-split,
hate_speech / reference_counter_narrative / language / dataset_name / region
records already used to build the RAG corpus (filter_target=True by
default, restricting ml_mtconan_kn's 4 languages to homophobia/transphobia/
LGBT+-relevant rows, matching this thesis's actual topic scope). TEST is
never touched here - iter_corpus_records() only ever reads "train" splits,
consistent with this project's existing test-split discipline (see
dataset_loader.py's own module docstring); TRAIN is plenty of data for
word-frequency-based candidate extraction.

English region split: each record already carries a "region" field
(config.DATASET_CULTURAL_REGION, resolved in dataset_loader._normalize_record) -
"European" for ml_mtconan_kn, "Indian" for english_codabench/tamil_codabench.
English rows are split into en_EUR and en_IN using this EXISTING metadata
(nothing invented for this script), in ADDITION to a combined "en" bucket
covering both regions together. es/it/eu/ta have no such regional metadata
and are not split further - each stays one bucket.

Tokenization: groups Letter + Number + combining-Mark Unicode categories
into one token (see _tokenize) - NOT a plain `\\w+` regex, which does not
cover Unicode category Mark and so splits Tamil syllables mid-character
(a base consonant plus its combining vowel sign/virama). This is still not
a real per-language tokenizer - agglutinative morphology (Tamil, to a lesser
extent Basque) means a "word" here is still a whitespace-delimited unit, not
a linguistically meaningful root/stem (same caveat already given for
ROUGE-L in evaluation.py / docs/07_EVALUATION_STRATEGY.md). It's a coarse
approximation, good enough to surface CANDIDATES for human/LLM review in
Stage 5, not a claim of linguistic correctness. Nothing here is treated as
a final answer.

hate_count/reference_count are DOCUMENT frequencies (number of distinct
rows containing the term at least once), not raw token counts - avoids one
repetitive row dominating a candidate's score with internal repetition.

hate_association_score = log((hate_count + 1) / (reference_count + 1)),
exactly as specified. Higher means more hate-associated, NOT proof of being
unsafe - a term can be hate-associated because it's a common topic word
(e.g. "orientation") rather than because it's inherently harmful. Deciding
that is Stage 5's job (LLM/human validation), not this script's.

Output: outputs/unsafe_candidates_<bucket>.csv, columns: language, term,
ngram_size, hate_count, reference_count, hate_association_score,
example_hate_snippets (up to 3, "||"-joined).
"""
import argparse
import math
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
import dataset_loader

_MAX_EXAMPLES = 3
_MIN_HATE_COUNT = 1  # a candidate must appear in at least one hate_text row to be considered at all


def _tokenize(text: str) -> list:
    """Groups Letter + Number + combining-Mark characters into one token,
    splitting only on everything else (whitespace/punctuation) - NOT a plain
    `\\w+` regex. `\\w` does not include Unicode category Mark (Mn/Mc/Me),
    which is exactly what Tamil vowel signs and virama are: a Tamil
    syllable is a base consonant (category Lo) followed by one or more
    combining marks that attach to it, not separate characters. A bare
    `\\w+` regex splits mid-syllable there, producing meaningless fragments
    (confirmed via live testing: 'render' as an English gloss - Tamil verb-
    ending fragments like isolated vowel signs showed up as their own
    "candidates" before this fix). Latin-script languages with precomposed
    accented letters (Spanish "á" etc.) were never affected by this, since
    NFC-normalized precomposed letters are already single Letter-category
    codepoints - this fix is Tamil-specific in practice, general in
    implementation (works the same way for any script using combining marks).

    NFC-normalizes first: the same visible character can be represented as
    either one precomposed codepoint or a base+combining-mark sequence
    depending on how the source text was encoded - without normalizing,
    two occurrences of what looks like the identical word could silently
    become different token strings and get counted as separate candidates."""
    text = unicodedata.normalize("NFC", (text or "").lower())
    tokens = []
    current = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in ("L", "N", "M"):  # Letter, Number, or combining Mark
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _ngrams_in_document(tokens: list) -> set:
    """Unigrams + bigrams + trigrams present in ONE document, as a SET (not
    a multiset) - callers are computing document frequency, so a term
    appearing twice in the same document must only count once for that
    document's contribution to hate_count/reference_count."""
    terms = set()
    for n in (1, 2, 3):
        if len(tokens) < n:
            continue
        terms |= {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    return terms


def _resolve_language_bucket(record: dict) -> str:
    """English splits into en_EUR/en_IN using the record's own pre-existing
    "region" field (dataset_loader._normalize_record, sourced from
    config.DATASET_CULTURAL_REGION) - not re-derived or invented here.
    es/it/eu/ta have no such regional metadata and are returned as-is."""
    language = record.get("language")
    if language != "en":
        return language
    region = record.get("region")
    if region == "European":
        return "en_EUR"
    if region == "Indian":
        return "en_IN"
    return "en"


def extract_candidates(filter_target: bool = True) -> dict:
    """Returns {bucket: DataFrame}. "en" (combined, both regions) is always
    populated alongside en_EUR/en_IN when regional metadata is available, so
    both the coarse and fine view exist side by side."""
    records = dataset_loader.iter_corpus_records(filter_target=filter_target)

    hate_texts_by_bucket = defaultdict(list)
    ref_texts_by_bucket = defaultdict(list)
    for r in records:
        hate_text = r.get("hate_speech") or ""
        ref_text = r.get("reference_counter_narrative") or ""
        bucket = _resolve_language_bucket(r)
        hate_texts_by_bucket[bucket].append(hate_text)
        ref_texts_by_bucket[bucket].append(ref_text)
        if bucket in ("en_EUR", "en_IN"):
            hate_texts_by_bucket["en"].append(hate_text)
            ref_texts_by_bucket["en"].append(ref_text)

    return {
        bucket: _build_candidate_table(bucket, hate_texts_by_bucket[bucket], ref_texts_by_bucket[bucket])
        for bucket in hate_texts_by_bucket
    }


def _build_candidate_table(language: str, hate_texts: list, ref_texts: list) -> pd.DataFrame:
    hate_count = defaultdict(int)
    ref_count = defaultdict(int)
    examples = defaultdict(list)

    for text in hate_texts:
        for term in _ngrams_in_document(_tokenize(text)):
            hate_count[term] += 1
            if len(examples[term]) < _MAX_EXAMPLES:
                examples[term].append(text)

    for text in ref_texts:
        for term in _ngrams_in_document(_tokenize(text)):
            ref_count[term] += 1

    rows = []
    for term, h_count in hate_count.items():
        if h_count < _MIN_HATE_COUNT:
            continue
        r_count = ref_count.get(term, 0)
        score = math.log((h_count + 1) / (r_count + 1))
        rows.append({
            "language": language,
            "term": term,
            "ngram_size": len(term.split()),
            "hate_count": h_count,
            "reference_count": r_count,
            "hate_association_score": round(score, 4),
            "example_hate_snippets": " || ".join(examples[term]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("hate_association_score", ascending=False).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract candidate unsafe terms/phrases per language "
                                                   "(candidates only - not validated unsafe keywords).")
    parser.add_argument("--no-filter-target", dest="filter_target", action="store_false", default=True,
                         help="Include all hate targets, not just homophobia/transphobia/LGBT+-relevant "
                              "rows (default: filtered, matching the RAG corpus build's own default).")
    args = parser.parse_args()

    results = extract_candidates(filter_target=args.filter_target)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for bucket, df in sorted(results.items()):
        out_path = config.OUTPUTS_DIR / f"unsafe_candidates_{bucket}.csv"
        df.to_csv(out_path, index=False)
        print(f"{bucket}: {len(df)} candidates -> {out_path}")


if __name__ == "__main__":
    main()
