"""
scripts/build_unsafe_patterns_json.py - Stage 6 of the v22 unsafe-keyword
redesign. Promotes Stage 5's VALIDATED candidate CSVs
(outputs/unsafe_candidates_<bucket>_validated.csv, written by
scripts/validate_unsafe_candidates.py) into the curated
data/unsafe_patterns_by_lang.json file that Stage 7's negation-aware
matcher (and anything else that wants a real per-language unsafe-term list)
reads via utils.get_validated_unsafe_patterns().

Promotion rule (deliberately conservative - see the user's Stage 6 spec):
a candidate is promoted ONLY if decision is "unsafe" or
"unsafe_only_in_context" AND needs_human_review is False. Neutral identity
terms, "unclear" decisions, and anything still flagged needs_human_review
are never promoted - those stay for a human reviewer to look at directly in
the validated CSV, not silently folded into the curated list.

A bucket with no *_validated.csv file yet (Stage 5 hasn't been run for real
for that language) is left untouched at "not_yet_validated" with an empty
terms list - this script never invents entries and never marks a bucket
"validated" just because the file happened to run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config

_PROMOTABLE_DECISIONS = {"unsafe", "unsafe_only_in_context"}
_PATTERNS_PATH = config.DATA_DIR / "unsafe_patterns_by_lang.json"
_ALL_BUCKETS = ["en", "en_EUR", "en_IN", "es", "eu", "it", "ta"]


def _promote_rows(df: pd.DataFrame) -> list:
    """Returns the subset of validated rows that clear the promotion bar,
    as plain dicts with only the fields the curated file needs."""
    if df.empty:
        return []
    keep = df["decision"].isin(_PROMOTABLE_DECISIONS) & (~df["needs_human_review"].astype(bool))
    promoted = df[keep]
    return [
        {
            "term": row["term"],
            "decision": row["decision"],
            "category": row.get("category"),
            "severity": row.get("severity"),
            "hate_association_score": row.get("hate_association_score"),
        }
        for row in promoted.to_dict("records")
    ]


def build_patterns_json(outputs_dir: Path = None, patterns_path: Path = None) -> dict:
    """Reads whatever data already exists at patterns_path (so buckets with
    no validated CSV yet keep their prior not_yet_validated state), then
    overwrites only the buckets that have a real *_validated.csv present."""
    outputs_dir = outputs_dir or config.OUTPUTS_DIR
    patterns_path = patterns_path or _PATTERNS_PATH

    with open(patterns_path, "r", encoding="utf-8") as f:
        patterns = json.load(f)

    updated_buckets = []
    for language in _ALL_BUCKETS:
        in_path = outputs_dir / f"unsafe_candidates_{language}_validated.csv"
        if not in_path.exists():
            continue
        df = pd.read_csv(in_path, encoding="utf-8")
        terms = _promote_rows(df)
        patterns[language] = {
            "status": "validated" if terms else "not_yet_validated",
            "note": (f"Promoted from {in_path.name} - {len(terms)} of {len(df)} validated "
                     f"candidates met the promotion bar (decision in "
                     f"{sorted(_PROMOTABLE_DECISIONS)}, needs_human_review=False)."
                     if terms else
                     f"{in_path.name} was validated but no candidate met the promotion bar "
                     f"(decision in {sorted(_PROMOTABLE_DECISIONS)}, needs_human_review=False) - "
                     f"see the CSV directly for human review."),
            "terms": terms,
        }
        updated_buckets.append(language)

    patterns.setdefault("_meta", {})["last_built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(patterns_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return {"updated_buckets": updated_buckets, "path": str(patterns_path)}


def main():
    parser = argparse.ArgumentParser(description="Promote Stage 5's validated candidate CSVs into "
                                                   "data/unsafe_patterns_by_lang.json.")
    args = parser.parse_args()
    result = build_patterns_json()
    if not result["updated_buckets"]:
        print(f"No outputs/unsafe_candidates_*_validated.csv files found under {config.OUTPUTS_DIR} - "
              f"run scripts/validate_unsafe_candidates.py first. {_PATTERNS_PATH} left unchanged.")
        return
    print(f"Updated buckets: {', '.join(result['updated_buckets'])} -> {result['path']}")


if __name__ == "__main__":
    main()
