"""
scripts/validate_unsafe_candidates.py - Stage 5 of the v22 unsafe-keyword
redesign. Takes Stage 4's FILTERED candidate CSVs
(outputs/unsafe_candidates_<bucket>_filtered.csv) and asks an LLM to
suggest a decision/category/severity for each. This is a SUGGESTION for
human review, never final truth - see prompts.build_unsafe_term_validation_prompt.
Writes outputs/unsafe_candidates_<bucket>_validated.csv, one row per
candidate, adding: decision, category, severity, notes, needs_human_review
(plus the original candidate columns, kept for traceability so a reviewer
doesn't need to cross-reference back to the filtered CSV).

Uses an independent model by default (gpt-4.1-mini via the openai backend) -
same reasoning as language_quality.py's evaluation-time scoring: a model
validating dataset-derived candidates has no reason to be tied to whichever
model generated the counter-narratives evaluated elsewhere in this project.

LLM validation costs real API calls. --limit caps how many candidates get
validated per language (default: top 50 by hate_association_score, the
highest-priority ones to look at first) - NOT unlimited by default. Pass
--limit 0 to validate every filtered candidate, which can mean thousands of
calls for some languages (Tamil had 1,338 filtered candidates as of this
writing).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
import prompts
from model_api import get_client

_REQUIRED_FIELDS = ["decision", "category", "severity", "notes", "needs_human_review"]
_VALID_DECISIONS = {"unsafe", "unsafe_only_in_context", "neutral_identity_term", "neutral/general", "unclear"}
_VALID_CATEGORIES = {"violence_or_threat", "dehumanisation", "disease_or_pathology_framing", "criminalisation",
                      "sexual_predation_or_grooming_framing", "exclusion_or_silencing", "insult_or_slur",
                      "neutral_identity_term", "unclear"}
_VALID_SEVERITIES = {"low", "medium", "high"}


def _fallback_result(note: str) -> dict:
    """Fails toward maximum caution on any parse failure - "unclear" +
    needs_human_review=True - never silently defaults to "safe" (which
    would mean nothing gets flagged) or "unsafe" (which would mean a
    harmless term gets flagged) on a call that didn't actually succeed."""
    return {"decision": "unclear", "category": "unclear", "severity": "low",
            "notes": note, "needs_human_review": True}


def validate_candidate(client, term: str, language: str, example_snippets: list,
                        hate_association_score: float = None) -> dict:
    """Single-candidate validation - returns exactly the 5 fields the
    output CSV needs. Never modifies term/language; those are passed
    through unchanged by the caller, not returned from here."""
    prompt = prompts.build_unsafe_term_validation_prompt(term, language, example_snippets, hate_association_score)
    parsed = client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.0, max_tokens=300)
    if not parsed:
        return _fallback_result("validation call failed to parse")

    decision = parsed.get("decision") if parsed.get("decision") in _VALID_DECISIONS else "unclear"
    category = parsed.get("category") if parsed.get("category") in _VALID_CATEGORIES else "unclear"
    severity = parsed.get("severity") if parsed.get("severity") in _VALID_SEVERITIES else "low"
    # A model saying decision="unclear" but forgetting to also set
    # needs_human_review=true is treated as needing review anyway - the two
    # fields should never disagree, and "unclear" without a review flag
    # would defeat the point of having the flag at all.
    needs_review = bool(parsed.get("needs_human_review")) or decision == "unclear"
    return {"decision": decision, "category": category, "severity": severity,
            "notes": parsed.get("notes") or "", "needs_human_review": needs_review}


def validate_dataframe(client, df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """df is one language's FILTERED candidate table (Stage 4's output
    shape: language, term, ngram_size, hate_count, reference_count,
    hate_association_score, example_hate_snippets). Returns the same rows,
    highest-scoring first when limited, with the 5 validation fields added."""
    if df.empty:
        return df.copy()
    if limit and limit > 0:
        df = df.sort_values("hate_association_score", ascending=False).head(limit).reset_index(drop=True)

    rows = []
    for row in df.to_dict("records"):
        examples = [s for s in str(row.get("example_hate_snippets") or "").split(" || ") if s]
        result = validate_candidate(client, row["term"], row["language"], examples,
                                     row.get("hate_association_score"))
        rows.append({**row, **result})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 4's filtered candidates via an LLM - "
                                                   "produces a suggestion for human review, not final truth.")
    parser.add_argument("--limit", type=int, default=50,
                         help="Validate only the top N filtered candidates per language, by "
                              "hate_association_score (default 50). Pass 0 for no limit - can mean "
                              "thousands of API calls for some languages.")
    parser.add_argument("--backend", default="openai",
                         help="Defaults to an independent model (openai/gpt-4.1-mini), not tied to "
                              "whichever model generated the counter-narratives evaluated elsewhere "
                              "in this project.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    input_files = sorted(config.OUTPUTS_DIR.glob("unsafe_candidates_*_filtered.csv"))
    if not input_files:
        print(f"No unsafe_candidates_*_filtered.csv files found under {config.OUTPUTS_DIR} - "
              f"run scripts/extract_unsafe_candidates.py then scripts/filter_unsafe_candidates.py first.")
        return

    client = get_client(backend=args.backend, model=args.model)
    for in_path in input_files:
        language = in_path.stem.replace("unsafe_candidates_", "").replace("_filtered", "")
        df = pd.read_csv(in_path, encoding="utf-8")
        if df.empty:
            print(f"{language}: 0 filtered candidates, skipping")
            continue
        validated = validate_dataframe(client, df, args.limit)
        out_path = config.OUTPUTS_DIR / f"unsafe_candidates_{language}_validated.csv"
        validated.to_csv(out_path, index=False)
        unsafe_count = (validated["decision"].isin(["unsafe", "unsafe_only_in_context"])).sum()
        review_count = validated["needs_human_review"].sum()
        print(f"{language}: {len(validated)} validated ({unsafe_count} unsafe/context-unsafe, "
              f"{review_count} flagged needs_human_review) -> {out_path}")


if __name__ == "__main__":
    main()
