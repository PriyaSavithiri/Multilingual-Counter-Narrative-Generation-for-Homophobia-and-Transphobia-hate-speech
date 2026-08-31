"""
Covers scripts/build_unsafe_patterns_json.py - Stage 6's promotion step
from Stage 5's validated candidate CSVs into data/unsafe_patterns_by_lang.json.
Uses tmp_path for both the outputs dir and the patterns file so this never
touches the real (currently all not_yet_validated) shipped file.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pandas as pd

from build_unsafe_patterns_json import build_patterns_json


def _seed_patterns_file(tmp_path, languages=("en", "es")) -> Path:
    path = tmp_path / "unsafe_patterns_by_lang.json"
    data = {lang: {"status": "not_yet_validated", "terms": []} for lang in languages}
    data["_meta"] = {"last_built_at": None}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_promotes_unsafe_and_unsafe_only_in_context_decisions(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    patterns_path = _seed_patterns_file(tmp_path, languages=("es",))
    pd.DataFrame([
        {"term": "maricones", "decision": "unsafe", "category": "insult_or_slur",
         "severity": "high", "needs_human_review": False, "hate_association_score": 2.1},
        {"term": "raro", "decision": "unsafe_only_in_context", "category": "insult_or_slur",
         "severity": "medium", "needs_human_review": False, "hate_association_score": 1.0},
    ]).to_csv(outputs_dir / "unsafe_candidates_es_validated.csv", index=False)

    result = build_patterns_json(outputs_dir=outputs_dir, patterns_path=patterns_path)
    assert "es" in result["updated_buckets"]
    data = json.loads(patterns_path.read_text(encoding="utf-8"))
    assert data["es"]["status"] == "validated"
    promoted_terms = {t["term"] for t in data["es"]["terms"]}
    assert promoted_terms == {"maricones", "raro"}


def test_does_not_promote_neutral_or_unclear_or_needs_review(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    patterns_path = _seed_patterns_file(tmp_path, languages=("en",))
    pd.DataFrame([
        {"term": "gay", "decision": "neutral_identity_term", "category": "neutral_identity_term",
         "severity": "low", "needs_human_review": False, "hate_association_score": 0.1},
        {"term": "weird", "decision": "unclear", "category": "unclear",
         "severity": "low", "needs_human_review": True, "hate_association_score": 0.5},
        {"term": "kill", "decision": "unsafe", "category": "violence_or_threat",
         "severity": "high", "needs_human_review": True, "hate_association_score": 3.0},
    ]).to_csv(outputs_dir / "unsafe_candidates_en_validated.csv", index=False)

    build_patterns_json(outputs_dir=outputs_dir, patterns_path=patterns_path)
    data = json.loads(patterns_path.read_text(encoding="utf-8"))
    assert data["en"]["terms"] == []
    assert data["en"]["status"] == "not_yet_validated"


def test_bucket_with_no_validated_csv_is_left_untouched(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    patterns_path = _seed_patterns_file(tmp_path, languages=("en", "ta"))

    result = build_patterns_json(outputs_dir=outputs_dir, patterns_path=patterns_path)
    assert result["updated_buckets"] == []
    data = json.loads(patterns_path.read_text(encoding="utf-8"))
    assert data["ta"]["status"] == "not_yet_validated"
    assert data["ta"]["terms"] == []


if __name__ == "__main__":
    print("test_build_unsafe_patterns_json.py: run via pytest (uses tmp_path fixture)")
