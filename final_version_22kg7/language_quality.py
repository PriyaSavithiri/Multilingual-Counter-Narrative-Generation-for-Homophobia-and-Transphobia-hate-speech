"""
Language-quality diagnostic layer (v21) - runs AFTER the Final Counter-
Narrative Agent produces its output, and ONLY scores/flags it. Never
rewrites, retries, or otherwise changes the counter-narrative or explanation
text - kept in its own module, called from pipeline.py, so that decision is
structural (nothing in here even has a way to return modified text) rather
than just a promise in a docstring.

Motivation: live testing on real Basque/Tamil output (Aug 2026 thesis
review) found the multi-agent debate architecture producing responses that
stayed on-topic (after the Judge anchor-language fix landed) but were not
always fluent or used invalid/garbled identity terminology - e.g. Qwen2.5-
32B producing non-words like "hormaera"/"ezkerraldi" in Basque, or garbled
attempts at LGBTQ+ terminology in Tamil ("ஹேஸ்லிஸ்", "பானிஸ்மேஸ்லிஸ்").
None of final_cn_agent.py's existing checks (topic-drift, source-echo,
semantic-fidelity) are designed to catch this - they check WHAT is being
argued, not HOW WELL the language itself is constructed. This module adds
that missing check as a diagnostic only; whether to act on it (rewrite/
retry) is a separate, later decision, not made here.
"""
import prompts
from utils import get_logger

logger = get_logger("new_arch.language_quality")

_REQUIRED_FIELDS = ["target_language_match", "fluency_score", "terminology_score",
                     "clarity_score", "quality_issues", "problem_spans", "confidence_score"]

# A response is flagged for review if ANY of the following hold:
#   - target_language_match is False (wrong language entirely), or
#   - fluency_score, terminology_score, or clarity_score individually is
#     below this threshold (catches a single bad dimension even if the other
#     two are perfect - e.g. fluency=1, terminology=2, clarity=2 averages to
#     1.67, which would NOT trip an average-only check but should still be
#     flagged), or
#   - the overall (averaged) language_quality_score is below this threshold, or
#   - confidence_score is 0 (the judge itself says it isn't confident enough
#     in this language to trust its own scores - see build_language_quality_prompt,
#     which already instructs it not to give a 2 in that case; this is the
#     belt-and-braces check on top of that instruction, not a replacement for it).
# The third condition is mathematically implied by the second (if no
# individual score is below the threshold, their average can't be either),
# but is kept explicit rather than relied upon as an inference, matching
# exactly how this is meant to read: separate reasons to flag, not one
# clever one. 1.5 is deliberately sensitive - flagging costs nothing at this
# diagnostic-only stage (no rewrite happens), and under-flagging would defeat
# the point of adding this check in the first place.
_FLAG_THRESHOLD = 1.5


def _parse_bool(value) -> bool:
    """Safe boolean coercion for LLM JSON output. Plain bool(value) is a real
    bug here: Python's bool("false") is True (any non-empty string is
    truthy), and models sometimes emit "true"/"false" as quoted JSON strings
    rather than real JSON booleans - a naive bool() cast would then silently
    treat an explicit "this is the WRONG language" answer as a match.
    Unrecognized values default to False (not matching) rather than True,
    since this check exists specifically to catch language mismatches -
    failing toward "flag it for review" is the safe direction here, unlike
    the fail-open default used elsewhere in this module for total parse
    failures (a different, unambiguous kind of failure)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def _result(language_quality_score=None, fluency_score=None, terminology_score=None,
            clarity_score=None, target_language_match=None, quality_flag="unscored",
            quality_issues=None, needs_rewrite=False, problem_spans=None, confidence_score=None) -> dict:
    return {
        "language_quality_score": language_quality_score,
        "fluency_score": fluency_score,
        "terminology_score": terminology_score,
        "clarity_score": clarity_score,
        "target_language_match": target_language_match,
        "quality_flag": quality_flag,
        "quality_issues": quality_issues or [],
        "needs_rewrite": needs_rewrite,
        "problem_spans": problem_spans or [],
        "confidence_score": confidence_score,
    }


def evaluate_language_quality(client, counter_narrative: str, language: str, comment: str,
                               core_claim: str = "") -> dict:
    """Diagnostic only - scores an already-generated counter_narrative, never
    modifies or returns modified text. Returns a dict with exactly these
    keys, ready to merge into a trace/CSV row: language_quality_score,
    fluency_score, terminology_score, clarity_score, target_language_match,
    quality_flag, quality_issues, needs_rewrite, problem_spans, confidence_score.

    Fails open (quality_flag="unscored", needs_rewrite=False) on any parse
    failure or unexpected error - a fallible diagnostic failing should never
    itself become a new source of false-positive flags, matching the same
    philosophy as final_cn_agent.py's own _check_semantic_fidelity fallback."""
    if not counter_narrative:
        return _result(quality_flag="unscored", quality_issues=["empty counter_narrative - nothing to score"])

    try:
        prompt = prompts.build_language_quality_prompt(counter_narrative, language, comment, core_claim)
        parsed = client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.0, max_tokens=400)
    except Exception as exc:
        logger.warning("Language-quality diagnostic raised an exception for language=%s (%s) - leaving unscored.",
                        language, exc)
        return _result(quality_flag="unscored", quality_issues=[f"language_quality_check_raised: {exc}"])

    if not parsed:
        logger.warning("Language-quality diagnostic parse failure for language=%s - leaving unscored.", language)
        return _result(quality_flag="unscored", quality_issues=["language_quality_parse_failure"])

    try:
        fluency = max(0.0, min(2.0, float(parsed.get("fluency_score"))))
        terminology = max(0.0, min(2.0, float(parsed.get("terminology_score"))))
        clarity = max(0.0, min(2.0, float(parsed.get("clarity_score"))))
    except (TypeError, ValueError):
        logger.warning("Language-quality diagnostic returned non-numeric scores for language=%s - leaving unscored.",
                        language)
        return _result(quality_flag="unscored", quality_issues=["language_quality_invalid_scores"])

    overall = round((fluency + terminology + clarity) / 3, 2)
    target_language_match = _parse_bool(parsed.get("target_language_match"))
    quality_issues = parsed.get("quality_issues") or []
    if not isinstance(quality_issues, list):
        quality_issues = [str(quality_issues)]
    problem_spans = parsed.get("problem_spans") or []
    if not isinstance(problem_spans, list):
        problem_spans = [str(problem_spans)]
    try:
        confidence_score = max(0.0, min(2.0, float(parsed.get("confidence_score"))))
    except (TypeError, ValueError):
        # Missing/invalid confidence_score is treated as "not confident" (0),
        # not silently ignored - the whole point of this field is to catch
        # exactly this kind of judge unreliability.
        confidence_score = 0.0

    flagged = (
        not target_language_match
        or fluency < _FLAG_THRESHOLD
        or terminology < _FLAG_THRESHOLD
        or clarity < _FLAG_THRESHOLD
        or overall < _FLAG_THRESHOLD
        or confidence_score == 0
    )
    return _result(
        language_quality_score=overall, fluency_score=fluency, terminology_score=terminology,
        clarity_score=clarity, target_language_match=target_language_match,
        quality_flag="needs_review" if flagged else "ok",
        quality_issues=quality_issues, needs_rewrite=flagged,
        problem_spans=problem_spans, confidence_score=confidence_score,
    )
