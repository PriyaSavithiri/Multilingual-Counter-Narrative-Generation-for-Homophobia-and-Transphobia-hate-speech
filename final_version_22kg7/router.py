"""
Deterministic Router - no LLM call. Uses the Case Analysis result's
hate_type field only. Explicit -> Fast-Track (1 round). Implicit ->
Deep-Dive (K rounds). Exactly one track runs per input; the personas
themselves (generated once, before this call) do not determine the track.
"""
from schemas import RouterDecision, CaseAnalysis


def route(case_analysis: CaseAnalysis) -> RouterDecision:
    if case_analysis.hate_type == "explicit":
        return RouterDecision(
            selected_track="fast_track",
            confidence=case_analysis.confidence,
            rationale="Case Analysis classified this comment as explicit hate.",
        )
    return RouterDecision(
        selected_track="deep_dive",
        confidence=case_analysis.confidence,
        rationale="Case Analysis classified this comment as implicit hate (or classification was uncertain, "
                   "which conservatively routes to the more thorough Deep-Dive track).",
    )
