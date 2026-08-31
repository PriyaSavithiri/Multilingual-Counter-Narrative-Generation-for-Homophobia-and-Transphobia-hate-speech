"""
Case Analysis Agent - one structured LLM call producing a CaseAnalysis
(language, target, hate category, intent, explicit/implicit classification
that deterministically drives the Router, hidden claim, evidence topics, and
a cautious region SUGGESTION - never a confirmed region).
"""
import config
import prompts
from schemas import CaseAnalysis, RegionSuggestion
from utils import detect_language, get_logger

logger = get_logger("new_arch.case_analysis_agent")

_REQUIRED_FIELDS = [
    "language", "target_group", "hate_category", "intent", "strategy_hint",
    "hate_type", "confidence", "hidden_claim", "evidence_topics",
    "cultural_context_needed", "region_suggestion", "safety_notes", "rationale",
]


class CaseAnalysisAgent:
    def __init__(self, client):
        self.client = client

    def run(self, comment: str, language_hint: str = None, region_hint: str = None) -> CaseAnalysis:
        prompt = prompts.build_case_analysis_prompt(comment, language_hint, region_hint)
        # max_tokens=1200 (was 600) - same truncation bug class as
        # prosecutor_agent.py/defender_agent.py: a "thinking" model (verified
        # live with Qwen3-8B via hf-transformers on Colab) spends most/all of
        # a smaller budget on its <think>...</think> reasoning block before
        # ever reaching the real JSON answer, truncating it mid-field. 1200
        # gives enough headroom for both the reasoning and this schema's 12
        # fields to complete.
        parsed = self.client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.2, max_tokens=1200)

        if not parsed:
            logger.warning("Case Analysis parse failure - using conservative fallback.")
            return CaseAnalysis(
                language=language_hint or detect_language(comment),
                hate_type="implicit",  # safer default: routes to the more thorough Deep-Dive track
                confidence=0.0,
                rationale="LLM output failed to parse; using a conservative fallback.",
                safety_notes=["case_analysis_parse_failure"],
            )

        language = parsed.get("language")
        if language not in config.SUPPORTED_LANGUAGES:
            language = detect_language(comment, language_hint)

        region_raw = parsed.get("region_suggestion") or {}
        try:
            confidence = float(region_raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        region_suggestion = RegionSuggestion(
            region=region_raw.get("region"), country=region_raw.get("country"),
            confidence=confidence, rationale=region_raw.get("rationale", ""),
        )
        if not region_suggestion.is_confident:
            # Below-threshold or unsupported suggestions are downgraded to
            # Unknown - never silently treated as a confirmed region.
            region_suggestion = RegionSuggestion(
                region=None, country=None, confidence=region_suggestion.confidence,
                rationale=region_suggestion.rationale or "below confidence threshold",
            )

        hate_type = parsed.get("hate_type")
        if hate_type not in ("explicit", "implicit"):
            hate_type = "implicit"

        try:
            overall_confidence = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            overall_confidence = 0.0

        return CaseAnalysis(
            language=language,
            target_group=parsed.get("target_group") or "unclear",
            hate_category=parsed.get("hate_category") or "unclear",
            intent=parsed.get("intent") or [],
            strategy_hint=parsed.get("strategy_hint") or [],
            hate_type=hate_type,
            confidence=overall_confidence,
            hidden_claim=parsed.get("hidden_claim"),
            evidence_topics=parsed.get("evidence_topics") or [],
            cultural_context_needed=bool(parsed.get("cultural_context_needed")),
            region_suggestion=region_suggestion,
            safety_notes=parsed.get("safety_notes") or [],
            rationale=parsed.get("rationale") or "",
        )
