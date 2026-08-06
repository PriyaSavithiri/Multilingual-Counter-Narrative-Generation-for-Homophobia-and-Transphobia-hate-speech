"""
Judge/Critic Agent - the single synthesis step. Reviews the complete debate
history (including the internal Prosecutor content, for its own review
only - a server-side reviewer, never the end-user output) and produces a
JudgePlan for the Final Counter-Narrative Agent. Does not rerun a duplicate
fact-debate pipeline - it reviews the evidence assessments the Defender
already produced each round.

Also guards against a real failure mode observed via live testing: under
the heavy accumulated context of a full Deep-Dive debate (comment + case
analysis + persona selection + all rounds' evidence), a 7B model can lose
the thread entirely and produce a plan describing a completely unrelated
scenario (observed: a customer-service/product-safety FAQ instead of a
hate-speech counter-narrative plan) rather than reasoning about the actual
case. _looks_on_topic() is a language-agnostic lexical check (no
per-language tokenizer needed) that catches this and triggers one retry
with an explicit reminder of the actual case before falling back to a safe
plan built directly from the debate itself.
"""
import prompts
from schemas import JudgePlan, EvidenceItem, RejectedContent
from utils import as_dict_item, get_logger, shares_key_terms

logger = get_logger("new_arch.judge_critic_agent")

_REQUIRED_FIELDS = [
    "selected_language", "core_claim_to_counter", "recommended_strategy",
    "approved_evidence", "rejected_content", "cultural_guidance",
    "safety_guidance", "final_response_plan",
]


def _fallback_plan(case_analysis, debate_rounds: list, reason: str) -> JudgePlan:
    """Safe plan built directly from the case/debate, bypassing the Judge's
    own (failed or off-topic) synthesis entirely."""
    last_response = debate_rounds[-1].defender_response if debate_rounds else ""
    return JudgePlan(
        selected_language=case_analysis.language,
        core_claim_to_counter=case_analysis.hidden_claim or case_analysis.target_group,
        final_response_plan=last_response or "Write a calm, respectful, evidence-cautious counter-narrative.",
        safety_guidance=[reason],
    )


def _looks_on_topic(parsed: dict, anchor_texts: list) -> bool:
    judge_text = f"{parsed.get('core_claim_to_counter') or ''} {parsed.get('final_response_plan') or ''}"
    return shares_key_terms(judge_text, anchor_texts)


class JudgeCriticAgent:
    def __init__(self, client):
        self.client = client

    def run(self, comment: str, case_analysis, persona_selection, router_decision,
            debate_rounds: list, region: str = None) -> JudgePlan:
        full_rounds = [r.to_dict(include_prosecutor=True) for r in debate_rounds]
        prompt = prompts.build_judge_prompt(
            comment, case_analysis.to_dict(), persona_selection.to_dict(),
            router_decision.to_dict(), full_rounds, region or "unknown",
        )
        parsed = self.client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.2, max_tokens=1500)

        if not parsed:
            logger.warning("Judge/Critic parse failure - falling back to the last round's Defender response.")
            return _fallback_plan(case_analysis, debate_rounds, "judge_parse_failure - using conservative fallback plan")

        # Anchor terms drawn from the actual case (never from the Judge's own
        # output) - a genuinely on-topic plan should share at least one
        # substantive word with at least one of these.
        anchor_texts = [
            case_analysis.target_group, case_analysis.hate_category, case_analysis.hidden_claim or "",
            case_analysis.rationale, comment,
        ] + [claim for r in debate_rounds for claim in r.surfaced_claims]

        if not _looks_on_topic(parsed, anchor_texts):
            logger.warning(
                "Judge output appears unrelated to the actual case (no shared key terms with the "
                "comment/target group/hate category/surfaced claims) - retrying once with an "
                "explicit reminder before falling back."
            )
            reinforced_prompt = prompt + (
                f"\n\nIMPORTANT CORRECTION: your previous reply appeared to describe a completely "
                f"different scenario, unrelated to this case. Remember: this case is specifically "
                f"about {case_analysis.hate_category} targeting {case_analysis.target_group}. Your "
                f"core_claim_to_counter and final_response_plan MUST relate directly to THIS case."
            )
            retry_parsed = self.client.generate(reinforced_prompt, response_schema=_REQUIRED_FIELDS,
                                                 temperature=0.2, max_tokens=1500)
            if retry_parsed and _looks_on_topic(retry_parsed, anchor_texts):
                parsed = retry_parsed
            else:
                logger.warning(
                    "Judge output still appears off-topic after retry - falling back to a safe plan "
                    "built directly from the debate."
                )
                return _fallback_plan(case_analysis, debate_rounds,
                                       "judge_output_appeared_off_topic_after_retry - using conservative fallback plan")

        approved_evidence = []
        for e in (parsed.get("approved_evidence") or []):
            # Some models (observed: mistral:7b-instruct-v0.3) return this list as
            # plain strings rather than structured objects - coerce defensively
            # rather than crashing with AttributeError on .get().
            e = as_dict_item(e, fallback_key="passage")
            try:
                score = float(e.get("retrieval_score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            approved_evidence.append(EvidenceItem(
                source_id=str(e.get("source_id") or ""), title=e.get("title") or "",
                passage=e.get("passage") or "", retrieval_score=score,
                source_type=e.get("source_type") or "cultural", language=e.get("language") or "",
            ))
        rejected_content = [
            RejectedContent(content=r.get("content") or "", reason=r.get("reason") or "unsupported")
            for r in (as_dict_item(item, fallback_key="content") for item in (parsed.get("rejected_content") or []))
        ]

        llm_language = parsed.get("selected_language")
        if llm_language and llm_language != case_analysis.language:
            # case_analysis.language is authoritative (already cross-checked against
            # detect_language()'s script/marker heuristics) - the Judge does not get to
            # re-decide it. A disagreement here usually means the model defaulted to
            # the prompt's own JSON-schema example rather than actually reasoning about
            # the case - log it so a regression like this is visible, but never let it
            # change behavior (see the ta/eu output-language bug this guards against).
            logger.warning(
                "Judge returned selected_language=%r but case_analysis.language=%r - "
                "overriding with the authoritative value.", llm_language, case_analysis.language,
            )

        return JudgePlan(
            selected_language=case_analysis.language,
            core_claim_to_counter=parsed.get("core_claim_to_counter") or "",
            recommended_strategy=parsed.get("recommended_strategy") or [],
            approved_evidence=approved_evidence,
            rejected_content=rejected_content,
            cultural_guidance=parsed.get("cultural_guidance") or [],
            safety_guidance=parsed.get("safety_guidance") or [],
            final_response_plan=parsed.get("final_response_plan") or "",
        )
