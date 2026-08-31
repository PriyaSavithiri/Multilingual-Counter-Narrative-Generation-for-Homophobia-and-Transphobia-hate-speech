"""
Final Counter-Narrative Agent - writes the final counter-narrative +
explanation from the Judge's plan. Never sees or repeats Prosecutor content
directly (the Judge's plan is already redacted of raw Prosecutor reasoning -
final_response_plan is a synthesis, not a copy of the debate transcript).

Also guards against a real failure mode observed with weaker models on
low-resource languages (e.g. Basque): instead of genuinely refuting the
harmful claim, the model echoes the hate comment's own wording/structure
back near-verbatim - which can read as affirming the claim rather than
countering it. _echoes_source_wording() is a language-agnostic lexical
check (no per-language tokenizer needed, works identically for
en/ta/eu/es/it) that catches this and triggers one corrective retry.

Known limitation, stated plainly: this is a LEXICAL check, not a semantic
one. It reliably catches near-verbatim copying (a long matching phrase
lifted straight from the input) but cannot catch a model that paraphrases
into different words while still semantically affirming the harmful claim
rather than refuting it (e.g. restating "[group] have a disease" using
synonyms instead of negating it) - that failure mode requires actual
language understanding to detect and isn't something a string-similarity
heuristic can fix. Treat this as one safety net among several (the Judge's
own review, the prompt's explicit anti-echo instruction), not a guarantee.

Separately, this also guards against topic drift/hallucination as a last
line of defense (the Judge's own topical-relevance guard is the primary
defense - see judge_critic_agent.py) in case a hallucinated plan still
slips through: _shares_key_terms-based check that the counter-narrative
shares at least one substantive word with the actual case.

Finally, this also runs a SEMANTIC fidelity check - a genuine
understanding-based judgment (an LLM call), not a lexical heuristic. Found
via live testing: a response can share plenty of real, on-topic vocabulary
with the case (so it passes both lexical guards above) while its actual
argument quietly answers a different, adjacent claim - e.g. a transphobic
gender-identity-denial comment rebutted with generic gender-ROLE/sexism
language instead. No word-overlap check can catch that; it requires
actually reading the argument. See prompts.build_semantic_fidelity_prompt.
"""
import difflib

import prompts
import config
from counter_narrative_kg import (
    build_counter_narrative_kg, validate_kg_consistency, validate_kg_consistency_full,
    validate_kg_consistency_v2, validate_kg_consistency_v3, validate_kg_consistency_v4,
    apply_kg_safety_fallback,
)
from utils import get_logger, shares_key_terms

logger = get_logger("new_arch.final_cn_agent")

_REQUIRED_FIELDS = ["counter_narrative", "explanation"]
_FIDELITY_FIELDS = ["fidelity", "reason"]

# Two independent signals, calibrated against real observed cases - either
# one firing is treated as "echoing":
#   - RATIO: overall SequenceMatcher ratio. Catches gross, whole-text
#     duplication. Alone, this is NOT reliably discriminating: a real
#     mistral:7b-instruct-v0.3 case that verbatim-copied a 64-character
#     phrase scored 0.503, while a hand-written GENUINE negation of the
#     same claim ("it is NOT a disease...") scored 0.510 - nearly
#     identical, because a real rebuttal necessarily reuses some of the
#     claim's own vocabulary to refute it (you can't say "it's not a
#     disease" without the word "disease").
#   - LONGEST_BLOCK: length (in characters) of the single longest
#     contiguous matching substring. This is what actually separates the
#     two cases above: the verbatim-copying case had a 64-character
#     matching block; the genuine negation had only an 11-character one
#     (just the shared root word). A long contiguous match is a much more
#     specific "this phrase was lifted, not just topically related" signal
#     than an aggregate ratio.
_ECHO_RATIO_THRESHOLD = 0.6
_ECHO_LONGEST_BLOCK_THRESHOLD = 30  # characters


def _echoes_source_wording(generated: str, source: str) -> bool:
    if not generated or not source:
        return False
    matcher = difflib.SequenceMatcher(None, generated.lower(), source.lower())
    if matcher.ratio() >= _ECHO_RATIO_THRESHOLD:
        return True
    longest = max((block.size for block in matcher.get_matching_blocks()), default=0)
    return longest >= _ECHO_LONGEST_BLOCK_THRESHOLD


def _check_semantic_fidelity(client, core_claim: str, counter_narrative: str, language: str) -> dict:
    """Returns {"fidelity": "on_target"|"partially_drifted"|"off_target", "reason": "..."}.
    Degrades to "on_target" (i.e. not flagged) if core_claim is empty or the
    check itself fails to parse - a fallible semantic judge failing should
    never itself become a new source of false-positive flags on top of
    everything else that can already go wrong in one generation."""
    if not core_claim or not counter_narrative:
        return {"fidelity": "on_target", "reason": "insufficient input to judge fidelity"}
    prompt = prompts.build_semantic_fidelity_prompt(core_claim, counter_narrative, language)
    parsed = client.generate(prompt, response_schema=_FIDELITY_FIELDS, temperature=0.0, max_tokens=150)
    if not parsed or parsed.get("fidelity") not in ("on_target", "partially_drifted", "off_target"):
        return {"fidelity": "on_target", "reason": "fidelity check itself failed to parse - not flagging"}
    return parsed


class FinalCNAgent:
    def __init__(self, client):
        self.client = client

    def run(self, comment: str, judge_plan, region: str = None, case_analysis=None) -> dict:
        # case_analysis is optional and, for every style except v22kg, unused - only threaded
        # through so the v22kg Knowledge Graph (see counter_narrative_kg.py) can populate its
        # target_group node from real data instead of falling back to "unclear".
        case_analysis_dict = case_analysis.to_dict() if case_analysis else None
        prompt = prompts.build_final_cn_prompt(
            comment, judge_plan.selected_language, judge_plan.to_dict(), region or "unknown",
            case_analysis=case_analysis_dict, use_kg=config.use_kg(),
        )
        # max_tokens=900 (was 400) - same truncation bug class as
        # case_analysis_agent.py/prosecutor_agent.py/defender_agent.py: a
        # "thinking" model (verified live with Qwen3-8B via hf-transformers
        # on Colab) spends much of a smaller budget on its <think>...</think>
        # reasoning block before ever reaching the real 2-field JSON answer,
        # truncating it mid-field. 900 matches judge_critic_agent.py's
        # budget for a comparably-sized prose response.
        parsed = self.client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.3, max_tokens=900)

        if not parsed:
            # A total parse failure (as opposed to the echo-check below, which only
            # applies once parsing has already succeeded) previously gave up
            # immediately with no retry at all - found via live testing (a Judge
            # hallucination produced an internally-contradictory prompt that a 7B
            # model failed to turn into valid JSON at all). One retry, with an
            # explicit "JSON only" nudge, mirrors model_api.py's own corrective-retry
            # pattern and gives a second chance before accepting empty output.
            logger.warning("Final CN parse failure - retrying once before giving up.")
            retry_prompt = prompt + (
                "\n\nIMPORTANT: your previous reply could not be parsed as valid JSON. Reply again with "
                'ONLY a single valid JSON object with exactly the fields "counter_narrative" and '
                '"explanation" - no commentary, no markdown code fences, nothing else.'
            )
            parsed = self.client.generate(retry_prompt, response_schema=_REQUIRED_FIELDS,
                                           temperature=0.3, max_tokens=900)

        if not parsed:
            logger.warning("Final CN parse failure persisted after retry - returning empty output "
                            "(caller must handle this).")
            return {"counter_narrative": "", "explanation": "", "safety_flags": ["final_cn_parse_failure"]}

        counter_narrative = (parsed.get("counter_narrative") or "").strip()
        explanation = (parsed.get("explanation") or "").strip()
        safety_flags = []

        anchor_texts = [comment, judge_plan.core_claim_to_counter, judge_plan.final_response_plan]
        if not shares_key_terms(counter_narrative, anchor_texts):
            logger.warning(
                "Final CN appears unrelated to the actual case (no shared key terms with the "
                "comment/judge plan) - retrying once with an explicit reminder before flagging."
            )
            reinforced_prompt = prompt + (
                "\n\nIMPORTANT CORRECTION: your previous reply appeared to describe a completely "
                "different scenario, unrelated to this case. Your counter_narrative MUST respond "
                "directly to the actual comment given above - stay strictly grounded in it."
            )
            retry_parsed = self.client.generate(reinforced_prompt, response_schema=_REQUIRED_FIELDS,
                                                 temperature=0.3, max_tokens=900)
            retry_cn = (retry_parsed.get("counter_narrative") or "").strip() if retry_parsed else ""
            retry_explanation = (retry_parsed.get("explanation") or "").strip() if retry_parsed else ""
            if retry_cn and shares_key_terms(retry_cn, anchor_texts):
                counter_narrative, explanation = retry_cn, retry_explanation
            else:
                safety_flags.append("counter_narrative_appeared_off_topic")
                logger.warning("Final CN still appears off-topic after one corrective retry - flagged.")

        if _echoes_source_wording(counter_narrative, comment):
            logger.warning(
                "Final CN echoed the source comment's own wording (ratio>=%.2f or a %d+ char verbatim "
                "match) - retrying once with a corrective instruction.",
                _ECHO_RATIO_THRESHOLD, _ECHO_LONGEST_BLOCK_THRESHOLD,
            )
            corrective_prompt = prompt + (
                "\n\nIMPORTANT CORRECTION: your previous answer repeated the original hate comment's own "
                "wording and framing instead of genuinely refuting it - this could read as affirming the "
                "harmful claim rather than countering it. Rewrite the counter-narrative from scratch, "
                "using entirely different vocabulary and sentence structure, stating the OPPOSITE of the "
                "harmful claim clearly in your own words."
            )
            retry_parsed = self.client.generate(corrective_prompt, response_schema=_REQUIRED_FIELDS,
                                                 temperature=0.3, max_tokens=900)
            retry_cn = (retry_parsed.get("counter_narrative") or "").strip() if retry_parsed else ""
            retry_explanation = (retry_parsed.get("explanation") or "").strip() if retry_parsed else ""
            if retry_cn:
                counter_narrative, explanation = retry_cn, retry_explanation
            if not retry_cn or _echoes_source_wording(retry_cn, comment):
                safety_flags.append("counter_narrative_may_echo_source_wording")
                logger.warning("Final CN still echoes source wording after one corrective retry - flagged.")

        # Semantic-fidelity check - catches a subtler failure than the two lexical
        # guards above: a response that stays superficially on-topic (shares plenty
        # of real vocabulary, so it already passed both checks above) while its
        # actual argument quietly drifts to a different, adjacent claim (e.g. a
        # gender-identity-denial comment rebutted with generic gender-role/sexism
        # language). Requires an LLM call to judge, not a word-overlap heuristic.
        fidelity = _check_semantic_fidelity(
            self.client, judge_plan.core_claim_to_counter, counter_narrative, judge_plan.selected_language,
        )
        if fidelity["fidelity"] != "on_target":
            logger.warning(
                "Final CN semantic-fidelity check flagged '%s' (%s) - retrying once with the specific claim "
                "named explicitly.", fidelity["fidelity"], fidelity["reason"],
            )
            fidelity_corrective_prompt = prompt + (
                f"\n\nIMPORTANT CORRECTION: your previous answer drifted away from the specific claim it "
                f"needs to rebut. The claim is specifically: {judge_plan.core_claim_to_counter}. "
                f"Rewrite the counter-narrative so it directly and specifically rebuts THIS claim - not a "
                f"related but different point."
            )
            retry_parsed = self.client.generate(fidelity_corrective_prompt, response_schema=_REQUIRED_FIELDS,
                                                 temperature=0.3, max_tokens=900)
            retry_cn = (retry_parsed.get("counter_narrative") or "").strip() if retry_parsed else ""
            retry_explanation = (retry_parsed.get("explanation") or "").strip() if retry_parsed else ""
            retry_fidelity = _check_semantic_fidelity(
                self.client, judge_plan.core_claim_to_counter, retry_cn, judge_plan.selected_language,
            ) if retry_cn else {"fidelity": "off_target", "reason": "retry produced no counter_narrative"}
            if retry_cn:
                counter_narrative, explanation = retry_cn, retry_explanation
            if retry_fidelity["fidelity"] != "on_target":
                safety_flags.append("counter_narrative_semantic_drift")
                logger.warning(
                    "Final CN still flagged '%s' after one corrective retry - flagged, not blocked.",
                    retry_fidelity["fidelity"],
                )

        result = {"counter_narrative": counter_narrative, "explanation": explanation, "safety_flags": safety_flags}

        # KG attachment - v22kg style only. In no_kg ablation mode, the KG is removed
        # from BOTH final generation (see use_kg passed into build_final_cn_prompt above) and
        # post-generation validation/fallback. Disabled stubs keep the trace schema stable.
        if prompts.FINAL_CN_STYLE in ("v22kg", "v22kg1", "v22kg2", "v22kg3", "v22kg4", "v22kg5", "v22kg6", "v22kg7"):
            if not config.use_kg():
                result["cn_knowledge_graph"] = {"disabled": True, "reason": "ablation_no_kg"}
                result["kg_consistency"] = {"disabled": True, "risk_level": "disabled", "reason": "ablation_no_kg"}
                if prompts.FINAL_CN_STYLE == "v22kg7":
                    result["kg_safety_fallback"] = {
                        "disabled": True, "applied": False, "reason": "ablation_no_kg",
                    }
                return result

            kg = build_counter_narrative_kg(
                comment, judge_plan.selected_language, case_analysis=case_analysis_dict,
                judge_plan=judge_plan.to_dict(), region_context=region,
            )
            result["cn_knowledge_graph"] = kg
            if prompts.FINAL_CN_STYLE == "v22kg7":
                # v22kg7 only - the one behavior change beyond a validator swap: a small,
                # deterministic, no-LLM-call post-generation safety fallback (see
                # counter_narrative_kg.py's apply_kg_safety_fallback / validate_kg_consistency_v5
                # for the full reasoning). This can REPLACE counter_narrative/explanation (English
                # only, when validate_kg_consistency_v5 finds a real violation) - every other style
                # only ever attaches kg_consistency, never touches the generated text itself.
                fallback_result = apply_kg_safety_fallback(
                    kg, judge_plan.to_dict(), counter_narrative=counter_narrative, explanation=explanation,
                    comment=comment,
                )
                counter_narrative = fallback_result["counter_narrative"]
                explanation = fallback_result["explanation"]
                result["counter_narrative"] = counter_narrative
                result["explanation"] = explanation
                result["kg_safety_fallback"] = fallback_result["kg_safety_fallback"]
                result["kg_consistency"] = fallback_result["kg_safety_fallback"]["post_fallback_kg_consistency"]
            elif prompts.FINAL_CN_STYLE in ("v22kg5", "v22kg6"):
                # v22kg5 AND v22kg6 both use validate_kg_consistency_v4() unchanged - v22kg6 is a
                # tiny, prompt-only English cleanup (explicit instruction: reuse the validator if
                # sufficient, and it is - already confirmed it does not flag "avoids unsupported
                # scientific claims" or "without using science/legal/religious/country claims",
                # already fixes the Basque "dio" false positive, and still flags real unsupported
                # claims), so it deliberately reuses v22kg5's own validator rather than adding a
                # new one. validate_kg_consistency_v3 above still serves v22kg3/v22kg4 unchanged -
                # see counter_narrative_kg.py's own comment for the full reasoning.
                result["kg_consistency"] = validate_kg_consistency_v4(
                    kg, judge_plan.to_dict(), counter_narrative=counter_narrative, explanation=explanation,
                )
            elif prompts.FINAL_CN_STYLE in ("v22kg3", "v22kg4"):
                # v22kg3 AND v22kg4 both use validate_kg_consistency_v3() unchanged - v22kg4 is a
                # prompt-only Basque fluency fallback (explicit instruction: do not change KG
                # architecture, the validator, RAG, Judge, Defender, tokenizer, model, or
                # evaluation), so it deliberately reuses v22kg3's own validator rather than adding
                # a new one. validate_kg_consistency_v2 above still serves v22kg2 unchanged. See
                # counter_narrative_kg.py's own comment for validate_kg_consistency_v3's reasoning
                # (per-language trigger scanning + negation guard).
                result["kg_consistency"] = validate_kg_consistency_v3(
                    kg, judge_plan.to_dict(), counter_narrative=counter_narrative, explanation=explanation,
                )
            elif prompts.FINAL_CN_STYLE == "v22kg2":
                # v22kg2 only - validate_kg_consistency_v2() (untouched validate_kg_consistency/
                # validate_kg_consistency_full above still serve v22kg/v22kg1 unchanged) adds full
                # multilingual (en/ta/eu/es/it) evidence-conditional trigger coverage plus the
                # orientation_immutability claim family - see counter_narrative_kg.py's own
                # comment on why: the real v22kg1 Basque rerun showed a science-claim leak
                # (EU125) and an orientation-immutability leak (EU130) that validate_kg_
                # consistency_full's Tamil/English-only trigger list had no way to catch.
                result["kg_consistency"] = validate_kg_consistency_v2(
                    kg, judge_plan.to_dict(), counter_narrative=counter_narrative, explanation=explanation,
                )
            elif prompts.FINAL_CN_STYLE == "v22kg1":
                # v22kg1 only - validate_kg_consistency() above (used by v22kg, untouched) only
                # ever checked judge_plan text. v22kg1 additionally checks the actual generated
                # counter_narrative/explanation (plus Tamil code-mixing and claim_type/disease-
                # wording consistency) - see counter_narrative_kg.py's own comment on why: a
                # plan-only validator can never catch an unsupported claim that only appears in
                # the model's own output, which is exactly what happened in the real v22kg
                # Tamil rerun (V4_67/V4_64/V4_668).
                result["kg_consistency"] = validate_kg_consistency_full(
                    kg, judge_plan.to_dict(), counter_narrative=counter_narrative, explanation=explanation,
                )
            else:
                result["kg_consistency"] = validate_kg_consistency(kg, judge_plan.to_dict())

        return result
