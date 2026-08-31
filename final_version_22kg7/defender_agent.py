"""
Defender Agent - open-ended persona (same one across all rounds of a
track). Evidence verification is folded into every turn here: per-round,
each Prosecutor-surfaced claim is checked against bilingual RAG retrieval
and assessed SUPPORTED / REFUTED / NEI in the same LLM call that produces
the rebuttal - there is no separate post-hoc fact-checking pass.
"""
import time

import config
import evidence_verifier
import prompts
from schemas import DebateRound, EvidenceAssessment
from utils import as_dict_item, get_logger

logger = get_logger("new_arch.defender_agent")

_REQUIRED_FIELDS = ["defender_response", "claim_assessments", "new_information",
                    "unresolved_questions", "cultural_notes", "safety_notes"]
_VALID_VERDICTS = ("SUPPORTED", "REFUTED", "NEI")


class DefenderAgent:
    def __init__(self, client):
        self.client = client

    def run_round(self, comment: str, case_analysis, defender_persona: dict, prosecutor_argument: str,
                   surfaced_claims: list, round_num: int, rag_mode: str, region: str,
                   filter_target: bool, query_strategy: str = None) -> DebateRound:
        # Query formulation for all of this round's claims is batched into a
        # single model call (gather_evidence_for_claims), instead of one
        # sequential call per claim - the claims are independent of each
        # other, so there was nothing gained by formulating their queries
        # one at a time. config.BATCH_QUERY_FORMULATION=false forces the old
        # per-claim path (same timing wrapper either way) purely to get an
        # apples-to-apples before/after comparison on the same code version.
        evidence_by_claim = {}
        all_query_texts = []
        blocks = []
        _t0 = time.time()
        if config.BATCH_QUERY_FORMULATION:
            claim_results = evidence_verifier.gather_evidence_for_claims(
                self.client, surfaced_claims or [], case_analysis.language, region, rag_mode, filter_target, query_strategy
            )
        else:
            claim_results = {}
            for claim in (surfaced_claims or []):
                claim_results[claim] = evidence_verifier.gather_evidence_for_claim(
                    self.client, claim, case_analysis.language, region, rag_mode, filter_target, query_strategy
                )
        evidence_gathering_seconds = round(time.time() - _t0, 2)
        for claim in (surfaced_claims or []):
            items, block = claim_results[claim]
            evidence_by_claim[claim] = items
            all_query_texts += [it.query_used for it in items if it.query_used]
            blocks.append(f"Claim: {claim}\n{block}")
        evidence_block = "\n\n".join(blocks) if blocks else "No claims were surfaced this round to verify."

        region_context = region or "unknown"
        prompt = prompts.build_defender_prompt(
            comment, case_analysis.to_dict(), defender_persona, prosecutor_argument,
            surfaced_claims, round_num, evidence_block, rag_mode, region_context,
            web_search_enabled=config.ENABLE_WEB_SEARCH,
        )
        # max_tokens=1800 (was 1000) - same truncation bug class recurring
        # for non-Latin-script languages (e.g. Tamil): this schema's 6
        # fields (one a claim_assessments list) plus a full rebuttal in a
        # script that costs more tokens per character than English can
        # still exceed 1000 tokens mid-JSON, causing every parse attempt -
        # including the corrective retry, which reuses this same budget -
        # to fail (observed as "Defender round 1 parse failure" on a
        # Tamil test-set row).
        _t1 = time.time()
        parsed = self.client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.3, max_tokens=1800)
        response_generation_seconds = round(time.time() - _t1, 2)
        timings = {"evidence_gathering_seconds": evidence_gathering_seconds,
                   "response_generation_seconds": response_generation_seconds,
                   "num_claims": len(surfaced_claims or []),
                   "batched": config.BATCH_QUERY_FORMULATION}
        logger.info("Defender round %d timing: %d claim(s), evidence_gathering=%.2fs, response_generation=%.2fs",
                    round_num, len(surfaced_claims or []), evidence_gathering_seconds, response_generation_seconds)

        if not parsed:
            logger.warning("Defender round %d parse failure - returning an empty rebuttal for this round.", round_num)
            return DebateRound(
                round=round_num, prosecutor_argument=prosecutor_argument, surfaced_claims=surfaced_claims or [],
                safety_notes=["defender_parse_failure"], timings=timings,
            )

        assessments = []
        for idx, ca in enumerate(parsed.get("claim_assessments") or []):
            # Defensive: a weaker model may return this list as plain strings
            # rather than structured objects - coerce rather than crash.
            ca = as_dict_item(ca, fallback_key="claim")
            claim_text = ca.get("claim", "")
            verdict = ca.get("verdict") if ca.get("verdict") in _VALID_VERDICTS else "NEI"
            # evidence_by_claim is keyed by the ORIGINAL surfaced_claims text, but
            # the model is free to paraphrase a claim when restating it here -
            # an exact-text lookup would then silently return [] even though
            # real evidence was retrieved for that claim. Exact match first
            # (the common case, and the only correct one when there are
            # duplicate/similar claims); if that fails, fall back to the
            # position this assessment appears at, aligned with surfaced_claims'
            # original order - a much closer guess than giving up on evidence
            # entirely. Logged so a persistently high fallback rate is visible
            # rather than a silent, invisible degradation of what the Judge sees.
            if claim_text in evidence_by_claim:
                evidence = evidence_by_claim[claim_text]
            else:
                ordered_claims = surfaced_claims or []
                if idx < len(ordered_claims):
                    fallback_claim = ordered_claims[idx]
                    evidence = evidence_by_claim.get(fallback_claim, [])
                    logger.warning(
                        "Defender round %d: claim_assessments[%d] text %r did not exactly match any "
                        "surfaced claim - falling back to positional match with surfaced_claims[%d]=%r.",
                        round_num, idx, claim_text, idx, fallback_claim,
                    )
                else:
                    evidence = []
            assessments.append(EvidenceAssessment(
                claim=claim_text, verdict=verdict,
                evidence=evidence,
                reasoning_summary=ca.get("reasoning_summary") or "",
                usable_in_final_cn=bool(ca.get("usable_in_final_cn")),
            ))

        return DebateRound(
            round=round_num,
            round_objective=f"Prosecutor/Defender exchange, round {round_num}",
            prosecutor_argument=prosecutor_argument,
            surfaced_claims=surfaced_claims or [],
            retrieval_queries=all_query_texts,
            evidence_assessments=assessments,
            defender_response=parsed.get("defender_response") or "",
            new_information=parsed.get("new_information") or [],
            unresolved_questions=parsed.get("unresolved_questions") or [],
            cultural_notes=parsed.get("cultural_notes") or [],
            safety_notes=parsed.get("safety_notes") or [],
            timings=timings,
        )
