"""
Defender Agent - open-ended persona (same one across all rounds of a
track). Evidence verification is folded into every turn here: per-round,
each Prosecutor-surfaced claim is checked against bilingual RAG retrieval
and assessed SUPPORTED / REFUTED / NEI in the same LLM call that produces
the rebuttal - there is no separate post-hoc fact-checking pass.
"""
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
        evidence_by_claim = {}
        all_query_texts = []
        blocks = []
        for claim in (surfaced_claims or []):
            items, block = evidence_verifier.gather_evidence_for_claim(
                self.client, claim, case_analysis.language, region, rag_mode, filter_target, query_strategy
            )
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
        parsed = self.client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.3, max_tokens=1800)

        if not parsed:
            logger.warning("Defender round %d parse failure - returning an empty rebuttal for this round.", round_num)
            return DebateRound(
                round=round_num, prosecutor_argument=prosecutor_argument, surfaced_claims=surfaced_claims or [],
                safety_notes=["defender_parse_failure"],
            )

        assessments = []
        for ca in (parsed.get("claim_assessments") or []):
            # Defensive: a weaker model may return this list as plain strings
            # rather than structured objects - coerce rather than crash.
            ca = as_dict_item(ca, fallback_key="claim")
            claim_text = ca.get("claim", "")
            verdict = ca.get("verdict") if ca.get("verdict") in _VALID_VERDICTS else "NEI"
            assessments.append(EvidenceAssessment(
                claim=claim_text, verdict=verdict,
                evidence=evidence_by_claim.get(claim_text, []),
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
        )
