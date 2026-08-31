import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evidence_verifier
from defender_agent import DefenderAgent, _REQUIRED_FIELDS
from model_api import ModelClient
from schemas import CaseAnalysis, EvidenceItem

_REAL_EVIDENCE = [EvidenceItem(source_id="1", title="test-source", passage="real retrieved evidence",
                                retrieval_score=0.9, source_type="fact", language="en")]


class _ParaphrasingClient(ModelClient):
    """Returns a claim_assessments entry whose "claim" text is a paraphrase of
    the original surfaced claim, not a verbatim copy - simulating what a real
    model does (see EU114/basque quality investigation)."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema == _REQUIRED_FIELDS:
            return {
                "defender_response": "a rebuttal",
                "claim_assessments": [
                    {"claim": "Vaccines cause autism, according to some people",  # paraphrased, not verbatim
                     "verdict": "REFUTED", "reasoning_summary": "not supported", "usable_in_final_cn": True}
                ],
                "new_information": [], "unresolved_questions": [], "cultural_notes": [], "safety_notes": [],
            }
        return {"input_language_query": "q", "english_query": "q"}


def _fake_gather_evidence_for_claims(client, claims, language, region, rag_mode, filter_target, query_strategy=None):
    """Stands in for real RAG retrieval - returns canned evidence keyed by the
    EXACT original claim text, same contract as evidence_verifier's real
    function, so the test doesn't need a live FAISS index."""
    return {claim: (_REAL_EVIDENCE, "evidence block text") for claim in claims}


def test_paraphrased_claim_still_receives_evidence(monkeypatch):
    monkeypatch.setattr(evidence_verifier, "gather_evidence_for_claims", _fake_gather_evidence_for_claims)
    agent = DefenderAgent(_ParaphrasingClient())
    ca = CaseAnalysis(language="en")
    result = agent.run_round(
        comment="some comment", case_analysis=ca, defender_persona={"name": "d", "objective": "o"},
        prosecutor_argument="argument", surfaced_claims=["Vaccines cause autism"], round_num=1,
        rag_mode="dual_rag", region=None, filter_target=True,
    )
    assert len(result.evidence_assessments) == 1
    assessment = result.evidence_assessments[0]
    # The claim text on the assessment is the model's paraphrase (unchanged,
    # this fix never rewrites what the model said)...
    assert assessment.claim == "Vaccines cause autism, according to some people"
    # ...but the evidence attached is still the real retrieved evidence, found
    # via positional fallback rather than lost to an exact-text mismatch.
    assert assessment.evidence == _REAL_EVIDENCE


def test_exact_claim_match_still_works_unchanged(monkeypatch):
    """The common case (model returns the claim verbatim) must still work via
    the exact-match path, with no positional fallback needed."""
    monkeypatch.setattr(evidence_verifier, "gather_evidence_for_claims", _fake_gather_evidence_for_claims)

    class _VerbatimClient(ModelClient):
        backend_name = "stub"

        def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
            if response_schema == _REQUIRED_FIELDS:
                return {
                    "defender_response": "a rebuttal",
                    "claim_assessments": [{"claim": "Vaccines cause autism", "verdict": "REFUTED",
                                            "reasoning_summary": "not supported", "usable_in_final_cn": True}],
                    "new_information": [], "unresolved_questions": [], "cultural_notes": [], "safety_notes": [],
                }
            return {"input_language_query": "q", "english_query": "q"}

    agent = DefenderAgent(_VerbatimClient())
    ca = CaseAnalysis(language="en")
    result = agent.run_round(
        comment="some comment", case_analysis=ca, defender_persona={"name": "d", "objective": "o"},
        prosecutor_argument="argument", surfaced_claims=["Vaccines cause autism"], round_num=1,
        rag_mode="dual_rag", region=None, filter_target=True,
    )
    assert result.evidence_assessments[0].evidence == _REAL_EVIDENCE


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_paraphrased_claim_still_receives_evidence(_FakeMonkeypatch())
    test_exact_claim_match_still_works_unchanged(_FakeMonkeypatch())
    print("test_defender_agent.py: ALL PASSED")
