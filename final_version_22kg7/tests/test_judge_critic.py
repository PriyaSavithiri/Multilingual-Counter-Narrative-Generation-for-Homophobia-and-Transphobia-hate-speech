"""
Regression coverage for a real bug found via live testing: the Judge's
LLM call sometimes returns selected_language="en" regardless of the actual
input language (observed on a Basque "eu" input, most likely because the
prompt's own JSON-schema example literally showed "en" as a placeholder -
a weaker local model pattern-matched the example instead of substituting
the real value). case_analysis.language (already cross-checked against
detect_language()'s script/marker heuristics) must always win.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge_critic_agent import JudgeCriticAgent
from model_api import ModelClient
from schemas import CaseAnalysis, DebateRound, PersonaCandidate, PersonaSelection, RouterDecision, SelectedPersona


class _WrongLanguageClient(ModelClient):
    """Reproduces the exact observed bug: returns selected_language="en"
    no matter what language the case actually is."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        payload = {
            "selected_language": "en",  # wrong on purpose - simulates the observed bug
            "core_claim_to_counter": "some claim about the comentario", "recommended_strategy": [],
            "approved_evidence": [], "rejected_content": [],
            "cultural_guidance": [], "safety_guidance": [],
            "final_response_plan": "plan",
        }
        return {k: payload.get(k) for k in response_schema if k in payload}


class _HallucinatingClient(ModelClient):
    """Reproduces a real failure observed via live testing: under heavy
    Deep-Dive debate context, the Judge produced a plausible-but-unrelated
    plan (a customer-service/product-safety FAQ) instead of reasoning about
    the actual hate-speech case. First call is off-topic; the corrective
    retry produces an on-topic plan."""
    backend_name = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        self.calls += 1
        if response_schema is None:
            return "n/a"
        if self.calls == 1:
            payload = {
                "selected_language": "en",
                "core_claim_to_counter": "Customers should contact support if the product arrives damaged.",
                "recommended_strategy": [], "approved_evidence": [], "rejected_content": [],
                "cultural_guidance": [], "safety_guidance": [],
                "final_response_plan": "Advise the customer to request a refund within 30 days.",
            }
        else:
            payload = {
                "selected_language": "en",
                "core_claim_to_counter": "Gay men are not a threat to other men in shared spaces.",
                "recommended_strategy": [], "approved_evidence": [], "rejected_content": [],
                "cultural_guidance": [], "safety_guidance": [],
                "final_response_plan": "Write a calm rebuttal citing evidence about gay men and public spaces.",
            }
        return {k: payload.get(k) for k in response_schema if k in payload}


class _AlwaysHallucinatingClient(ModelClient):
    """Off-topic on both the original call and the corrective retry - must
    end up falling back to a safe plan, not silently accepted."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        payload = {
            "selected_language": "en",
            "core_claim_to_counter": "Customers should contact support if the product arrives damaged.",
            "recommended_strategy": [], "approved_evidence": [], "rejected_content": [],
            "cultural_guidance": [], "safety_guidance": [],
            "final_response_plan": "Advise the customer to request a refund within 30 days.",
        }
        return {k: payload.get(k) for k in response_schema if k in payload}


class _EmptyClient(ModelClient):
    backend_name = "empty"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        return "n/a" if response_schema is None else {}


class _StringShapedEvidenceClient(ModelClient):
    """Reproduces a real crash observed with mistral:7b-instruct-v0.3:
    approved_evidence/rejected_content returned as lists of plain strings
    instead of structured objects, which crashed with AttributeError on
    the old (unguarded) `.get()` calls."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        payload = {
            "selected_language": "eu", "core_claim_to_counter": "claim about the comment", "recommended_strategy": [],
            "approved_evidence": ["some evidence passage as a bare string"],
            "rejected_content": ["some rejected content as a bare string"],
            "cultural_guidance": [], "safety_guidance": [], "final_response_plan": "plan",
        }
        return {k: payload.get(k) for k in response_schema if k in payload}


def _persona_selection():
    prosecutor = SelectedPersona(name="p", role="prosecutor", objective="o")
    defender = SelectedPersona(name="d", role="defender", objective="o")
    return PersonaSelection(candidate_personas=[PersonaCandidate(name="p", role_type="x")],
                             selected_prosecutor=prosecutor, selected_defender=defender)


def test_judge_never_overrides_the_authoritative_case_analysis_language():
    ca = CaseAnalysis(language="eu", hate_type="implicit")  # Basque, like the real bug report
    router_decision = RouterDecision(selected_track="deep_dive", confidence=0.8, rationale="r")
    rounds = [DebateRound(round=1, defender_response="resp")]

    agent = JudgeCriticAgent(_WrongLanguageClient())
    plan = agent.run("comentario", ca, _persona_selection(), router_decision, rounds, region=None)

    assert plan.selected_language == "eu", (
        f"Judge overrode the authoritative case_analysis.language - got {plan.selected_language!r}"
    )


def test_judge_parse_failure_also_uses_the_authoritative_language():
    ca = CaseAnalysis(language="ta", hate_type="explicit")
    router_decision = RouterDecision(selected_track="fast_track", confidence=0.9, rationale="r")
    rounds = [DebateRound(round=1, defender_response="resp")]

    agent = JudgeCriticAgent(_EmptyClient())
    plan = agent.run("comment", ca, _persona_selection(), router_decision, rounds, region=None)

    assert plan.selected_language == "ta"


def test_judge_handles_string_shaped_evidence_lists_without_crashing():
    ca = CaseAnalysis(language="eu", hate_type="explicit")
    router_decision = RouterDecision(selected_track="fast_track", confidence=0.9, rationale="r")
    rounds = [DebateRound(round=1, defender_response="resp")]

    agent = JudgeCriticAgent(_StringShapedEvidenceClient())
    plan = agent.run("comment", ca, _persona_selection(), router_decision, rounds, region=None)  # must not raise

    assert len(plan.approved_evidence) == 1
    assert plan.approved_evidence[0].passage == "some evidence passage as a bare string"
    assert len(plan.rejected_content) == 1
    assert plan.rejected_content[0].content == "some rejected content as a bare string"


def test_judge_retries_and_recovers_from_hallucinated_off_topic_output():
    ca = CaseAnalysis(language="en", hate_type="implicit", target_group="gay men", hate_category="homophobia")
    router_decision = RouterDecision(selected_track="deep_dive", confidence=0.8, rationale="r")
    rounds = [DebateRound(round=1, defender_response="resp", surfaced_claims=["gay men are a threat"])]

    client = _HallucinatingClient()
    agent = JudgeCriticAgent(client)
    plan = agent.run("Gay men are a threat to other men in the locker room.", ca, _persona_selection(),
                      router_decision, rounds, region=None)

    assert client.calls == 2, "must retry exactly once on off-topic output"
    assert "gay men" in plan.core_claim_to_counter.lower()
    assert plan.safety_guidance == []


def test_judge_falls_back_to_safe_plan_if_still_off_topic_after_retry():
    ca = CaseAnalysis(language="en", hate_type="implicit", target_group="gay men", hate_category="homophobia")
    router_decision = RouterDecision(selected_track="deep_dive", confidence=0.8, rationale="r")
    rounds = [DebateRound(round=1, defender_response="Actual on-topic fallback response about gay men.",
                           surfaced_claims=["gay men are a threat"])]

    agent = JudgeCriticAgent(_AlwaysHallucinatingClient())
    plan = agent.run("Gay men are a threat to other men in the locker room.", ca, _persona_selection(),
                      router_decision, rounds, region=None)

    assert plan.final_response_plan == "Actual on-topic fallback response about gay men."
    assert "judge_output_appeared_off_topic_after_retry" in plan.safety_guidance[0]


def test_judge_handles_nested_list_shaped_surfaced_claims_without_crashing():
    # Regression test for a real crash found via live testing on Colab: a real
    # model's Prosecutor output put a nested list inside surfaced_claims (instead
    # of a flat list of strings), which crashed shares_key_terms()'s regex with
    # "TypeError: expected string or bytes-like object, got 'list'" deep inside a
    # real run - caught by pipeline.py's broad exception handler, but that row's
    # generation was still ruined. utils._flatten_to_text() now coerces this.
    ca = CaseAnalysis(language="en", hate_type="implicit", target_group="gay men", hate_category="homophobia")
    router_decision = RouterDecision(selected_track="deep_dive", confidence=0.8, rationale="r")
    rounds = [DebateRound(round=1, defender_response="resp",
                           surfaced_claims=["gay men are a threat", ["nested", "list", "instead", "of", "string"]])]

    agent = JudgeCriticAgent(_HallucinatingClient())
    plan = agent.run("Gay men are a threat to other men in the locker room.", ca, _persona_selection(),
                      router_decision, rounds, region=None)  # must not raise

    assert "gay men" in plan.core_claim_to_counter.lower()


if __name__ == "__main__":
    test_judge_never_overrides_the_authoritative_case_analysis_language()
    test_judge_parse_failure_also_uses_the_authoritative_language()
    test_judge_handles_string_shaped_evidence_lists_without_crashing()
    test_judge_retries_and_recovers_from_hallucinated_off_topic_output()
    test_judge_falls_back_to_safe_plan_if_still_off_topic_after_retry()
    test_judge_handles_nested_list_shaped_surfaced_claims_without_crashing()
    print("test_judge_critic.py: ALL PASSED")
