"""
Regression coverage for a real bug found via live testing: on a Basque
input, a weaker model's Final-CN output substantially echoed the original
hate comment's own wording/structure ("[target group] have a disease...")
instead of genuinely refuting it - which can read as affirming the claim
rather than countering it. _echoes_source_wording() is a language-agnostic
lexical-similarity check (works identically for en/ta/eu/es/it, no
per-language tokenizer needed) that catches this and triggers one
corrective retry.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from final_cn_agent import FinalCNAgent, _echoes_source_wording
from model_api import ModelClient
from schemas import JudgePlan


def test_echoes_source_wording_detects_near_verbatim_reuse():
    source = "Homosexualak gaixotasun bat dira eta gure ohiko gune gizonak izorratzen dituzte"
    echoing_output = "Euskal gizonak gaixotasun bat dituzte eta izorratzen dituzte."
    assert _echoes_source_wording(echoing_output, source) is True


def test_echoes_source_wording_allows_genuine_ontopic_rebuttal():
    source = "Homosexualak gaixotasun bat dira eta gure ohiko gune gizonak izorratzen dituzte"
    genuine_rebuttal = (
        "Homosexualitatea ez da gaixotasuna; adituek berretsi dute jarrera "
        "naturala eta osasungarria dela pertsona ororentzat."
    )
    assert _echoes_source_wording(genuine_rebuttal, source) is False


def test_echoes_source_wording_catches_real_mistral_case_ratio_alone_missed():
    # Regression test for a real mistral:7b-instruct-v0.3 output that verbatim-copied
    # a 64-character phrase from the input yet scored ratio=0.503 - almost identical
    # to a hand-written GENUINE negation's ratio of 0.510 above. Ratio alone cannot
    # tell these apart; the longest-matching-block signal is what catches this one.
    source = (
        "Benetako euskal gizon bat kuadrilan egon behar da gizon bezala portatzeko, ez "
        "marikoikeriak erakusteko. Homosexualak gaixotasun bat dira eta gure ohiko gune "
        "gizonak izorratzen dituzte"
    )
    real_bad_output = (
        "Ez dago behar da gizon bezala portatzeko, ez marikoikeriak erakusteko. Gure ohiko "
        "gune gizonak izorratzen duzu eta haukizunak eman dira. Eman daude, euskararen eta "
        "gaur bat, euskal-gazteluak eta heterosexualak gaixotasunak batezko dituen zure "
        "erabiltzaileak."
    )
    assert _echoes_source_wording(real_bad_output, source) is True


# Every stub below must also answer the semantic-fidelity check's own LLM
# call (response_schema=["fidelity", "reason"]) - it fires after a
# counter_narrative is settled on, for every test, regardless of what that
# test is actually exercising. Defaulting to "on_target" keeps each test
# isolated to the one guard it's meant to test, mirroring how
# test_final_cn_retries_once_on_total_parse_failure_and_recovers already
# hand-picks vocabulary to avoid tripping the off-topic guard by accident.
def _fidelity_stub_response(response_schema):
    if response_schema and "fidelity" in response_schema:
        return {"fidelity": "on_target", "reason": "stub - not under test here"}
    return None


class _EchoingClient(ModelClient):
    """First call echoes the source; retry produces a genuine rebuttal."""
    backend_name = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        self.calls += 1
        if response_schema is None:
            return "n/a"
        fidelity = _fidelity_stub_response(response_schema)
        if fidelity is not None:
            return fidelity
        if self.calls == 1:
            return {"counter_narrative": "Euskal gizonak gaixotasun bat dituzte eta izorratzen dituzte.",
                    "explanation": "exp"}
        return {"counter_narrative": "Jendeak errespetua merezi du, edonor izanik ere.", "explanation": "exp2"}


class _AlwaysEchoingClient(ModelClient):
    """Both the original call and the corrective retry echo the source -
    must end up flagged, not silently accepted."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        fidelity = _fidelity_stub_response(response_schema)
        if fidelity is not None:
            return fidelity
        return {"counter_narrative": "Euskal gizonak gaixotasun bat dituzte eta izorratzen dituzte.",
                "explanation": "exp"}


def test_final_cn_retries_once_and_recovers_from_echoing_output():
    comment = "Homosexualak gaixotasun bat dira eta gure ohiko gune gizonak izorratzen dituzte"
    plan = JudgePlan(selected_language="eu", final_response_plan="plan")
    agent = FinalCNAgent(_EchoingClient())
    result = agent.run(comment, plan, region=None)
    assert result["counter_narrative"] == "Jendeak errespetua merezi du, edonor izanik ere."
    assert result["safety_flags"] == []


def test_final_cn_flags_when_retry_also_echoes():
    comment = "Homosexualak gaixotasun bat dira eta gure ohiko gune gizonak izorratzen dituzte"
    plan = JudgePlan(selected_language="eu", final_response_plan="plan")
    agent = FinalCNAgent(_AlwaysEchoingClient())
    result = agent.run(comment, plan, region=None)
    assert "counter_narrative_may_echo_source_wording" in result["safety_flags"]


class _OffTopicClient(ModelClient):
    """Reproduces a hallucinated Final-CN output slipping through from an
    off-topic Judge plan: the first reply describes an unrelated
    customer-service scenario; the corrective retry produces an on-topic
    counter-narrative."""
    backend_name = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        self.calls += 1
        if response_schema is None:
            return "n/a"
        fidelity = _fidelity_stub_response(response_schema)
        if fidelity is not None:
            return fidelity
        if self.calls == 1:
            return {"counter_narrative": "Please contact customer support for a refund on your damaged product.",
                    "explanation": "exp"}
        return {"counter_narrative": ("Being gay does not make someone a threat to those around them; this is "
                                       "a harmful stereotype, not a fact."), "explanation": "exp2"}


class _AlwaysOffTopicClient(ModelClient):
    """Off-topic on both the original call and the corrective retry - must
    end up flagged, not silently accepted."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        fidelity = _fidelity_stub_response(response_schema)
        if fidelity is not None:
            return fidelity
        return {"counter_narrative": "Please contact customer support for a refund on your damaged product.",
                "explanation": "exp"}


def test_final_cn_retries_and_recovers_from_off_topic_output():
    comment = "Gay men are a threat to other men in the locker room."
    plan = JudgePlan(selected_language="en", core_claim_to_counter="gay men are not a threat",
                      final_response_plan="plan")
    agent = FinalCNAgent(_OffTopicClient())
    result = agent.run(comment, plan, region=None)
    assert result["counter_narrative"] == (
        "Being gay does not make someone a threat to those around them; this is "
        "a harmful stereotype, not a fact."
    )
    assert result["safety_flags"] == []


def test_final_cn_flags_when_retry_is_still_off_topic():
    comment = "Gay men are a threat to other men in the locker room."
    plan = JudgePlan(selected_language="en", core_claim_to_counter="gay men are not a threat",
                      final_response_plan="plan")
    agent = FinalCNAgent(_AlwaysOffTopicClient())
    result = agent.run(comment, plan, region=None)
    assert "counter_narrative_appeared_off_topic" in result["safety_flags"]


class _SemanticDriftClient(ModelClient):
    """Reproduces the real semantic-drift bug found via live testing (a
    transphobic gender-identity-denial comment - "you were born a man,
    you'll always be a man" - answered with a rebuttal about gender ROLE
    stereotypes/sexism instead of gender identity itself). Both the drifted
    and corrected texts share real vocabulary with the case (e.g. "identity"),
    so neither lexical guard (shares_key_terms/_echoes_source_wording) would
    ever catch this - only the semantic-fidelity check can. First reply
    drifts; the corrective retry stays on the specific claim."""
    backend_name = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        self.calls += 1
        if response_schema is None:
            return "n/a"
        if response_schema and "fidelity" in response_schema:
            if self.calls <= 2:
                return {"fidelity": "off_target", "reason": "answers gender roles, not identity denial"}
            return {"fidelity": "on_target", "reason": "directly addresses gender identity vs birth sex"}
        if self.calls == 1:
            return {"counter_narrative": ("Everyone's identity deserves respect, and no one should be limited "
                                           "to traditional gender roles or forced into rigid stereotypes."),
                    "explanation": "exp"}
        return {"counter_narrative": ("Gender identity is a real and valid part of who someone is, and it is "
                                       "not determined solely by the sex assigned at birth."), "explanation": "exp2"}


class _AlwaysSemanticDriftClient(ModelClient):
    """Semantically drifts on both the original call and the corrective
    retry - must end up flagged, not silently accepted."""
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        if response_schema and "fidelity" in response_schema:
            return {"fidelity": "off_target", "reason": "answers gender roles, not identity denial"}
        return {"counter_narrative": ("Everyone's identity deserves respect, and no one should be limited "
                                       "to traditional gender roles or forced into rigid stereotypes."),
                "explanation": "exp"}


def test_final_cn_retries_and_recovers_from_semantic_drift():
    comment = "You were born a man, you will always be a man, no matter what you pretend."
    plan = JudgePlan(selected_language="en",
                      core_claim_to_counter="gender identity is invalid because it contradicts birth sex",
                      final_response_plan="plan")
    agent = FinalCNAgent(_SemanticDriftClient())
    result = agent.run(comment, plan, region=None)
    assert result["counter_narrative"] == (
        "Gender identity is a real and valid part of who someone is, and it is "
        "not determined solely by the sex assigned at birth."
    )
    assert result["safety_flags"] == []


def test_final_cn_flags_when_retry_still_semantically_drifts():
    comment = "You were born a man, you will always be a man, no matter what you pretend."
    plan = JudgePlan(selected_language="en",
                      core_claim_to_counter="gender identity is invalid because it contradicts birth sex",
                      final_response_plan="plan")
    agent = FinalCNAgent(_AlwaysSemanticDriftClient())
    result = agent.run(comment, plan, region=None)
    assert "counter_narrative_semantic_drift" in result["safety_flags"]


class _FailsOnceThenSucceedsClient(ModelClient):
    """Regression test for a real bug found via live testing: a Judge
    hallucination produced an internally-contradictory Final-CN prompt, and
    the resulting JSON failed to parse - with no retry at all (the echo-check
    retry only applies once parsing has already succeeded). This client
    reproduces "first call unparseable, retry succeeds"."""
    backend_name = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        self.calls += 1
        if response_schema is None:
            return "n/a"
        fidelity = _fidelity_stub_response(response_schema)
        if fidelity is not None:
            return fidelity
        if self.calls == 1:
            return {}  # simulates parse_json_object() failing entirely
        return {"counter_narrative": "Errespetua eta duintasuna denontzat.", "explanation": "exp"}


class _AlwaysFailsToParseClient(ModelClient):
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        return "n/a" if response_schema is None else {}


def test_final_cn_retries_once_on_total_parse_failure_and_recovers():
    # core_claim_to_counter shares "errespetua"/"duintasuna" with the eventual
    # recovered output on purpose - keeps this test isolated from the separate
    # off-topic guard (see test_final_cn_retries_and_recovers_from_off_topic_output).
    plan = JudgePlan(selected_language="eu", core_claim_to_counter="errespetua eta duintasuna",
                      final_response_plan="plan")
    client = _FailsOnceThenSucceedsClient()
    agent = FinalCNAgent(client)
    result = agent.run("comment", plan, region=None)
    # 1 failed parse + 1 successful retry + 1 semantic-fidelity check call = 3
    assert client.calls == 3, "must retry exactly once on total parse failure, plus the fidelity check call"
    assert result["counter_narrative"] == "Errespetua eta duintasuna denontzat."
    assert result["safety_flags"] == []


def test_final_cn_gives_up_gracefully_if_retry_also_fails_to_parse():
    plan = JudgePlan(selected_language="eu", final_response_plan="plan")
    agent = FinalCNAgent(_AlwaysFailsToParseClient())
    result = agent.run("comment", plan, region=None)  # must not raise
    assert result["counter_narrative"] == ""
    assert "final_cn_parse_failure" in result["safety_flags"]


if __name__ == "__main__":
    test_echoes_source_wording_detects_near_verbatim_reuse()
    test_echoes_source_wording_allows_genuine_ontopic_rebuttal()
    test_echoes_source_wording_catches_real_mistral_case_ratio_alone_missed()
    test_final_cn_retries_once_and_recovers_from_echoing_output()
    test_final_cn_flags_when_retry_also_echoes()
    test_final_cn_retries_once_on_total_parse_failure_and_recovers()
    test_final_cn_gives_up_gracefully_if_retry_also_fails_to_parse()
    test_final_cn_retries_and_recovers_from_off_topic_output()
    test_final_cn_flags_when_retry_is_still_off_topic()
    test_final_cn_retries_and_recovers_from_semantic_drift()
    test_final_cn_flags_when_retry_still_semantically_drifts()
    print("test_final_cn.py: ALL PASSED")
