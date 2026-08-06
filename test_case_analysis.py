import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from case_analysis_agent import CaseAnalysisAgent
from model_api import ModelClient


class _StubClient(ModelClient):
    backend_name = "stub"

    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        return {k: self.payload.get(k) for k in response_schema if k in self.payload}


def _base_payload(**overrides):
    payload = {
        "language": "en", "target_group": "gay men", "hate_category": "homophobia",
        "intent": ["moral_panic"], "strategy_hint": ["myth_correction"], "hate_type": "implicit",
        "confidence": 0.8, "hidden_claim": "being gay is unnatural", "evidence_topics": ["biology"],
        "cultural_context_needed": False,
        "region_suggestion": {"region": None, "country": None, "confidence": 0.1, "rationale": "weak"},
        "safety_notes": [], "rationale": "test",
    }
    payload.update(overrides)
    return payload


def test_valid_output_parses_into_case_analysis():
    agent = CaseAnalysisAgent(_StubClient(_base_payload()))
    result = agent.run("Being gay is unnatural.")
    assert result.language == "en"
    assert result.hate_type == "implicit"
    assert result.hidden_claim == "being gay is unnatural"


def test_low_confidence_region_suggestion_downgraded_to_unknown():
    payload = _base_payload(region_suggestion={
        "region": "Indian", "country": "India", "confidence": 0.2, "rationale": "weak guess",
    })
    agent = CaseAnalysisAgent(_StubClient(payload))
    result = agent.run("some comment")
    assert result.region_suggestion.region is None, "Below-threshold suggestions must never be treated as confirmed."


def test_high_confidence_region_suggestion_is_kept_as_a_suggestion_only():
    payload = _base_payload(region_suggestion={
        "region": "Indian", "country": "India", "confidence": 0.95,
        "rationale": "explicit mention of an Indian city and a named local law",
    })
    agent = CaseAnalysisAgent(_StubClient(payload))
    result = agent.run("some comment")
    assert result.region_suggestion.region == "Indian"
    assert result.region_suggestion.is_confident is True


def test_invalid_hate_type_falls_back_to_implicit():
    payload = _base_payload(hate_type="ambiguous_value")
    agent = CaseAnalysisAgent(_StubClient(payload))
    result = agent.run("some comment")
    assert result.hate_type == "implicit"


def test_empty_llm_output_produces_conservative_fallback():
    agent = CaseAnalysisAgent(_StubClient({}))
    result = agent.run("some comment", language_hint="en")
    assert result.language == "en"
    assert result.hate_type == "implicit"
    assert "case_analysis_parse_failure" in result.safety_notes


if __name__ == "__main__":
    test_valid_output_parses_into_case_analysis()
    test_low_confidence_region_suggestion_downgraded_to_unknown()
    test_high_confidence_region_suggestion_is_kept_as_a_suggestion_only()
    test_invalid_hate_type_falls_back_to_implicit()
    test_empty_llm_output_produces_conservative_fallback()
    print("test_case_analysis.py: ALL PASSED")
