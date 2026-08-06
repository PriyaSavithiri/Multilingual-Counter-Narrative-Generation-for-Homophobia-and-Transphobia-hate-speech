import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from model_api import ModelClient
from persona_generator_agent import PersonaGeneratorAgent
from schemas import CaseAnalysis


class _StubClient(ModelClient):
    backend_name = "stub"

    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        return {k: self.payload.get(k) for k in response_schema if k in self.payload}


def _payload_with_n_candidates(n):
    candidates = [
        {"name": f"expert {i}", "role_type": "specialist", "expertise": ["topic"],
         "cultural_relevance": "n/a", "strategy": "s", "relevance_score": 0.5, "suited_for": "either"}
        for i in range(n)
    ]
    return {
        "candidate_personas": candidates,
        "selected_prosecutor": {"name": "expert 0", "role": "prosecutor", "objective": "o", "boundaries": []},
        "selected_defender": {"name": "expert 1", "role": "defender", "objective": "o", "expertise": [],
                               "cultural_guidance": []},
        "selection_rationale": "test",
    }


def test_default_candidate_count_is_seven():
    ca = CaseAnalysis(language="en")
    agent = PersonaGeneratorAgent(_StubClient(_payload_with_n_candidates(7)))
    selection = agent.run("comment", ca)
    assert len(selection.candidate_personas) == 7


def test_candidate_count_is_clamped_into_5_to_10_range():
    assert config.clamp_persona_candidate_count(2) == config.PERSONA_CANDIDATE_COUNT_MIN
    assert config.clamp_persona_candidate_count(50) == config.PERSONA_CANDIDATE_COUNT_MAX
    assert config.clamp_persona_candidate_count(8) == 8
    assert config.clamp_persona_candidate_count("not a number") == config.DEFAULT_PERSONA_CANDIDATE_COUNT


def test_exactly_one_prosecutor_and_one_defender_selected():
    ca = CaseAnalysis(language="en")
    agent = PersonaGeneratorAgent(_StubClient(_payload_with_n_candidates(6)))
    selection = agent.run("comment", ca, candidate_count=6)
    assert selection.selected_prosecutor.role == "prosecutor"
    assert selection.selected_defender.role == "defender"
    assert selection.selected_prosecutor.name != selection.selected_defender.name


def test_personas_are_not_limited_to_any_fixed_list():
    # No ALLOWED_PERSONAS-style constant exists anywhere in new_arch - this
    # test documents that guarantee structurally: persona_generator_agent
    # accepts whatever names the LLM invents, with no allow-list filtering.
    import persona_generator_agent
    assert not hasattr(persona_generator_agent, "ALLOWED_PERSONAS")
    ca = CaseAnalysis(language="en")
    weird_payload = _payload_with_n_candidates(5)
    weird_payload["candidate_personas"][0]["name"] = "a completely novel invented role never seen before"
    agent = PersonaGeneratorAgent(_StubClient(weird_payload))
    selection = agent.run("comment", ca, candidate_count=5)
    assert selection.candidate_personas[0].name == "a completely novel invented role never seen before"


def test_parse_failure_uses_safe_generic_fallback_pair():
    ca = CaseAnalysis(language="en")
    agent = PersonaGeneratorAgent(_StubClient({}))
    selection = agent.run("comment", ca)
    assert selection.selected_prosecutor.name != selection.selected_defender.name
    assert len(selection.candidate_personas) == config.DEFAULT_PERSONA_CANDIDATE_COUNT


def test_personal_name_style_personas_are_replaced_with_role_type():
    # Regression test: reproduces the exact buggy output observed from a
    # real qwen2.5:7b-instruct run, which ignored the "role title, not a
    # personal name" instruction and invented "Dr./Rev./Prof. Firstname
    # Lastname" style names. These must never surface as-is.
    ca = CaseAnalysis(language="en")
    payload = {
        "candidate_personas": [
            {"name": "Dr. Michael Green", "role_type": "Forensic Linguist", "expertise": [],
             "cultural_relevance": "n/a", "strategy": "s", "relevance_score": 0.4, "suited_for": "prosecutor"},
            {"name": "Dr. Emily Carter", "role_type": "Psychologist", "expertise": [],
             "cultural_relevance": "n/a", "strategy": "s", "relevance_score": 0.8, "suited_for": "defender"},
            {"name": "Rev. John Harper", "role_type": "Religious Studies Scholar", "expertise": [],
             "cultural_relevance": "n/a", "strategy": "s", "relevance_score": 0.6, "suited_for": "defender"},
        ],
        "selected_prosecutor": {"name": "Dr. Michael Green", "role": "prosecutor", "objective": "o", "boundaries": []},
        "selected_defender": {"name": "Dr. Emily Carter", "role": "defender", "objective": "o", "expertise": [],
                               "cultural_guidance": []},
        "selection_rationale": "test",
    }
    agent = PersonaGeneratorAgent(_StubClient(payload))
    selection = agent.run("comment", ca, candidate_count=3)

    for candidate in selection.candidate_personas:
        assert "Dr." not in candidate.name and "Rev." not in candidate.name, (
            f"Personal-name-shaped persona leaked through: {candidate.name!r}"
        )
    assert selection.selected_prosecutor.name == "Forensic Linguist"
    assert selection.selected_defender.name == "Psychologist"


if __name__ == "__main__":
    test_default_candidate_count_is_seven()
    test_candidate_count_is_clamped_into_5_to_10_range()
    test_exactly_one_prosecutor_and_one_defender_selected()
    test_personas_are_not_limited_to_any_fixed_list()
    test_parse_failure_uses_safe_generic_fallback_pair()
    test_personal_name_style_personas_are_replaced_with_role_type()
    print("test_persona_generation.py: ALL PASSED")
