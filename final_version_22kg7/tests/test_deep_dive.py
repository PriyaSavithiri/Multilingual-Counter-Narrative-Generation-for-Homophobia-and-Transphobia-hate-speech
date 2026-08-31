import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import deep_dive_track
from schemas import CaseAnalysis, DebateRound, PersonaCandidate, PersonaSelection, SelectedPersona


class _FakeProsecutor:
    def __init__(self):
        self.calls = []

    def run_round(self, comment, case_analysis, persona, round_num, track, prior_rounds_summary=""):
        self.calls.append((round_num, track, persona.get("name")))
        return f"internal argument round {round_num}", [f"claim {round_num}"]


class _FakeDefender:
    def __init__(self):
        self.calls = []

    def run_round(self, comment, case_analysis, persona, prosecutor_argument, surfaced_claims, round_num,
                   rag_mode, region, filter_target, query_strategy=None):
        self.calls.append((round_num, persona.get("name")))
        return DebateRound(round=round_num, prosecutor_argument=prosecutor_argument, surfaced_claims=surfaced_claims,
                            defender_response=f"rebuttal {round_num}", unresolved_questions=["q"], new_information=["n"])


def _persona_selection():
    prosecutor = SelectedPersona(name="p", role="prosecutor", objective="o")
    defender = SelectedPersona(name="d", role="defender", objective="o")
    return PersonaSelection(candidate_personas=[PersonaCandidate(name="p", role_type="x"),
                                                 PersonaCandidate(name="d", role_type="y")],
                             selected_prosecutor=prosecutor, selected_defender=defender)


def test_deep_dive_runs_default_k_rounds():
    prosecutor, defender = _FakeProsecutor(), _FakeDefender()
    ca = CaseAnalysis(language="en", hate_type="implicit")
    rounds = deep_dive_track.run("comment", ca, _persona_selection(), prosecutor, defender,
                                  rag_mode="dual_rag", region=None, filter_target=True)
    assert len(rounds) == config.get_deep_dive_rounds() == 3
    assert [r.round for r in rounds] == [1, 2, 3]


def test_deep_dive_runs_configured_k_rounds_when_overridden():
    prosecutor, defender = _FakeProsecutor(), _FakeDefender()
    ca = CaseAnalysis(language="en", hate_type="implicit")
    rounds = deep_dive_track.run("comment", ca, _persona_selection(), prosecutor, defender,
                                  rag_mode="dual_rag", region=None, filter_target=True, k=5)
    assert len(rounds) == 5


def test_deep_dive_reuses_the_same_personas_across_all_rounds():
    prosecutor, defender = _FakeProsecutor(), _FakeDefender()
    ca = CaseAnalysis(language="en", hate_type="implicit")
    deep_dive_track.run("comment", ca, _persona_selection(), prosecutor, defender,
                         rag_mode="dual_rag", region=None, filter_target=True, k=3)
    prosecutor_names_used = {name for _, _, name in prosecutor.calls}
    defender_names_used = {name for _, name in defender.calls}
    assert prosecutor_names_used == {"p"}, "Prosecutor persona must not change across Deep-Dive rounds."
    assert defender_names_used == {"d"}, "Defender persona must not change across Deep-Dive rounds."


def test_deep_dive_passes_prior_round_context_forward():
    prosecutor, defender = _FakeProsecutor(), _FakeDefender()
    ca = CaseAnalysis(language="en", hate_type="implicit")
    deep_dive_track.run("comment", ca, _persona_selection(), prosecutor, defender,
                         rag_mode="dual_rag", region=None, filter_target=True, k=2)
    assert len(prosecutor.calls) == 2


if __name__ == "__main__":
    test_deep_dive_runs_default_k_rounds()
    test_deep_dive_runs_configured_k_rounds_when_overridden()
    test_deep_dive_reuses_the_same_personas_across_all_rounds()
    test_deep_dive_passes_prior_round_context_forward()
    print("test_deep_dive.py: ALL PASSED")
