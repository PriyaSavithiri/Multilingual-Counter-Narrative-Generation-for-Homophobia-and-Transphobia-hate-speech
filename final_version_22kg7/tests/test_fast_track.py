import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fast_track
from schemas import CaseAnalysis, DebateRound, PersonaCandidate, PersonaSelection, SelectedPersona


class _FakeProsecutor:
    def __init__(self):
        self.calls = []

    def run_round(self, comment, case_analysis, persona, round_num, track, prior_rounds_summary=""):
        self.calls.append((round_num, track, persona))
        return "internal argument", ["claim A"]


class _FakeDefender:
    def __init__(self):
        self.calls = []

    def run_round(self, comment, case_analysis, persona, prosecutor_argument, surfaced_claims, round_num,
                   rag_mode, region, filter_target, query_strategy=None):
        self.calls.append((round_num, persona))
        return DebateRound(round=round_num, prosecutor_argument=prosecutor_argument, surfaced_claims=surfaced_claims,
                            defender_response=f"rebuttal {round_num}")


def _persona_selection():
    prosecutor = SelectedPersona(name="p", role="prosecutor", objective="o")
    defender = SelectedPersona(name="d", role="defender", objective="o")
    return PersonaSelection(candidate_personas=[PersonaCandidate(name="p", role_type="x"),
                                                 PersonaCandidate(name="d", role_type="y")],
                             selected_prosecutor=prosecutor, selected_defender=defender)


def test_fast_track_runs_exactly_one_round():
    prosecutor, defender = _FakeProsecutor(), _FakeDefender()
    ca = CaseAnalysis(language="en", hate_type="explicit")
    rounds = fast_track.run("comment", ca, _persona_selection(), prosecutor, defender,
                             rag_mode="dual_rag", region=None, filter_target=True)
    assert len(rounds) == 1
    assert len(prosecutor.calls) == 1
    assert len(defender.calls) == 1
    assert prosecutor.calls[0][1] == "fast_track"


def test_fast_track_uses_the_generated_personas_not_new_ones():
    prosecutor, defender = _FakeProsecutor(), _FakeDefender()
    ca = CaseAnalysis(language="en", hate_type="explicit")
    selection = _persona_selection()
    fast_track.run("comment", ca, selection, prosecutor, defender, rag_mode="dual_rag", region=None,
                    filter_target=True)
    assert prosecutor.calls[0][2]["name"] == "p"
    assert defender.calls[0][1]["name"] == "d"


if __name__ == "__main__":
    test_fast_track_runs_exactly_one_round()
    test_fast_track_uses_the_generated_personas_not_new_ones()
    print("test_fast_track.py: ALL PASSED")
