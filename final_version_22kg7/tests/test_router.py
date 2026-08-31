import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import route
from schemas import CaseAnalysis


def test_explicit_routes_to_fast_track_only():
    decision = route(CaseAnalysis(language="en", hate_type="explicit", confidence=0.9))
    assert decision.selected_track == "fast_track"


def test_implicit_routes_to_deep_dive_only():
    decision = route(CaseAnalysis(language="en", hate_type="implicit", confidence=0.6))
    assert decision.selected_track == "deep_dive"


def test_unknown_hate_type_conservatively_routes_to_deep_dive():
    # CaseAnalysis's own validation normally coerces bad hate_type values to
    # "implicit" before route() ever sees them (see case_analysis_agent.py) -
    # route() itself has the same conservative behaviour if a caller bypasses that.
    decision = route(CaseAnalysis(language="en", hate_type="something_else", confidence=0.0))
    assert decision.selected_track == "deep_dive"


def test_only_one_track_selected_never_both():
    for hate_type in ("explicit", "implicit"):
        decision = route(CaseAnalysis(language="en", hate_type=hate_type, confidence=0.5))
        assert decision.selected_track in ("fast_track", "deep_dive")
        assert not (decision.selected_track == "fast_track" and hate_type == "implicit")


if __name__ == "__main__":
    test_explicit_routes_to_fast_track_only()
    test_implicit_routes_to_deep_dive_only()
    test_unknown_hate_type_conservatively_routes_to_deep_dive()
    test_only_one_track_selected_never_both()
    print("test_router.py: ALL PASSED")
