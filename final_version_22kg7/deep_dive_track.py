"""
Deep-Dive - implicit hate. K rounds (config.get_deep_dive_rounds(), hard
default K=3, overridable via DEEP_DIVE_ROUNDS env var), using the SAME two
personas across every round (generated once before routing - not
regenerated per round, per the confirmed architecture: persona generation
and persona deployment depth are independent).
"""
import config


def run(comment: str, case_analysis, persona_selection, prosecutor_agent, defender_agent,
        rag_mode: str, region: str, filter_target: bool, query_strategy: str = None, k: int = None) -> list:
    k = k or config.get_deep_dive_rounds()
    rounds = []
    prior_summary = ""
    for round_num in range(1, k + 1):
        prosecutor_argument, surfaced_claims = prosecutor_agent.run_round(
            comment, case_analysis, persona_selection.selected_prosecutor.to_dict(),
            round_num=round_num, track="deep_dive", prior_rounds_summary=prior_summary,
        )
        round_result = defender_agent.run_round(
            comment, case_analysis, persona_selection.selected_defender.to_dict(), prosecutor_argument,
            surfaced_claims, round_num=round_num, rag_mode=rag_mode, region=region,
            filter_target=filter_target, query_strategy=query_strategy,
        )
        rounds.append(round_result)
        prior_summary = (
            f"Round {round_num} unresolved questions: {round_result.unresolved_questions}; "
            f"new information surfaced: {round_result.new_information}"
        )
    return rounds
