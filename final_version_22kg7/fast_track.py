"""
Fast-Track - explicit hate. Exactly ONE Prosecutor <-> Defender exchange,
using the two personas generated once before routing (persona consistency -
this track does not generate new personas).
"""
def run(comment: str, case_analysis, persona_selection, prosecutor_agent, defender_agent,
        rag_mode: str, region: str, filter_target: bool, query_strategy: str = None) -> list:
    prosecutor_argument, surfaced_claims = prosecutor_agent.run_round(
        comment, case_analysis, persona_selection.selected_prosecutor.to_dict(),
        round_num=1, track="fast_track",
    )
    round_result = defender_agent.run_round(
        comment, case_analysis, persona_selection.selected_defender.to_dict(), prosecutor_argument,
        surfaced_claims, round_num=1, rag_mode=rag_mode, region=region,
        filter_target=filter_target, query_strategy=query_strategy,
    )
    return [round_result]
