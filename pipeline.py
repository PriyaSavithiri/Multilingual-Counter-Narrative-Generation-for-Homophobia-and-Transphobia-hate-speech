"""
new_arch pipeline orchestration - the single place the full flow is visible:

  Input -> Case Analysis -> Persona Generator (once, open-ended) -> Router
  -> Fast-Track (1 round) / Deep-Dive (K rounds), Prosecutor<->Defender with
  evidence verification folded into every Defender turn -> Judge/Critic
  (single synthesis) -> Final Counter-Narrative + Explanation.

No neutral/without-persona branch anywhere. Prosecutor content is
unconditionally redacted from the returned trace (safety.public_trace) -
the only way to see it is the explicitly-gated, clearly-marked "_debug" key
(present only when config.SHOW_INTERNAL_TRACES=True, and even then limited
to a short excerpt, never the full text - see safety.debug_summary).
"""
import time
import traceback

import config
import fast_track
import deep_dive_track
import safety
from case_analysis_agent import CaseAnalysisAgent
from persona_generator_agent import PersonaGeneratorAgent
from prosecutor_agent import ProsecutorAgent
from defender_agent import DefenderAgent
from judge_critic_agent import JudgeCriticAgent
from final_cn_agent import FinalCNAgent
from router import route
from schemas import InputRecord, validate_input, FinalOutput
from utils import get_logger

logger = get_logger("new_arch.pipeline")


class NewArchPipeline:
    def __init__(self, client, backend_name: str = "", model_name: str = ""):
        self.client = client
        self.backend_name = backend_name
        self.model_name = model_name
        self.case_analysis_agent = CaseAnalysisAgent(client)
        self.persona_generator_agent = PersonaGeneratorAgent(client)
        self.prosecutor_agent = ProsecutorAgent(client)
        self.defender_agent = DefenderAgent(client)
        self.judge_critic_agent = JudgeCriticAgent(client)
        self.final_cn_agent = FinalCNAgent(client)

    def generate(self, input_record: InputRecord, accept_region_suggestion: bool = False,
                 persona_candidate_count: int = None, deep_dive_rounds: int = None,
                 query_strategy: str = None, filter_target: bool = True,
                 progress_callback=None) -> dict:
        report = progress_callback or (lambda stage: None)
        validate_input(input_record)
        rag_mode = input_record.rag_mode or config.DEFAULT_RAG_MODE
        query_strategy = query_strategy or config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE
        started = time.time()
        errors = []

        case_analysis = None
        persona_selection = None
        router_decision = None
        debate_rounds = []
        judge_plan = None
        confirmed_region = input_record.region_hint if input_record.region_hint not in (
            None, "", "Auto / Unknown", "unknown", "Unknown") else None

        try:
            report("Analyzing the comment (Case Analysis)...")
            case_analysis = self.case_analysis_agent.run(
                input_record.text, language_hint=input_record.language_hint, region_hint=input_record.region_hint,
            )
            confirmed_region = self._resolve_region(input_record, case_analysis, accept_region_suggestion)

            report("Generating instance-specific Prosecutor/Defender personas...")
            persona_selection = self.persona_generator_agent.run(
                input_record.text, case_analysis, candidate_count=persona_candidate_count,
            )

            report("Routing (explicit hate -> Fast-Track, implicit hate -> Deep-Dive)...")
            router_decision = route(case_analysis)

            report(f"Running {router_decision.selected_track} debate (Prosecutor <-> Defender)...")
            if router_decision.selected_track == "fast_track":
                debate_rounds = fast_track.run(
                    input_record.text, case_analysis, persona_selection, self.prosecutor_agent,
                    self.defender_agent, rag_mode, confirmed_region, filter_target, query_strategy,
                )
            else:
                debate_rounds = deep_dive_track.run(
                    input_record.text, case_analysis, persona_selection, self.prosecutor_agent,
                    self.defender_agent, rag_mode, confirmed_region, filter_target, query_strategy,
                    k=deep_dive_rounds,
                )

            report("Judge/Critic synthesizing the final response plan...")
            judge_plan = self.judge_critic_agent.run(
                input_record.text, case_analysis, persona_selection, router_decision, debate_rounds, confirmed_region,
            )

            report("Writing final counter-narrative + explanation...")
            final = self.final_cn_agent.run(input_record.text, judge_plan, confirmed_region)
        except Exception as exc:
            logger.exception("Pipeline stage failed")
            errors.append(f"{exc}\n{traceback.format_exc(limit=3)}")
            final = {"counter_narrative": "", "explanation": "", "safety_flags": []}

        safety.maybe_save_debug_trace(input_record.id, debate_rounds, extra={
            "case_analysis": case_analysis.to_dict() if case_analysis else {},
            "router_decision": router_decision.to_dict() if router_decision else {},
        })

        output = FinalOutput(
            input_id=input_record.id,
            # case_analysis.language is the authoritative source (cross-checked against
            # detect_language()'s script/marker heuristics) - never judge_plan.selected_language,
            # which is an LLM's own free-form guess and has been observed defaulting to
            # English regardless of the actual input language (see judge_critic_agent.py).
            language=(case_analysis.language if case_analysis else
                      (judge_plan.selected_language if judge_plan else input_record.language_hint)),
            hate_type=case_analysis.hate_type if case_analysis else None,
            selected_track=router_decision.selected_track if router_decision else None,
            rag_mode=rag_mode,
            counter_narrative=final["counter_narrative"],
            explanation=final["explanation"],
            evidence_trace=[
                {"claim": ea.claim, "verdict": ea.verdict, "source_ids": [e.source_id for e in ea.evidence]}
                for r in debate_rounds for ea in r.evidence_assessments
            ],
            metadata={
                "backend": self.backend_name, "model": self.model_name,
                "duration_seconds": round(time.time() - started, 2),
                "persona_prosecutor": persona_selection.selected_prosecutor.name if persona_selection else None,
                "persona_defender": persona_selection.selected_defender.name if persona_selection else None,
                "region_confirmed": confirmed_region,
                "region_suggestion": case_analysis.region_suggestion.to_dict() if case_analysis else None,
                "query_strategy": query_strategy,
                "filter_target": filter_target,
                "final_cn_safety_flags": final.get("safety_flags", []),
                "errors": errors,
            },
        )

        report("Done.")

        trace = output.to_dict()
        trace["case_analysis"] = case_analysis.to_dict() if case_analysis else {}
        trace["persona_selection"] = persona_selection.to_dict() if persona_selection else {}
        trace["router_decision"] = router_decision.to_dict() if router_decision else {}
        trace["debate_rounds"] = safety.public_debate_rounds(debate_rounds)
        trace["judge_plan"] = judge_plan.to_dict() if judge_plan else {}

        if config.SHOW_INTERNAL_TRACES:
            # Restricted, opt-in developer view only - a concise structured
            # summary (surfaced claims + a short excerpt), never the full
            # unrestricted Prosecutor reasoning text or any hidden prompt.
            trace["_debug"] = {"warning": safety.DEBUG_WARNING_TEXT, "rounds": safety.debug_summary(debate_rounds)}

        return safety.public_trace(trace)

    @staticmethod
    def _resolve_region(input_record: InputRecord, case_analysis, accept_region_suggestion: bool):
        """Priority: manual user selection > trusted dataset metadata >
        explicitly-accepted Case Analysis suggestion > None ("unknown").
        A region is NEVER silently inferred or applied from language alone."""
        hint = input_record.region_hint
        if hint and hint not in (None, "", "Auto / Unknown", "unknown", "Unknown"):
            return hint
        dataset_region = (input_record.metadata or {}).get("dataset_region")
        if dataset_region:
            return dataset_region
        if accept_region_suggestion and case_analysis.region_suggestion.is_confident:
            return case_analysis.region_suggestion.region
        return None
