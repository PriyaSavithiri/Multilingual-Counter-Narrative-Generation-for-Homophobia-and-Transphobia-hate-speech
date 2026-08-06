"""
Data shapes for the new_arch pipeline. Plain dataclasses (not pydantic) -
keeps the schema readable at a glance and dependency-free, per the "keep
schemas understandable, no overengineering" requirement. Each class has a
to_dict() for JSON/trace serialization.

Two families:
  - Input/output records that flow between pipeline stages.
  - Validation helpers (validate_input) that enforce the "do not silently
    accept empty input / unsupported language" rule.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import SUPPORTED_LANGUAGES

VALID_HATE_TYPES = ("explicit", "implicit")
VALID_VERDICTS = ("SUPPORTED", "REFUTED", "NEI")  # NEI = Not Enough Information
VALID_TRACKS = ("fast_track", "deep_dive")
VALID_QUERY_LANGUAGE_ORIGIN = ("input_language", "english")


class InputValidationError(ValueError):
    """Raised for empty input or an unsupported language hint - never
    silently coerced."""


@dataclass
class InputRecord:
    """The validated entry point. Text is treated as data throughout the
    pipeline, never as instructions (see safety.py's injection-guard framing
    applied wherever `text` is interpolated into a prompt)."""
    text: str
    id: Optional[str] = None
    language_hint: Optional[str] = None
    region_hint: Optional[str] = None
    rag_mode: str = "dual_rag"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_input(record: InputRecord) -> None:
    if not record.text or not record.text.strip():
        raise InputValidationError("Input text is empty.")
    if record.language_hint is not None and record.language_hint not in SUPPORTED_LANGUAGES:
        raise InputValidationError(
            f"Unsupported language_hint '{record.language_hint}'. Supported: {SUPPORTED_LANGUAGES}"
        )


@dataclass
class RegionSuggestion:
    """A Case Analysis suggestion only - never a confirmed region. The UI/
    caller must treat this as something the user can accept, override, or
    ignore (defaulting to Unknown)."""
    region: Optional[str] = None      # e.g. "Indian" / "European" / None
    country: Optional[str] = None     # e.g. "India" / None
    confidence: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_confident(self) -> bool:
        from config import REGION_SUGGESTION_CONFIDENCE_THRESHOLD
        return self.confidence >= REGION_SUGGESTION_CONFIDENCE_THRESHOLD and bool(self.region)


@dataclass
class CaseAnalysis:
    language: str
    target_group: str = "unclear"
    hate_category: str = "unclear"          # homophobia | transphobia | mixed | unclear | unsupported_category
    intent: list = field(default_factory=list)
    strategy_hint: list = field(default_factory=list)
    hate_type: str = "implicit"             # explicit | implicit - drives the Router
    confidence: float = 0.0
    hidden_claim: Optional[str] = None
    evidence_topics: list = field(default_factory=list)
    cultural_context_needed: bool = False
    region_suggestion: RegionSuggestion = field(default_factory=RegionSuggestion)
    safety_notes: list = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class PersonaCandidate:
    name: str
    role_type: str
    expertise: list = field(default_factory=list)
    cultural_relevance: str = ""
    strategy: str = ""
    relevance_score: float = 0.0
    suited_for: str = "either"  # "prosecutor" | "defender" | "either" (model's own suitability read)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectedPersona:
    name: str
    role: str            # "prosecutor" | "defender"
    objective: str
    boundaries: list = field(default_factory=list)     # prosecutor only, may be empty for defender
    expertise: list = field(default_factory=list)       # defender only, may be empty for prosecutor
    cultural_guidance: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PersonaSelection:
    candidate_personas: list  # list[PersonaCandidate]
    selected_prosecutor: SelectedPersona
    selected_defender: SelectedPersona
    selection_rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate_personas": [c.to_dict() for c in self.candidate_personas],
            "selected_prosecutor": self.selected_prosecutor.to_dict(),
            "selected_defender": self.selected_defender.to_dict(),
            "selection_rationale": self.selection_rationale,
        }


@dataclass
class EvidenceItem:
    source_id: str
    title: str
    passage: str
    retrieval_score: float
    source_type: str            # "fact" | "cultural" | "web"
    language: str = ""          # language of the retrieved passage (preserved, never translated)
    query_used: str = ""
    query_language: str = "input_language"  # "input_language" | "english"
    origin: str = "local_rag"   # "local_rag" | "web_search" - explicit provenance
    url: str = ""               # populated only for origin="web_search"; empty for local_rag

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceAssessment:
    claim: str
    verdict: str = "NEI"        # SUPPORTED | REFUTED | NEI
    evidence: list = field(default_factory=list)   # list[EvidenceItem]
    reasoning_summary: str = ""
    usable_in_final_cn: bool = False

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "evidence": [e.to_dict() for e in self.evidence],
            "reasoning_summary": self.reasoning_summary,
            "usable_in_final_cn": self.usable_in_final_cn,
        }


@dataclass
class DebateRound:
    round: int
    round_objective: str = ""
    prosecutor_argument: str = ""          # INTERNAL ONLY - never surfaced to final output/UI normal view
    surfaced_claims: list = field(default_factory=list)
    retrieval_queries: list = field(default_factory=list)
    evidence_assessments: list = field(default_factory=list)  # list[EvidenceAssessment]
    defender_response: str = ""
    new_information: list = field(default_factory=list)
    unresolved_questions: list = field(default_factory=list)
    cultural_notes: list = field(default_factory=list)
    safety_notes: list = field(default_factory=list)

    def to_dict(self, include_prosecutor: bool = True) -> dict:
        d = {
            "round": self.round,
            "round_objective": self.round_objective,
            "surfaced_claims": self.surfaced_claims,
            "retrieval_queries": self.retrieval_queries,
            "evidence_assessments": [e.to_dict() for e in self.evidence_assessments],
            "defender_response": self.defender_response,
            "new_information": self.new_information,
            "unresolved_questions": self.unresolved_questions,
            "cultural_notes": self.cultural_notes,
            "safety_notes": self.safety_notes,
        }
        if include_prosecutor:
            d["prosecutor_argument"] = self.prosecutor_argument
        return d


@dataclass
class RouterDecision:
    selected_track: str       # "fast_track" | "deep_dive"
    confidence: float
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RejectedContent:
    content: str
    reason: str  # unsupported | unsafe | irrelevant | culturally_inappropriate | repetitive

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgePlan:
    selected_language: str
    core_claim_to_counter: str = ""
    recommended_strategy: list = field(default_factory=list)
    approved_evidence: list = field(default_factory=list)     # list[EvidenceItem]
    rejected_content: list = field(default_factory=list)      # list[RejectedContent]
    cultural_guidance: list = field(default_factory=list)
    safety_guidance: list = field(default_factory=list)
    final_response_plan: str = ""

    def to_dict(self) -> dict:
        return {
            "selected_language": self.selected_language,
            "core_claim_to_counter": self.core_claim_to_counter,
            "recommended_strategy": self.recommended_strategy,
            "approved_evidence": [e.to_dict() for e in self.approved_evidence],
            "rejected_content": [r.to_dict() for r in self.rejected_content],
            "cultural_guidance": self.cultural_guidance,
            "safety_guidance": self.safety_guidance,
            "final_response_plan": self.final_response_plan,
        }


@dataclass
class FinalOutput:
    input_id: Optional[str]
    language: str
    hate_type: str
    selected_track: str
    rag_mode: str
    counter_narrative: str
    explanation: str
    evidence_trace: list = field(default_factory=list)  # list of {"claim","verdict","source_ids"}
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
