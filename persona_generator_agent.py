"""
Persona Generator Agent - fully open-ended, instance-specific. No fixed
persona bank of any kind: every candidate is invented fresh for this one
comment. Generates config.DEFAULT_PERSONA_CANDIDATE_COUNT candidates by
default (clamped to [5, 10]), selects exactly one Prosecutor persona and one
Defender persona from that pool. Called ONCE, before routing - both tracks
reuse the same two selected personas for every round (persona consistency).
"""
import re

import config
import prompts
from schemas import PersonaCandidate, SelectedPersona, PersonaSelection
from utils import as_dict_item, get_logger

logger = get_logger("new_arch.persona_generator_agent")

_REQUIRED_FIELDS = ["candidate_personas", "selected_prosecutor", "selected_defender", "selection_rationale"]

# Deterministic guard against personal-name-shaped output: the prompt
# instructs the model to give a role title (e.g. "forensic linguist"), not
# a personal name, but models sometimes ignore this and invent "Dr. Jane
# Smith"-style names anyway - which is risky (could coincidentally resemble
# a real person) and violates the "professional/expert framing, never
# identity impersonation" principle this project relies on. This regex
# catches an honorific followed by 1-3 capitalized words and forces a
# fallback to the role_type field instead, so a personal-name persona can
# never surface even if the LLM ignores the prompt's instruction.
_PERSONAL_NAME_RE = re.compile(
    r"^(dr\.?|prof(essor)?\.?|rev(erend)?\.?|mr\.?|mrs\.?|ms\.?|miss|sir|dame|fr\.?)\s+"
    r"[a-z]+(\s+[a-z]+){0,2}$",
    re.IGNORECASE,
)


def _sanitize_persona_name(name: str, role_type: str, index: int) -> str:
    """Returns `name` unchanged unless it looks like a personal name (with
    or without an honorific), in which case it falls back to `role_type`
    (or a generic numbered placeholder if that's empty too)."""
    name = (name or "").strip()
    looks_personal = bool(_PERSONAL_NAME_RE.match(name)) or (
        # Two capitalized words with no role-describing noun and no honorific
        # (e.g. "Michael Green") is also treated as a personal name.
        bool(re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", name)) and role_type.strip().lower() not in name.lower()
    )
    if not looks_personal:
        return name or (role_type.strip() or f"expert persona {index}")
    logger.warning("Persona name '%s' looked like a personal name, not a role title - replaced with role_type.", name)
    return role_type.strip() or f"expert persona {index}"


def _fallback_selection(candidate_count: int) -> PersonaSelection:
    """Only used if the LLM output fails to parse even after model_api's
    built-in retries - a minimal, safe, generic pair so the pipeline can
    still proceed rather than crash."""
    candidates = [PersonaCandidate(name=f"generalist expert {i+1}", role_type="generalist",
                                    relevance_score=0.0) for i in range(candidate_count)]
    prosecutor = SelectedPersona(name="rhetoric analyst", role="prosecutor",
                                  objective="Identify the strongest internal logic of the harmful claim.",
                                  boundaries=["stay internal-only", "no gratuitous slurs", "no threats"])
    defender = SelectedPersona(name="inclusive communication specialist", role="defender",
                                objective="Rebut the claim with empathy, dignity, and available evidence.")
    return PersonaSelection(candidate_personas=candidates, selected_prosecutor=prosecutor,
                             selected_defender=defender,
                             selection_rationale="Fallback selection - persona generation failed to parse.")


class PersonaGeneratorAgent:
    def __init__(self, client):
        self.client = client

    def run(self, comment: str, case_analysis, candidate_count: int = None) -> PersonaSelection:
        candidate_count = config.clamp_persona_candidate_count(
            candidate_count if candidate_count is not None else config.DEFAULT_PERSONA_CANDIDATE_COUNT
        )
        prompt = prompts.build_persona_generation_prompt(comment, case_analysis.to_dict(), candidate_count)
        parsed = self.client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.5, max_tokens=1400)

        if not parsed:
            logger.warning("Persona generation parse failure - using minimal fallback pair.")
            return _fallback_selection(candidate_count)

        candidates = []
        name_lookup = {}  # raw LLM-provided name -> sanitized name, for selected_prosecutor/defender below
        for idx, c in enumerate(parsed.get("candidate_personas") or [], start=1):
            # Defensive: a weaker model may return this list as plain strings
            # rather than structured objects - coerce rather than crash.
            c = as_dict_item(c, fallback_key="name")
            try:
                score = float(c.get("relevance_score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            raw_name = c.get("name") or ""
            role_type = c.get("role_type") or ""
            sanitized_name = _sanitize_persona_name(raw_name, role_type, idx)
            if raw_name:
                name_lookup[raw_name] = sanitized_name
            candidates.append(PersonaCandidate(
                name=sanitized_name,
                role_type=role_type,
                expertise=c.get("expertise") or [],
                cultural_relevance=c.get("cultural_relevance") or "",
                strategy=c.get("strategy") or "",
                relevance_score=score,
                suited_for=c.get("suited_for") or "either",
            ))

        sp = parsed.get("selected_prosecutor") or {}
        sd = parsed.get("selected_defender") or {}
        # Prefer the already-sanitized candidate name if this selection's raw
        # name matches one of the candidates; otherwise sanitize independently
        # (falls back to a generic "prosecutor persona"/"defender persona"
        # since there's no role_type field on a SelectedPersona to fall back to).
        sp_raw_name = sp.get("name") or ""
        sd_raw_name = sd.get("name") or ""
        sp_name = name_lookup.get(sp_raw_name) or _sanitize_persona_name(sp_raw_name, "", 0) or "prosecutor persona"
        sd_name = name_lookup.get(sd_raw_name) or _sanitize_persona_name(sd_raw_name, "", 0) or "defender persona"
        selected_prosecutor = SelectedPersona(
            name=sp_name, role="prosecutor",
            objective=sp.get("objective") or "", boundaries=sp.get("boundaries") or [],
        )
        selected_defender = SelectedPersona(
            name=sd_name, role="defender",
            objective=sd.get("objective") or "", expertise=sd.get("expertise") or [],
            cultural_guidance=sd.get("cultural_guidance") or [],
        )

        if not candidates:
            candidates = [
                PersonaCandidate(name=selected_prosecutor.name, role_type="prosecutor", relevance_score=1.0,
                                  suited_for="prosecutor"),
                PersonaCandidate(name=selected_defender.name, role_type="defender", relevance_score=1.0,
                                  suited_for="defender"),
            ]

        return PersonaSelection(
            candidate_personas=candidates, selected_prosecutor=selected_prosecutor,
            selected_defender=selected_defender, selection_rationale=parsed.get("selection_rationale") or "",
        )
