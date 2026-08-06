"""
Prosecutor Agent - internal-only, adversarial stress-test role. Adopts
whichever open-ended persona persona_generator_agent.py selected (generated
once, reused every round). Genuinely constructs the harmful claim's
strongest internal logic - a deliberate, explicit, narrowly-scoped exception
to "never argue for the hateful claim", strictly internal and never surfaced
to the end user (see safety.py / prompts.py module docstrings).
"""
import prompts
from utils import get_logger

logger = get_logger("new_arch.prosecutor_agent")

_REQUIRED_FIELDS = ["round", "round_objective", "prosecutor_argument", "surfaced_claims"]


class ProsecutorAgent:
    def __init__(self, client):
        self.client = client

    def run_round(self, comment: str, case_analysis, prosecutor_persona: dict, round_num: int,
                   track: str, prior_rounds_summary: str = ""):
        """Returns (prosecutor_argument: str, surfaced_claims: list[str])."""
        prompt = prompts.build_prosecutor_prompt(
            comment, case_analysis.to_dict(), prosecutor_persona, round_num, track, prior_rounds_summary
        )
        # max_tokens=800 (was 500) - found via live testing: qwen3:8b is
        # noticeably more verbose than the other models tested this project
        # (llama3.1:8b, qwen2.5:7b-instruct, mistral:7b-instruct-v0.3) and
        # was getting hard-truncated mid-JSON at 500 tokens on this schema
        # (prosecutor_argument + a multi-item surfaced_claims list), causing
        # every parse attempt - including model_api.py's own corrective
        # retry, which reuses this same budget - to fail, since retrying
        # can't fix a truncation problem the way it fixes a formatting one.
        parsed = self.client.generate(prompt, response_schema=_REQUIRED_FIELDS, temperature=0.4, max_tokens=800)
        if not parsed:
            logger.warning("Prosecutor round %d parse failure - proceeding with no surfaced claims.", round_num)
            return "", []
        return parsed.get("prosecutor_argument") or "", parsed.get("surfaced_claims") or []
