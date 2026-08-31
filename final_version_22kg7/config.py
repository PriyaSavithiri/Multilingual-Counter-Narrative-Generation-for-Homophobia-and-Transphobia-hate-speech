"""
new_arch configuration.

Clean-room config for the Prosecutor/Defender/Judge architecture. This file
does NOT import anything from the old `config/`, `src/`, or `app/` trees
(per the repository restriction that new_arch must not depend at runtime on
the old application code) - it re-declares the small set of project-wide
constants (paths, languages, RAG modes, model defaults) that new_arch needs,
using the same values as the old project where those values describe
external facts (dataset locations, supported languages, hardware budget)
rather than old-architecture behaviour.

Academic honesty note (same statement as the old project, see repo root
README.md): this is a paper-inspired adaptation, not an exact replication of
any single paper.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NEW_ARCH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = NEW_ARCH_ROOT.parent  # THESIS/EDA/datasets


def _resolve_external_datasets_dir() -> Path:
    """Locate the read-only raw-dataset directory. Same resolution order as
    the old project's config/dataset_paths.py, re-implemented here (not
    imported) so new_arch has zero runtime dependency on old code. This
    directory is NEVER written to, renamed, or reorganized by new_arch."""
    def _safe_exists(p: Path) -> bool:
        try:
            return p.exists()
        except OSError:
            return False

    candidate = PROJECT_ROOT / "data_orig"
    if _safe_exists(candidate):
        return candidate
    legacy_candidate = PROJECT_ROOT / "datasets"
    if _safe_exists(legacy_candidate):
        return legacy_candidate
    sibling_candidate = PROJECT_ROOT.parent / "EDA" / "data_orig"
    if _safe_exists(sibling_candidate):
        return sibling_candidate
    return candidate


EXTERNAL_DATASETS_DIR = _resolve_external_datasets_dir()

# new_arch's own outputs live entirely under new_arch/ - never inside the
# old project's data/ or outputs/ trees, and never inside data_orig/.
DATA_DIR = NEW_ARCH_ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

OUTPUTS_DIR = NEW_ARCH_ROOT / "outputs"
GENERATIONS_DIR = OUTPUTS_DIR / "generations"
EVALUATIONS_DIR = OUTPUTS_DIR / "evaluations"
LOGS_DIR = OUTPUTS_DIR / "logs"
TRACES_DIR = OUTPUTS_DIR / "traces"
# Restricted path for Prosecutor-bearing internal traces, only ever written
# to when SAVE_INTERNAL_TRACES=true. Kept structurally separate from
# TRACES_DIR / GENERATIONS_DIR so a normal export of "results" never
# accidentally sweeps this directory up.
DEBUG_TRACES_DIR = OUTPUTS_DIR / "debug"

for _d in [KNOWLEDGE_DIR, VECTOR_STORE_DIR, GENERATIONS_DIR, EVALUATIONS_DIR,
           LOGS_DIR, TRACES_DIR, DEBUG_TRACES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VALIDATION_HOLDOUT_FRACTION = 0.15  # carved from TRAIN only, never TEST
MIN_ROWS_FOR_VALIDATION_SPLIT = 300  # below this, split is 70/30 train/test only
# ML_MTCONAN_KN's own official test split ships with every reference counter-narrative
# (KN_CN) empty across all 4 languages (verified against the real LanD-FBK/ML_MTCONAN_KN Hub
# data: 400/400 test rows, 0 non-empty KN_CN, vs. 0/1584 empty in train and 0/400 empty in
# validation) - a genuine leaderboard-style held-out test set, not a loading bug. Per explicit
# supervisor direction, dataset_loader.carve_test_from_train() carves a fresh test set with
# real ground truth out of the official TRAIN split instead, at this fraction.
ML_MTCONAN_TEST_HOLDOUT_FRACTION = 0.2

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = ["en", "ta", "eu", "es", "it"]
LANGUAGE_NAMES = {"en": "English", "ta": "Tamil", "eu": "Basque", "es": "Spanish", "it": "Italian"}

# ---------------------------------------------------------------------------
# Cultural region metadata (soft signal only - never an inferred certainty)
# ---------------------------------------------------------------------------
# Per-dataset region tag, used only to label RAG corpus provenance - this is
# metadata about WHERE A DATASET CAME FROM, not an inference about any given
# user's identity/nationality. English/Tamil CodaBench are LT-EDI shared-task
# data (predominantly Indian English/Tamil); ml_mtconan_kn is European in
# origin. These are the only 3 dataset families used anywhere in this project
# (per-language train/eval AND the RAG corpus) - see dataset_loader.py.
DATASET_CULTURAL_REGION = {
    "english_codabench": "Indian",
    "tamil_codabench": "Indian",
    "ml_mtconan_kn": "European",
}
KNOWN_REGIONS = ["Indian", "European"]  # extend if new corpora are added
# A Case Analysis region *suggestion* is downgraded to "Unknown" below this
# confidence - region is never silently inferred from language alone (see
# case_analysis_agent.py). This is a suggestion-gating threshold, not a
# retrieval-relevance threshold.
REGION_SUGGESTION_CONFIDENCE_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# RAG settings - 4 modes, ALL preserved (per explicit confirmation: no_rag,
# fact_rag, cultural_rag, dual_rag). dual_rag is the default because
# culturally-grounded generation is this project's primary proposed system,
# and dual_rag is defined to include factual retrieval/verification alongside
# cultural retrieval (cultural evidence must never replace factual grounding).
# ---------------------------------------------------------------------------
RAG_MODES = ["no_rag", "fact_rag", "cultural_rag", "dual_rag"]
DEFAULT_RAG_MODE = "dual_rag"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", 5))

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
FAISS_METADATA_PATH = VECTOR_STORE_DIR / "faiss_metadata.jsonl"
FAISS_INDEX_PATH_UNFILTERED = VECTOR_STORE_DIR / "faiss_unfiltered.index"
FAISS_METADATA_PATH_UNFILTERED = VECTOR_STORE_DIR / "faiss_metadata_unfiltered.jsonl"

# Bilingual retrieval query strategy (Defender evidence retrieval). "both" is
# the default: generate input-language queries AND semantically-equivalent
# English queries, run both, merge/dedupe/rerank. Especially relevant for
# ta/eu, where English queries improve factual recall while input-language
# queries preserve linguistically/culturally relevant evidence.
RETRIEVAL_QUERY_LANGUAGE_MODES = ["input_language", "english", "both"]
DEFAULT_RETRIEVAL_QUERY_LANGUAGE = "both"

# ---------------------------------------------------------------------------
# Persona generation - fully open-ended / instance-specific, no fixed bank.
# Generates PERSONA_CANDIDATE_COUNT candidates (default 5, clamped to
# [PERSONA_CANDIDATE_COUNT_MIN, PERSONA_CANDIDATE_COUNT_MAX]), selects exactly
# one Prosecutor persona and one Defender persona from that pool. Both
# personas are generated ONCE (before routing) and reused for every round of
# whichever track the Router selects - they do not change per round.
#
# Default lowered from 7 to 5 (the system's own pre-existing minimum, not a
# new value) after live Colab timing showed persona generation was the
# single largest per-row stage (~120s/row) - generating 7 full structured
# candidate objects (name/role_type/expertise/cultural_relevance/strategy/
# relevance_score/suited_for each) is genuinely most of that time, not an
# unused max_tokens ceiling. Still selects exactly one prosecutor + one
# defender either way; this only shrinks the candidate pool they're chosen
# from. Override with --persona-count if 7 is specifically required (e.g.
# to match an already-written methodology section).
# ---------------------------------------------------------------------------
PERSONA_CANDIDATE_COUNT_MIN = 5
PERSONA_CANDIDATE_COUNT_MAX = 10
DEFAULT_PERSONA_CANDIDATE_COUNT = 5


def clamp_persona_candidate_count(n) -> int:
    """Validate/safely correct a requested candidate count into
    [PERSONA_CANDIDATE_COUNT_MIN, PERSONA_CANDIDATE_COUNT_MAX]."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return DEFAULT_PERSONA_CANDIDATE_COUNT
    return max(PERSONA_CANDIDATE_COUNT_MIN, min(PERSONA_CANDIDATE_COUNT_MAX, n))



# ---------------------------------------------------------------------------
# Ablation study settings - full path is default; run_all_ablations intentionally
# runs only the three ablation modes so an already-running full v22kg7 job is not
# repeated.
# ---------------------------------------------------------------------------
ABLATION_MODES = ["full", "no_kg", "no_debate", "no_persona"]
ABLATION_BATCH_MODES = ["no_kg", "no_debate", "no_persona"]
ablation_mode = "full"


def set_ablation_mode(mode: str) -> str:
    """Set the process-local ablation mode for CLI/Colab runs.

    Kept as functions instead of snapshot booleans so --run_all_ablations can
    switch modes inside one Python process without re-importing modules.
    """
    global ablation_mode
    mode = mode or "full"
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unsupported ablation mode '{mode}'. Supported: {ABLATION_MODES}")
    ablation_mode = mode
    return ablation_mode


def use_kg() -> bool:
    return ablation_mode != "no_kg"


def use_kg_validation() -> bool:
    return use_kg()


def use_kg_fallback() -> bool:
    return use_kg()


def use_debate() -> bool:
    return ablation_mode != "no_debate"


def use_instance_personas() -> bool:
    return ablation_mode != "no_persona"

# ---------------------------------------------------------------------------
# Baseline methods (zero_shot/few_shot/cot/few_shot_cot) - see baseline_methods.py
# ---------------------------------------------------------------------------
NUM_FEW_SHOT_EXAMPLES = 3  # drawn from TRAIN split only, never test

# ---------------------------------------------------------------------------
# Debate / track settings
# ---------------------------------------------------------------------------
FAST_TRACK_ROUNDS = 1   # explicit hate - single Prosecutor/Defender exchange
DEEP_DIVE_ROUNDS = 3    # implicit hate - K rounds, hardcoded default for now (configurable below)


def get_deep_dive_rounds() -> int:
    """K is hardcoded to 3 by default but overridable via env var for
    experimentation, per the confirmed architecture (K=3 default)."""
    return int(os.environ.get("DEEP_DIVE_ROUNDS", DEEP_DIVE_ROUNDS))


# ---------------------------------------------------------------------------
# Prosecutor-content visibility / persistence (restricted developer mode)
# ---------------------------------------------------------------------------
# Both default to False. Prosecutor content must be unavailable from the UI,
# final counter-narrative, explanation, downloads, and standard output files
# in ALL default and normal execution modes. See safety.py for enforcement.
SHOW_INTERNAL_TRACES = os.environ.get("SHOW_INTERNAL_TRACES", "false").strip().lower() == "true"
SAVE_INTERNAL_TRACES = os.environ.get("SAVE_INTERNAL_TRACES", "false").strip().lower() == "true"

# Defaults True: batch each Defender round's per-claim query-formulation
# calls into one model call (defender_agent.py) instead of one sequential
# call per claim. Set BATCH_QUERY_FORMULATION=false to force the old
# one-call-per-claim path - both paths are timed identically (see each
# DebateRound's "timings" in the traces JSONL), so flipping this and
# re-running the same rows is how to get a real before/after comparison
# without needing two separate code checkouts.
BATCH_QUERY_FORMULATION = os.environ.get("BATCH_QUERY_FORMULATION", "true").strip().lower() == "true"

# ---------------------------------------------------------------------------
# Output format constraints (same as old project)
# ---------------------------------------------------------------------------
MIN_SENTENCES = 2
MAX_SENTENCES = 4

# ---------------------------------------------------------------------------
# Model / backend defaults (same values as old project's config/model_config.py
# - re-declared here, not imported, to keep new_arch dependency-free of the
# old config tree). Hardware target: RTX 4050 Laptop GPU (6GB VRAM), 24GB RAM,
# Intel i5-13450HX - never default to 24B/32B/70B-class models.
# ---------------------------------------------------------------------------
DEFAULT_BACKEND = "ollama"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
LIGHTWEIGHT_FALLBACK_MODEL = "llama3.2:3b"
SMOKE_TEST_MODEL = "qwen2.5:0.5b-instruct"

SUPPORTED_BACKENDS = ["ollama", "hf-inference", "hf-transformers", "openai", "auto"]

RECOMMENDED_OLLAMA_MODELS = [
    "qwen2.5:7b-instruct",
    "llama3.2:3b",
    "mistral:7b-instruct-v0.3-q4_K_M",
    "qwen3:4b",
]
LARGE_OLLAMA_MODELS_MANUAL_ONLY = [
    "mistral-small3.2:24b",
    "qwen2.5:32b-instruct",
    "qwen3:8b",
    "llama3.3:70b",
]
RECOMMENDED_HF_TRANSFORMERS_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B-Instruct",
]
RECOMMENDED_HF_INFERENCE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct:fastest",
    "Qwen/Qwen2.5-7B-Instruct:cheapest",
    "mistralai/Mistral-7B-Instruct-v0.3:fastest",
    "meta-llama/Llama-3.2-3B-Instruct:fastest",
]

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 512
USE_4BIT_QUANTIZATION = True

# Retry policy for model_api.py (new in new_arch - the old project had none;
# added because it's safe and improves robustness for long batch/eval runs).
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

# ---------------------------------------------------------------------------
# Optional web search (Tavily) - supplements local RAG evidence, never
# replaces it. Off by default. When enabled, every Defender evidence-
# gathering call runs BOTH local RAG and web search (not just as a
# fallback when local RAG is empty) - see evidence_verifier.py. Skipped
# entirely in "no_rag" mode, same as local retrieval. Every web result
# carries its source URL explicitly so the Defender/Judge can judge
# reliability rather than blending it in anonymously with local evidence.
# ---------------------------------------------------------------------------
ENABLE_WEB_SEARCH = os.environ.get("ENABLE_WEB_SEARCH", "false").strip().lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", 3))

# Restrict web search results to a curated set of domains considered reputable
# for THIS project's specific claim domain (homophobia/transphobia-related
# factual claims: sexual orientation/gender identity science, LGBTQ+ policy,
# human rights) - not a general-purpose allowlist. This is a technical filter
# applied at the Tavily API level, BEFORE the LLM ever sees a result -
# complements (does not replace) the Defender's own prompt-level credibility
# judgement (see prompts.build_defender_prompt's web_note), so a model that
# fails to apply that judgement well still can't be shown a low-quality or
# hostile source in the first place. Adjustable via WEB_SEARCH_TRUSTED_DOMAINS
# below - this list is a reasonable, reviewable default, not a claim of being
# exhaustive or the only valid set of sources.
WEB_SEARCH_RESTRICT_DOMAINS = os.environ.get("WEB_SEARCH_RESTRICT_DOMAINS", "true").strip().lower() == "true"
WEB_SEARCH_TRUSTED_DOMAINS = [
    # Medical / psychological / public health consensus
    "who.int", "apa.org", "psychiatry.org", "nih.gov", "ncbi.nlm.nih.gov", "cdc.gov",
    "thelancet.com", "nature.com", "bmj.com",
    # LGBTQ+-focused research, policy, and terminology bodies
    "williamsinstitute.law.ucla.edu", "glaad.org", "hrc.org", "ilga.org",
    "transequality.org", "stonewall.org.uk",
    # Human rights bodies
    "un.org", "ohchr.org", "echr.coe.int",
    # General encyclopedic reference
    "britannica.com",
]

# ---------------------------------------------------------------------------
# Env-driven secrets (never hardcode keys)
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Unset by default -> real OpenAI API, completely unchanged behavior. Only
# set for pointing OpenAIClient at a local OpenAI-compatible server instead
# (e.g. vLLM) - added specifically for a one-off latency validation test,
# never used in any of this project's actual reported generation runs.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or None
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
