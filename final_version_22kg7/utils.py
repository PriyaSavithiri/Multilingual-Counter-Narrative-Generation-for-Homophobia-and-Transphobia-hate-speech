"""
Shared low-level helpers for new_arch: tolerant JSON parsing, text/format
checks, file I/O, and run logging. Consolidated into one file (per the
approved new_arch file tree) rather than the old project's 4-file split,
since none of these pieces is large enough to justify its own module.
"""
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import config

# Windows consoles/terminals often default to a legacy code page (e.g.
# cp1252) rather than UTF-8. Any print()/logging call that happens to
# contain a character outside that code page (a Unicode arrow in a status
# message, a Tamil/Basque/accented character echoed from user input into a
# log line, etc.) would otherwise raise UnicodeEncodeError and crash the
# entire run - including inside logging's own exception handler, which is
# especially unhelpful since it then masks the real underlying error. This
# makes stdout/stderr degrade gracefully (unencodable characters are
# backslash-escaped instead of crashing) rather than fixing every string at
# the source - defense in depth, not a substitute for using ASCII-safe
# strings in print()/report() calls where practical.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# Tolerant JSON parsing - failures return {} / [] rather than raising, so a
# malformed LLM reply never crashes the pipeline. Callers must check for
# emptiness explicitly.
# ---------------------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
FINAL_CN_MARKER_RE = re.compile(r"counter-narrative\s*:\s*", re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    """Strips <think>...</think> reasoning blocks that some models (e.g.
    Qwen3's default HF chat template) prepend before their real answer.
    Found via live testing on Colab: Qwen3-8B via the hf-transformers
    backend emits a reasoning block by default (unlike the Ollama build of
    the same tag, whose template evidently disables thinking, which is why
    this never surfaced in local runs) - the reasoning text itself often
    contains stray '{'/'}' characters (discussing JSON structure), which
    broke the naive first-'{'/last-'}' bracket-extraction fallback below,
    causing spurious Case-Analysis parse failures. Also strips an unclosed
    trailing <think> block (no </think> at all) - that means generation was
    truncated mid-reasoning with no real answer ever produced, so nothing
    after it is recoverable anyway."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text


def _strip_code_fence(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def parse_json_object(raw_output: str) -> dict:
    """Always returns a dict (per the type hint) - never a list/str/number,
    even though json.loads() would happily parse any valid JSON value. Found
    via live testing: for one real judge call, qwen2.5:7b-instruct returned a
    flat JSON ARRAY (["relevance", 4, "factuality", 3, ...]) instead of an
    object. json.loads() parsed it fine, and the caller's schema check (`k in
    parsed for k in response_schema`) uses Python's `in` operator, which does
    *membership* on a list rather than key lookup - since the array happened
    to contain all the expected field-name strings as elements, the check
    passed by accident, silently corrupting downstream code that assumed a
    dict (pandas turned the list into positionally-indexed judge_0/judge_1/...
    columns instead of named ones). The isinstance checks below close that
    loophole at the source, so every caller of this function is protected."""
    if not raw_output:
        return {}
    text = _strip_code_fence(_strip_think_blocks(raw_output))
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def parse_jsonl_objects(raw_output: str) -> list:
    """One JSON object per line (used for persona-candidate list prompts).
    Falls back to parsing the whole text as a JSON array if no line parses."""
    if not raw_output:
        return []
    text = _strip_code_fence(raw_output)
    objects = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if objects:
        return objects
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def as_dict_item(item, fallback_key: str = None) -> dict:
    """Defensive coercion for a single element of a parsed LLM JSON list
    field (e.g. approved_evidence, claim_assessments, candidate_personas).
    Weaker models sometimes return a list of plain strings where a list of
    structured objects was expected (observed with mistral:7b-instruct-v0.3
    returning approved_evidence as bare strings) - calling .get() on a
    string crashes with AttributeError. This wraps a non-dict item into a
    minimal dict (its string value stored under fallback_key) so callers
    can keep using .get() uniformly instead of crashing; pass
    fallback_key=None to just drop non-dict items (returns {})."""
    if isinstance(item, dict):
        return item
    if item and fallback_key:
        return {fallback_key: str(item)}
    return {}


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _flatten_to_text(value) -> str:
    """Coerces an anchor value into a plain string for shares_key_terms(),
    defensively. Found via live testing: DebateRound.surfaced_claims (one
    source of anchor_texts) is supposed to be a list of strings, but a real
    model returned a nested list for one "claim" instead - crashing
    _WORD_RE.findall() with `TypeError: expected string or bytes-like
    object, got 'list'` deep inside a real run. Rather than chase every
    possible producer of a malformed surfaced_claims entry, this makes the
    safety-net function itself tolerant of whatever shape actually shows up,
    consistent with this project's as_dict_item() pattern elsewhere."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_to_text(v) for v in value)
    if value is None:
        return ""
    return str(value)


def shares_key_terms(text: str, anchor_texts, min_word_len: int = 4) -> bool:
    """Language-agnostic topical-relevance check (Python's \\w is Unicode-
    aware by default, so this needs no per-language tokenizer - works
    identically for en/ta/eu/es/it): does `text` share at least one
    substantive word (>= min_word_len characters) with any of
    `anchor_texts`? A single shared word is enough to pass - this is
    calibrated to catch gross topic drift/hallucination (a model describing
    a completely unrelated scenario - observed live: a Judge call that,
    under heavy Deep-Dive context, produced a customer-service/product-
    safety FAQ instead of reasoning about the actual hate-speech case), not
    to enforce close paraphrasing of the source. Returns False (fails the
    check) if `text` has no substantive words at all - an empty/near-empty
    response is exactly the case this should not wave through."""
    text_words = set(w.lower() for w in _WORD_RE.findall(_flatten_to_text(text)) if len(w) >= min_word_len)
    if not text_words:
        return False
    anchor_words = set()
    for anchor in anchor_texts:
        anchor_words.update(w.lower() for w in _WORD_RE.findall(_flatten_to_text(anchor)) if len(w) >= min_word_len)
    return bool(text_words & anchor_words)


def extract_final_counter_narrative(raw_output: str) -> str:
    if not raw_output:
        return ""
    parts = FINAL_CN_MARKER_RE.split(raw_output)
    return parts[-1].strip() if len(parts) > 1 else raw_output.strip()


# ---------------------------------------------------------------------------
# Text / format checks
# ---------------------------------------------------------------------------
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")

# Intentionally partial, coarse safety heuristic - NOT a substitute for the
# Judge/Critic's LLM-based safety review. Only flags obviously unsafe
# generations for the automatic-metrics report (and, via rag_pipeline.py's
# corpus quality filter, for excluding unsafe RAG corpus entries at build time).
UNSAFE_KEYWORDS_EN = [
    "kill", "disgusting", "abomination", "sin", "unnatural", "sick people",
    "pedophile", "pedo", "groomer", "deviant", "confused freaks",
]

# VALIDATED unsafe-keyword lists, one per language - "validated" meaning
# someone has actually checked these words are safe to match in that
# language (not just "the English list happens to also sort of apply").
# Only "en" has one right now. es/it/eu/ta deliberately have NO entry here -
# not because they're assumed safe, but because no one has built and
# validated a keyword list for them yet (see scripts/extract_unsafe_candidates.py
# and scripts/validate_unsafe_candidates.py, and data/unsafe_patterns_by_lang.json,
# for the dataset-driven process that will eventually populate these).
# contains_unsafe_keywords() returns None - not False - for any language
# with no entry here, and callers must not treat None as "checked and safe".
VALIDATED_UNSAFE_KEYWORDS = {
    "en": UNSAFE_KEYWORDS_EN,
}

# English regional variants (en_EUR, en_IN) fall back to the base "en" list
# when no validated regional-specific list exists yet - a documented
# limitation, not a claim that European and Indian English hate speech use
# identical vocabulary. Revisit once/if a region-specific validated list
# exists (same dataset-driven process as above, split by region metadata).
_ENGLISH_REGIONAL_FALLBACK = {"en_EUR": "en", "en_IN": "en"}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_evidence_text(text: str) -> str:
    """Strips literal '<EOS>' sentence-delimiter markers left over from the
    source corpus's own preprocessing (confirmed present in ~10% of
    knowledge_text records) before evidence text is ever shown to the
    Defender or Judge - this is display-time cleaning only, applied where an
    EvidenceItem's passage is built (evidence_verifier.py); the underlying
    corpus.jsonl file/dataset is never modified. Replaced with a period +
    space (a real sentence break, matching what '<EOS>' actually marked),
    not just deleted - deleting it would run two sentences together instead
    of separating them."""
    if not text:
        return text or ""
    cleaned = re.sub(r"<EOS>", ". ", text, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned)


def count_words(text: str) -> int:
    return len((text or "").strip().split())


def count_sentences(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    return max(1, len(SENTENCE_SPLIT_RE.split(text)))


def is_format_compliant(text: str, min_sentences: int = None, max_sentences: int = None) -> bool:
    min_sentences = config.MIN_SENTENCES if min_sentences is None else min_sentences
    max_sentences = config.MAX_SENTENCES if max_sentences is None else max_sentences
    n = count_sentences(text)
    return min_sentences <= n <= max_sentences


def contains_unsafe_keywords(text: str, language: str) -> Optional[bool]:
    """Returns True (checked, unsafe keyword/pattern found), False (checked,
    none found), or None (no validated keyword list exists for this
    language - this is NOT the same as "checked and safe", and callers must
    never silently treat it as False; see evaluation.py's unsafe_keyword_rate,
    which explicitly excludes None rows rather than counting them as safe).

    language is required, not defaulted - v20 previously defaulted to "en"
    silently, which is exactly the bug this redesign fixes (English keywords
    getting applied to non-English text by accident). en_EUR/en_IN fall back
    to the base "en" list (documented limitation, see
    _ENGLISH_REGIONAL_FALLBACK); es/it/eu/ta have no validated list yet and
    always return None until scripts/extract_unsafe_candidates.py +
    scripts/validate_unsafe_candidates.py populate
    data/unsafe_patterns_by_lang.json for them.

    Word-boundary matching (re.UNICODE, on by default for str patterns in
    Python 3), not a bare substring check - catches real words only, not
    "sin" appearing inside "using"/"cousin"/"since". Not extended to
    non-word-boundary phrase matching for Tamil here, since there is no
    validated Tamil list yet for that distinction to matter - revisit when
    one exists (see module docstring above)."""
    resolved_language = _ENGLISH_REGIONAL_FALLBACK.get(language, language)
    keywords = VALIDATED_UNSAFE_KEYWORDS.get(resolved_language)
    if keywords is None:
        return None
    lowered = (text or "").lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in keywords)


_UNSAFE_PATTERNS_PATH = config.DATA_DIR / "unsafe_patterns_by_lang.json"
_unsafe_patterns_cache = None


def load_unsafe_patterns(path=None) -> dict:
    """Loads data/unsafe_patterns_by_lang.json (Stage 6 of the v22 unsafe-
    keyword redesign) - the curated, per-language table of terms that
    actually passed real LLM/human validation (scripts/validate_unsafe_
    candidates.py + scripts/build_unsafe_patterns_json.py), as opposed to
    utils.VALIDATED_UNSAFE_KEYWORDS (a separate, pre-existing, hand-curated
    English-only list). Returns {} if the file doesn't exist rather than
    raising - callers must treat a missing/empty entry as non-evaluable
    (see get_validated_unsafe_patterns), never as "checked and safe".
    Cached only for the default path; an explicit path always re-reads,
    which is what tests need."""
    global _unsafe_patterns_cache
    p = Path(path) if path else _UNSAFE_PATTERNS_PATH
    if path is None and _unsafe_patterns_cache is not None:
        return _unsafe_patterns_cache
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if path is None:
        _unsafe_patterns_cache = data
    return data


def get_validated_unsafe_patterns(language: str, path=None) -> Optional[list]:
    """Returns the list of curated pattern dicts for `language` from
    data/unsafe_patterns_by_lang.json, or None if that language has no
    validated (non-empty) entry yet - mirrors contains_unsafe_keywords()'s
    Optional[bool] contract: None means "not evaluable yet", never "checked,
    found nothing". A language stays None until real validation (not a
    stubbed test client) has been run and promoted into the JSON file."""
    entry = load_unsafe_patterns(path).get(language)
    if not entry or entry.get("status") != "validated" or not entry.get("terms"):
        return None
    return entry["terms"]


# Negation/rebuttal cue words for Stage 7's context-aware matching - English
# only, since English is the only language with any evaluable term list
# right now (VALIDATED_UNSAFE_KEYWORDS["en"]); non-English negation isn't
# implemented until a language actually has validated terms to check
# negation around in the first place (see data/unsafe_patterns_by_lang.json).
_NEGATION_CUES_EN = {
    "not", "isn't", "aren't", "wasn't", "weren't", "never", "no",
    "cannot", "can't", "won't", "doesn't", "didn't", "nobody", "none",
}
_NEGATION_WINDOW_WORDS = 4  # how many words before a match count as "negating" it


def _is_negated_context(lowered_text: str, match_start: int) -> bool:
    """True if a negation cue appears within _NEGATION_WINDOW_WORDS words
    immediately before the match, WITHIN THE SAME SENTENCE - a negation in a
    previous sentence must not suppress a genuinely unsafe claim in the next
    one (e.g. "It's not like that. They are an abomination." must still
    flag). `lowered_text` must already be lowercased (caller's
    responsibility, so this never re-lowercases a large string per match)."""
    preceding = lowered_text[:match_start]
    boundary = max(preceding.rfind(". "), preceding.rfind("! "), preceding.rfind("? "), preceding.rfind("\n"))
    window_text = preceding[boundary + 2:] if boundary != -1 else preceding
    words = _WORD_RE.findall(window_text)
    return any(w in _NEGATION_CUES_EN for w in words[-_NEGATION_WINDOW_WORDS:])


def check_unsafe_keywords_with_context(text: str, language: str) -> dict:
    """Stage 7 of the v22 unsafe-keyword redesign: a structured, context-
    aware check for GENERATED counter-narrative text, layered on top of
    contains_unsafe_keywords()'s language-validation contract (same
    resolved-language/fallback rules, same term sources: utils.
    VALIDATED_UNSAFE_KEYWORDS plus anything promoted into data/
    unsafe_patterns_by_lang.json via get_validated_unsafe_patterns).

    A counter-narrative that QUOTES or NEGATES a harmful claim to rebut it
    (e.g. "Being gay is not a disease.", "That's not an abomination, it's a
    family.") is not the same as endorsing it - auto-flagging every raw
    keyword hit would penalize exactly the rebuttal writing this whole
    project is trying to produce. So a match inside a negated context does
    NOT set flag=True; it sets requires_review=True instead, for a human (or
    a later LLM pass) to judge intent, rather than guessing either way.

    Returns a dict:
      evaluable: bool - False if no validated term list exists at all for
        this language (mirrors contains_unsafe_keywords() returning None)
      flag: Optional[bool] - True only if at least one NON-negated match was
        found; False if evaluable but no unsafe (non-negated) match; None
        if not evaluable
      matches: list[str] - every distinct matched term, negated or not
      categories: list[str] - distinct known categories of matched terms
        (from data/unsafe_patterns_by_lang.json; empty for terms with no
        category, e.g. the hand-curated English list)
      requires_review: bool - True if any match was in a negated/rebuttal
        context and therefore was NOT auto-flagged
      language_scope: str - the resolved language whose HAND-CURATED list
        (utils.VALIDATED_UNSAFE_KEYWORDS) was used (e.g. "en" for "en_EUR"
        input, via the same fallback contains_unsafe_keywords() uses). The
        curated JSON lookup (see below) is scoped separately and more
        precisely, since it's not subject to that same limitation.
    """
    resolved_language = _ENGLISH_REGIONAL_FALLBACK.get(language, language)
    validated_keywords = VALIDATED_UNSAFE_KEYWORDS.get(resolved_language)
    # Curated JSON patterns (Stage 6) are looked up under the EXACT input
    # language first, not the fallback-resolved one - extract/filter/
    # validate_unsafe_candidates.py all treat "en", "en_EUR", "en_IN" as
    # three separate candidate pools with their own validated terms (e.g.
    # "en_EUR" has real validated entries "en" itself doesn't), so
    # collapsing en_EUR -> "en" here (as _ENGLISH_REGIONAL_FALLBACK does for
    # the single hand-curated English list) would silently make an entire
    # region's validated terms unreachable. Falls back to the resolved base
    # language only if the region-specific bucket has no validated terms yet.
    curated_patterns = get_validated_unsafe_patterns(language)
    if curated_patterns is None and language != resolved_language:
        curated_patterns = get_validated_unsafe_patterns(resolved_language)
    if validated_keywords is None and curated_patterns is None:
        return {"evaluable": False, "flag": None, "matches": [], "categories": [],
                "requires_review": False, "language_scope": resolved_language}

    term_categories = {kw: None for kw in (validated_keywords or [])}
    for p in (curated_patterns or []):
        term_categories[p["term"]] = p.get("category")

    lowered = (text or "").lower()
    matches, categories = [], []
    requires_review = False
    unsafe_found = False
    for term, category in term_categories.items():
        for m in re.finditer(rf"\b{re.escape(term.lower())}\b", lowered):
            matches.append(term)
            if category:
                categories.append(category)
            if _is_negated_context(lowered, m.start()):
                requires_review = True
            else:
                unsafe_found = True

    return {
        "evaluable": True,
        "flag": unsafe_found,
        "matches": sorted(set(matches)),
        "categories": sorted(set(categories)),
        "requires_review": requires_review,
        "language_scope": resolved_language,
    }


def is_exact_copy(generated: str, reference: str) -> bool:
    if not generated or not reference:
        return False
    return generated.strip().lower() == reference.strip().lower()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def require_file(path, friendly_name: str = None) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{friendly_name or 'Required file'} not found at {p}. This project never "
            "auto-downloads or fabricates dataset content - place the file there or "
            "point EXTERNAL_DATASETS_DIR at the right location."
        )
    return p


def write_jsonl(records: list, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Language detection (5 supported languages) - clean-room re-implementation
# of the same heuristics as the old project's src/language/detect_language.py:
# trusted hint first, then Tamil Unicode range, then Basque marker words,
# then langdetect (seeded for determinism), else fall back to English.
# ---------------------------------------------------------------------------
_TAMIL_RANGE_RE = re.compile(r"[஀-௿]")
_BASQUE_MARKERS = [
    " eta ", " da ", " dira ", " ez ", "-a da", "-ak dira", " zen ", "ren ",
    "tzat", " duen ", " dute ", " diren ", "ekin ", " gara ",
]
_SUPPORTED_LANG_SET = {"en", "ta", "eu", "es", "it"}


def _looks_tamil(text: str) -> bool:
    return bool(_TAMIL_RANGE_RE.search(text or ""))


def _looks_basque(text: str) -> bool:
    lowered = f" {(text or '').lower()} "
    return sum(1 for marker in _BASQUE_MARKERS if marker in lowered) >= 2


def detect_language(text: str, dataset_language: str = None) -> str:
    """Resolution order: (1) trusted dataset/caller hint, (2) Tamil script,
    (3) Basque function-word heuristic, (4) langdetect mapped onto the 5
    supported languages (re-checking Basque if langdetect guesses Spanish),
    (5) fall back to "en"."""
    if dataset_language in _SUPPORTED_LANG_SET:
        return dataset_language
    if _looks_tamil(text):
        return "ta"
    if _looks_basque(text):
        return "eu"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        guess = detect(text)
    except Exception:
        return "en"
    mapping = {"en": "en", "es": "es", "it": "it", "eu": "eu", "ta": "ta"}
    if guess == "es" and _looks_basque(text):
        return "eu"
    return mapping.get(guess, "en")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "new_arch") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class RunTimer:
    """Context manager: times a run, writes a one-line JSON summary to
    outputs/logs/{run_name}.jsonl on exit. Does not suppress exceptions."""

    def __init__(self, run_name: str, **fields):
        self.run_name = run_name
        self.fields = fields
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = time.time() - self._start
        summary = {
            "run_name": self.run_name,
            "duration_seconds": round(duration, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "errors": [] if exc is None else [str(exc)],
            **self.fields,
        }
        append_jsonl(summary, config.LOGS_DIR / f"{self.run_name}.jsonl")
        get_logger().info("run '%s' finished in %.2fs", self.run_name, duration)
        return False
