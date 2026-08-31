# new_arch — Prosecutor/Defender/Judge Counter-Narrative Generator

## Purpose

This folder is a clean-room rewrite of the thesis pipeline around an updated
architecture (inspired by, but not identical to, the ARCADE framework):
instead of a fixed persona bank debating "for safety" from the start, an
open-ended, instance-specific **Prosecutor** persona genuinely argues the
harmful claim's strongest internal logic (strictly internal, never shown to
end users), while an open-ended **Defender** persona rebuts it with
evidence and cultural awareness folded into every round. A single **Judge/
Critic** then synthesizes a final response plan, and a **Final
Counter-Narrative Agent** writes the user-facing output.

Paper-inspired adaptation - not an exact replication of any single paper or
of the ARCADE framework.

The old implementation (`src/`, `app/`, `config/`, `scripts/`) remains
completely untouched and is kept only as a reference. `new_arch/` has no
runtime dependency on any of it - every module here (dataset loading, RAG
retrieval, model clients, prompts, agents) is an independent, from-scratch
implementation guided by the same verified rules, not an import of the old
code.

## Updated architecture

```
Input (validated, language/region hints optional)
  -> Case Analysis Agent          (language, target, hate_type explicit/implicit,
                                    hidden claim, evidence topics, cautious region
                                    SUGGESTION only)
  -> Persona Generator Agent      (5-10 open-ended, instance-specific candidates,
                                    generated ONCE; selects 1 Prosecutor + 1 Defender)
  -> Router (deterministic, no LLM call)
       explicit -> Fast-Track  (1 Prosecutor<->Defender round)
       implicit -> Deep-Dive   (K=3 rounds by default, same 2 personas reused
                                 every round - personas are generated once and
                                 do not change per round or per track)
  -> Judge/Critic Agent           (single synthesis; reviews the full debate,
                                    including internal Prosecutor content, but
                                    never copies it into its output)
  -> Final Counter-Narrative Agent (counter-narrative + explanation, same
                                     language as the input, Prosecutor content
                                     never present)
```

Alongside the full pipeline, `new_arch` also implements 4 simple,
non-agentic **baseline methods** (`zero_shot`, `few_shot`, `cot`,
`few_shot_cot` - see `baseline_methods.py`) for evaluation comparison, so
every number in the thesis's baseline-vs-full-pipeline tables comes from
one consistent codebase rather than mixing this project with the old one.

Evidence verification (SUPPORTED / REFUTED / NEI) happens **inside every
Defender turn**, not as a separate post-debate stage - there is no
standalone "Tool-MAD" pipeline step anywhere in `new_arch`. There is also no
neutral/without-persona comparison branch - the system is persona-driven
only.

## File structure

| File | Role |
|---|---|
| `config.py` | Paths, languages, RAG modes/defaults, persona/K defaults, model defaults, env secrets |
| `schemas.py` | Plain dataclasses for every pipeline-stage record + input validation |
| `model_api.py` | One `ModelClient` interface, 4 backends (ollama/openai/hf-inference/hf-transformers), `auto` router, retry logic |
| `dataset_loader.py` | Read-only loaders for all 6 datasets, target filtering, splits, normalized records |
| `rag_pipeline.py` | Corpus build, FAISS/e5 retrieval, 4 RAG modes, bilingual query merge/rerank |
| `web_search.py` | Optional Tavily web search, off by default (`ENABLE_WEB_SEARCH`), supplements local RAG |
| `prompts.py` | Every prompt template, with prompt-injection framing built in |
| `safety.py` | Prompt-injection wrapper, Prosecutor-content redaction, debug-trace gating |
| `case_analysis_agent.py`, `persona_generator_agent.py`, `router.py`, `prosecutor_agent.py`, `defender_agent.py`, `evidence_verifier.py`, `fast_track.py`, `deep_dive_track.py`, `judge_critic_agent.py`, `final_cn_agent.py` | One file per architecture module |
| `pipeline.py` | Visible orchestration tying every stage together |
| `baseline_methods.py` | Simple zero_shot/few_shot/cot/few_shot_cot comparison methods (no RAG/personas/debate) |
| `main.py` | CLI (`build-index`, `generate`, `run-dataset`, `evaluate`) - `--method` selects `full_pipeline` (default) or a baseline |
| `app.py` | Streamlit UI - "Method" dropdown selects the pipeline or a baseline |
| `evaluation.py` | Automatic metrics + 19-dim LLM judge + new-arch-specific metrics |
| `utils.py` | JSON parsing, language detection, text/format checks, file I/O, logging |
| `tests/` | 11 test files covering routing, personas, both tracks, evidence, language, dataset integrity, baseline methods, web search, and the full mocked pipeline |
| `outputs/` | `generations/`, `evaluations/`, `logs/`, `traces/` (normal), `debug/` (restricted, gated by `SAVE_INTERNAL_TRACES`) |
| `data/` | `knowledge/` (built RAG corpus JSONL), `vector_store/` (FAISS index) - derived data only, never touches `data_orig/` |

## Setup

```
pip install -r new_arch/requirements.txt   # identical deps to the rest of the project
copy new_arch\.env.example new_arch\.env   # fill in only what you need
python new_arch/main.py build-index        # builds both filtered and unfiltered RAG corpora/indexes
```

Ollama (the default backend) must be installed and running (`ollama serve`),
with at least one model pulled (`ollama pull qwen2.5:7b-instruct`).

## Environment variables

See `.env.example` for the full list: `HF_TOKEN`, `OPENAI_API_KEY`,
`OLLAMA_HOST`, `EMBEDDING_MODEL_NAME`, `RAG_TOP_K`, `DEEP_DIVE_ROUNDS`,
`SHOW_INTERNAL_TRACES`, `SAVE_INTERNAL_TRACES`. No key is ever hardcoded;
a missing key fails immediately with a clear message.

## Model-provider configuration

Same 4 backends as the rest of the project, all real generation via
LangChain (`ChatOllama` / `ChatOpenAI` / `ChatHuggingFace`), never a direct
HTTP/SDK call except lightweight availability checks:

```
python new_arch/main.py generate --text "..." --backend ollama --model qwen2.5:7b-instruct
python new_arch/main.py generate --text "..." --backend auto        # ollama -> hf-inference -> hf-transformers -> openai
```

Never default to 24B/32B/70B-class models (hardware target: RTX 4050
laptop GPU, 6GB VRAM) - still usable manually via `--model` if you have the
hardware/patience.

## Streamlit usage

```
streamlit run new_arch/app.py
```

Same general workflow as the old UI (sidebar settings, live progress via
`st.status`, GPU readout, Generate/Evaluation tabs), with new controls:
persona candidate count (5-10, default 7), Deep-Dive round count, retrieval
query-language strategy, and a cultural-region selector that defaults to
**Auto / Unknown** - a Case Analysis region suggestion is shown but never
silently applied; you must either pick a region manually or explicitly
check "accept the suggestion" before it's used.

## Dataset configuration

Reads directly, read-only, from `data_orig/` (same directory as the rest of
the project - resolved via the same `data_orig/` -> `datasets/` -> sibling
fallback order). No dataset file, column, or row is ever modified, renamed,
or overwritten. `en`/`ta` use the CodaBench (LT-EDI) files; `eu`/`es`/`it`
use ML_MTCONAN_KN (fetched from the Hugging Face Hub on first use per
process, since it isn't present locally). Multitarget-CONAN, KN-grounded-CN,
and CONAN-MT-SP are multi-target hate-speech corpora used only to widen the
RAG knowledge corpus - never as a language's own train/eval source.
Target-group filtering (default on) restricts those 3 plus ML_MTCONAN_KN to
homophobia/transphobia/LGBT+-relevant rows via the same verified allow-list
as the old project (`{"lgbt+","lgbt","homophobia","transphobia"}`).

Validation is carved from TRAIN only (85/15, `random_state=42`), entirely
in memory - never written to disk, never touching TEST. TEST is opt-in
(`--split test`) and prints a warning.

## RAG modes

All 4 preserved: `no_rag`, `fact_rag`, `cultural_rag`, `dual_rag` (default).
`dual_rag` is the default because culturally-grounded generation is this
project's primary proposed system - it includes factual retrieval/
verification alongside cultural retrieval and never lets cultural context
replace factual grounding. Evaluation/ablation runs (`main.py run-dataset`)
must specify `--rag-mode` explicitly; they never rely on the UI default.

Bilingual retrieval (`--query-strategy both`, the default): for each
Defender-surfaced claim, one query is formulated in the input language and
one semantically-equivalent query in English; both are retrieved, merged,
deduplicated, and reranked together. Especially useful for `ta`/`eu`, where
English queries improve factual recall while input-language queries
preserve linguistically/culturally relevant evidence. The retrieved
source's own language is always preserved (never translated); which
query/language retrieved each passage is recorded on the evidence item.

`no_rag` mode generates **zero** retrieval queries - not just zero results,
the query-formulation LLM call itself is never made (see
`evidence_verifier.py` / `tests/test_evidence_verification.py`).

## Web search (optional, off by default)

`ENABLE_WEB_SEARCH=true` (env var, see `.env.example`) supplements local RAG
with a live web search via the [Tavily](https://tavily.com) API
(`TAVILY_API_KEY` required, `pip install tavily-python`) - a real,
non-stub implementation of what the old project's `optional_search_agent.py`
left as an unimplemented placeholder. When enabled, **every** Defender
evidence-gathering call runs both local RAG and web search together (not
only as a fallback when local RAG is empty). Every evidence item's
provenance is explicit - local corpus items carry `origin="local_rag"`;
web results carry `origin="web_search"` plus the source `url` - and the
Defender is instructed to only treat a web-sourced claim as `SUPPORTED` if
the source is identifiable and looks credible, otherwise `NEI`. Skipped
entirely in `no_rag` mode (no retrieval of any kind happens there). A web
search failure (missing key, network error) degrades gracefully to
local-only evidence for that claim rather than crashing the round -
missing configuration (`ENABLE_WEB_SEARCH=true` with no `TAVILY_API_KEY`)
raises a clear, actionable error instead of silently skipping, consistent
with how every other API key is handled in this project.

## Cultural-awareness behaviour

Applied at every stage (Case Analysis, Persona Generator, Prosecutor,
Defender, Judge, Final-CN) - never bolted on only at the final prompt.
Region is never inferred from language, name, religion, ethnicity, or
stereotype. A Case Analysis region suggestion is downgraded to `Unknown`
below a confidence threshold and is always presented as a suggestion, never
a confirmed fact; the resolution priority is manual UI selection > trusted
dataset metadata > an explicitly-accepted suggestion > `Unknown` (culturally
neutral language, no region-specific retrieval terms).

## Single-input execution

```
python new_arch/main.py generate --text "..." --language en --rag-mode dual_rag --rag-corpus filtered
```

`--method` defaults to `full_pipeline`. Use `zero_shot` / `few_shot` / `cot` /
`few_shot_cot` to run a simple non-agentic baseline instead (see
"Baseline methods" below) - useful for quick comparison points.

## Dataset execution

```
python new_arch/main.py run-dataset --language ta --split validation --rag-mode fact_rag --limit 30
python new_arch/main.py run-dataset --language ta --split validation --method zero_shot --limit 30
```

Writes `outputs/generations/{lang}_{split}_{method}_{model}_{rag_mode}_{rag_corpus}.csv`
and a matching `outputs/traces/{lang}_{split}_{method}_traces.jsonl` (public
view - Prosecutor content always withheld, per `safety.py`; baseline runs
always record `rag_mode="no_rag"`/`rag_corpus="n_a"` regardless of the flags
passed, since baselines never retrieve).

## Baseline methods

Alongside `full_pipeline`, 4 simple, non-agentic methods are available for
evaluation comparison (`baseline_methods.py`, `config.NUM_FEW_SHOT_EXAMPLES`
= 3 by default): `zero_shot`, `few_shot` (examples drawn from TRAIN only,
never test), `cot` (4-step chain-of-thought, final line extracted via
`utils.extract_final_counter_narrative()`), `few_shot_cot` (both combined).
None of these use RAG, personas, or debate - each is a single LLM call. Pass
`--method <name>` to `main.py generate` / `run-dataset`, or pick it from the
"Method" dropdown in `app.py` (selecting a baseline hides the RAG/region/
persona controls, since they don't apply).

## Output schema

Every `pipeline.generate(...)` call returns a redacted trace dict:
`input_id, language, hate_type, selected_track, rag_mode,
counter_narrative, explanation, evidence_trace, metadata, case_analysis,
persona_selection, router_decision, debate_rounds, judge_plan` (+ an
optional `_debug` key, only present when `SHOW_INTERNAL_TRACES=true`,
containing a short excerpt-only summary, never the raw Prosecutor text).

## Testing

```
python new_arch/tests/test_router.py
python new_arch/tests/test_case_analysis.py
python new_arch/tests/test_persona_generation.py
python new_arch/tests/test_fast_track.py
python new_arch/tests/test_deep_dive.py
python new_arch/tests/test_evidence_verification.py
python new_arch/tests/test_language_consistency.py
python new_arch/tests/test_dataset_integrity.py
python new_arch/tests/test_baseline_methods.py
python new_arch/tests/test_web_search.py
python new_arch/tests/test_pipeline.py
```

(or `python -m pytest new_arch/tests/` if pytest is installed - every test
is a plain `assert`-based function, framework-agnostic). All 11 files pass
as of this writing, run against the real local datasets (read-only,
checksum-verified unchanged) and a mocked/stub LLM client for the
agent-level, baseline-method, and full-pipeline tests. The `zero_shot`
baseline has also been verified end-to-end against a real Ollama model,
both via the CLI and by clicking through the Streamlit UI.

## Known limitations

- ML_MTCONAN_KN (`eu`/`es`/`it`) requires internet access on first use per
  process (Hugging Face Hub fetch, no local copy under `data_orig/`).
- The Judge/Critic and Final-CN agents are single LLM calls each - like the
  old project, there is no multi-sample self-consistency voting.
- `persona_uniqueness_rate` (in `evaluation.py`) is a proxy metric, not a
  semantic-diversity measure - two personas with different names but very
  similar substance would still count as "unique".
- No automated check enforces that the Prosecutor and Defender personas
  selected by the LLM are never the same name beyond the prompt's own
  instruction; `judge_critic_agent`/`pipeline.py` do not currently reject a
  same-name pair, they only log it implicitly via the persona_selection
  trace (worth tightening if this is observed in practice).
- No CI/lint pipeline is configured for `new_arch/` specifically.
