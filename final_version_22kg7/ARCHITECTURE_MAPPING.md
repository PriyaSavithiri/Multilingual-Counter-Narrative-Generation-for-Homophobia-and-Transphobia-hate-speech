# Architecture Mapping — old `src/`/`app/`/`config/` → `new_arch/`

`new_arch/` has zero runtime imports from the old code. Every row below
describes what was **reused as a design/logic reference** (re-implemented
from scratch, values/behaviour verified against the original) vs. what was
**removed** or **redistributed**.

## Providers / infrastructure

| Previous component | New component | Preserved logic | Adapted | Removed |
|---|---|---|---|---|
| `src/llm/base_model.py`, `model_router.py`, `ollama_client.py`, `openai_client.py`, `hf_inference_client.py`, `hf_transformers_client.py` | `model_api.py` | LangChain-only generation (`ChatOllama`/`ChatOpenAI`/`ChatHuggingFace`), availability-check-only direct HTTP, env-only keys, `auto` fallback order | `generate()` signature changed to `(messages, response_schema, temperature, max_tokens)`; bounded retry with backoff added (new, safe) | - |
| `src/data_loading/*.py` (6 loaders + registry + split_utils) | `dataset_loader.py` | Exact on-disk column names (re-verified against actual CSV headers, not memory), `LGBT_TARGET_VALUES` allow-list, 85/15 validation carve, `random_state=42`, TEST never touched, HF Hub fallback for ML_MTCONAN_KN | Consolidated into one file; per-language registry simplified to 3 routed dataset families | - |
| `src/retrieval/*.py` (6 files: corpus_builder, embedding_model, vector_store, retriever, reranker, knowledge_filter) | `rag_pipeline.py` | FAISS `IndexFlatIP` over normalized `intfloat/multilingual-e5-small` embeddings, dual filtered/unfiltered corpus + index cache, TEST/VALIDATION exclusion from the corpus, soft region/language reranking bonuses, diversity penalty | Added `retrieve_bilingual()` (merge/dedupe/rerank across input-language + English query variants) | Old 4th mode `dual` semantics kept as `dual_rag` (renamed, still fact+cultural combined) |
| `src/evaluation/automatic_metrics.py`, `llm_judge.py` | `evaluation.py` | BLEU, ROUGE-L, chrF, BERTScore-F1, Distinct-2, format/safety/copy rates, 19-dimension LLM-judge rubric (verbatim field list and scoring direction) | Added `summarize_traces()`: route distribution, avg debate rounds, evidence-verdict distribution, persona-uniqueness rate | Accuracy/precision/recall, EA/VE/GM - not implemented, same as before |
| `src/utils/*.py` (text_utils, prompt_parsing, file_utils, logging_utils) | `utils.py` | Tolerant JSON parsing (`{}`/`[]` on failure), unsafe-keyword list, sentence/word counting, file I/O, `RunTimer` | Consolidated into one file; `detect_language` (Tamil script / Basque markers / langdetect) folded in from `src/language/` | - |

## Agents / pipeline

| Previous component | New component | Preserved concept? | Required adaptation |
|---|---|---|---|
| `src/agents/harmful_claim_agent.py` | `case_analysis_agent.py` | Yes | Added `hate_type` (explicit/implicit, drives the Router), `hidden_claim`, `evidence_topics`, `region_suggestion` (region/country/confidence/rationale, downgraded to Unknown below threshold) |
| `src/agents/persona_selection_agent.py` + `config/prompts.py ALLOWED_PERSONAS`/`FORBIDDEN_PERSONA_RULES` | `persona_generator_agent.py` | **No — replaced.** No fixed persona bank of any kind survives into `new_arch`. | Fully open-ended, instance-specific generation of 5-10 candidates (default 7, clamped); selects exactly 1 Prosecutor + 1 Defender, generated once before routing and reused for every round |
| `src/agents/persona_response_agent.py` | `defender_agent.py` | Adapted | Evidence verification (SUPPORTED/REFUTED/NEI) is now produced in the same call, per round |
| `src/agents/neutral_response_agent.py` | — | **Removed.** | The neutral/without-persona comparison branch does not exist anywhere in `new_arch`, per explicit instruction. |
| `src/agents/debate/debate_manager.py`, `intent_agent.py`, `evidence_agent.py`, `cultural_safety_agent.py` | `fast_track.py` / `deep_dive_track.py` | Adapted | Fixed 3-round Intent/Evidence/Cultural-Safety debate replaced by a 2-party (Prosecutor/Defender) debate: 1 round (Fast-Track, explicit hate) or K=3 rounds by default (Deep-Dive, implicit hate). Intent-agent responsibilities live in Case Analysis; Evidence/Cultural-Safety responsibilities live in the Defender. |
| `src/agents/debate/critic_agent.py` | `judge_critic_agent.py` | Yes | Now the single synthesis point — also absorbs what `checker_judge_agent.py` used to do (see below) |
| `src/agents/final_writer_agent.py` | `final_cn_agent.py` | Yes | Consumes one `JudgePlan` instead of a separate `debate_plan` + `checker_feedback` |
| **`src/agents/checking/checker_manager.py`** (the old post-hoc Tool-MAD orchestrator) | — | **Removed as a standalone pipeline stage.** | Nothing in `new_arch` runs a fact-check pass *after* the debate completes. |
| `src/agents/checking/query_formulation_agent.py`, `rag_fact_agent.py`, `fact_checker_agent.py` | `evidence_verifier.py` | Redistributed | Called from *inside* `defender_agent.run_round()`, every round — not a separate stage |
| `src/agents/checking/fact_debate_agent.py` (separate 2-round fact-debate) | — | **Removed.** | No duplicate fact-debate pipeline; the one main Prosecutor/Defender debate is the only debate. |
| `src/agents/checking/faithfulness_checker_agent.py`, `safety_checker_agent.py`, `checker_judge_agent.py` | `judge_critic_agent.py` | Merged | One Judge reviews faithfulness/safety/culture together from the Defender's own per-round evidence assessments — it does not rerun a duplicate fact-debate. |
| `src/agents/checking/optional_search_agent.py` | `web_search.py` | **Implemented, by later explicit request** | The old project's `ENABLE_WEB_SEARCH` env var existed but the agent was an unimplemented stub. `new_arch` implements it for real via the Tavily Search API - off by default, and when on, supplements (never replaces) local RAG inside every Defender turn, with explicit `origin`/`url` provenance labeling so web-sourced claims are never trusted anonymously (see `evidence_verifier.py`, `03_PIPELINE_FLOW_EXPLAINED.md`). |

## Prompts (`config/prompts.py` → `prompts.py`)

- `BASE_RULES`, the "never argue for the hateful claim" framing, and the
  2-4 sentence format constraint are preserved conceptually in
  `safety.BASE_SAFETY_RULES` and `build_final_cn_prompt`.
- The 10-item `ALLOWED_PERSONAS` list and `FORBIDDEN_PERSONA_RULES` are
  **not carried forward** - persona generation is fully open-ended (see
  above). The *spirit* of "never an identity impersonation of a
  marginalized group" is preserved as an instruction inside
  `build_persona_generation_prompt`, just without a fixed list to enforce it.
- Of the old project's 10 baseline-method prompt builders, 4 were **ported**
  by later explicit request (`build_prompt_zero`, `build_prompt_few`,
  `build_prompt_cot`, `build_prompt_few_shot_cot` - see `prompts.py` section
  8 and `baseline_methods.py`), for evaluation comparison against the full
  pipeline. The other 4 (`build_prompt_target_aware`, `build_rag_prompt`,
  `build_dynamic_persona_prompt`, `build_fixed_persona_prompt`) were **not**
  ported - they were the old project's own persona/RAG baseline variants,
  made redundant by the fact that `new_arch`'s full pipeline already *is* a
  persona+RAG-driven method (comparing it against baselines resembling
  itself would add little).
- `build_harmful_claim_prompt`, `build_persona_pool_prompt`,
  `build_persona_selection_prompt`, `build_debate_round_prompt`,
  `build_critic_plan_prompt`, `build_final_cn_prompt`,
  `build_fact_query_prompt`, `build_rag_fact_agent_prompt`,
  `build_tool_mad_judge_prompt` are the direct conceptual ancestors of
  `build_case_analysis_prompt`, `build_persona_generation_prompt`,
  `build_prosecutor_prompt`/`build_defender_prompt`, `build_judge_prompt`,
  `build_final_cn_prompt`, and `build_claim_query_prompt` respectively.
- **New**: every prompt that interpolates the raw hate comment wraps it via
  `safety.wrap_untrusted_text()` - an explicit prompt-injection defense that
  did not exist anywhere in the old project's prompts.
- **Re-added, by later explicit request**: `build_prompt_zero`,
  `build_prompt_few`, `build_prompt_cot`, `build_prompt_few_shot_cot` (in
  `prompts.py`) - direct-ish ports of the old project's baseline-method
  prompts (same structure: `BASE_RULES`-equivalent, "Counter-narrative:"
  marker for CoT extraction), reusing `safety.wrap_untrusted_text()` and
  `safety.BASE_SAFETY_RULES` for consistency with the rest of `new_arch`.
  Wired up in the new `baseline_methods.py`, not the agent/pipeline layer -
  these are simple prompting strategies, not architecture stages, so
  keeping them out of `pipeline.py` avoids blurring what's being compared
  against what.

## Streamlit UI

- Same general layout (sidebar settings, live `st.status` progress with GPU
  readout, Generate/Evaluation tabs) is preserved in `app.py`.
- The "Method" dropdown (`full_pipeline`/`zero_shot`/`few_shot`/`cot`/
  `few_shot_cot`) was initially removed (only one pipeline existed), then
  **re-added by explicit request** so baseline comparisons could be run
  from one consistent codebase - selecting a baseline hides the
  RAG/region/persona controls in the sidebar, since none of them apply.
- New controls: persona candidate count, Deep-Dive round count, retrieval
  query-language strategy, and a cultural-region selector defaulting to
  "Auto / Unknown" with an explicit accept-suggestion checkbox.
- New: a restricted, `SHOW_INTERNAL_TRACES`-gated developer expander
  (warning-labelled, excerpt-only) - the old UI had no equivalent concept
  since it had no Prosecutor-style content to hide in the first place.
- New: a download button for the (redacted) result JSON - the old UI had
  none.

## Dataset reuse

Same 6 datasets, same `data_orig/` directory, same original columns/labels,
same target-group allow-list, same validation-carve/TEST-exclusion rules -
re-verified against the actual files rather than trusted from memory (see
`tests/test_dataset_integrity.py`, which checksums the original files
before/after every loader call).

## RAG mode reuse

All 4 modes preserved (`no_rag`/`fact_rag`/`cultural_rag`/`dual_rag`,
renamed from the old `none`/`factual`/`cultural`/`dual`), with `dual_rag` as
the default (was `dual` in the old project too). New: bilingual
(input-language + English) query generation and merged/deduped/reranked
retrieval, gated per query-strategy config.
