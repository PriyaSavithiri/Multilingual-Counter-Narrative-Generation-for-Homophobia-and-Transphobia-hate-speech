"""
new_arch Streamlit UI - same general workflow/style as the old project's
app/streamlit_app.py (2 tabs, sidebar settings, live st.status progress,
GPU readout), adapted for the Prosecutor/Defender/Judge architecture. A
"Method" dropdown lets you pick the full pipeline OR one of the simple
zero_shot/few_shot/cot/few_shot_cot baselines (evaluation comparison
points - no RAG/personas/debate). Region is a manual selector defaulting
to "Auto / Unknown" with the Case Analysis suggestion shown (never
silently applied), and Prosecutor content is never shown in the normal
result view - only in a restricted, config-gated developer expander.
"""
import argparse
import json
import subprocess

import pandas as pd
import streamlit as st

import baseline_methods
import config
import dataset_loader
import main as main_cli
from model_api import get_client, list_local_models
from pipeline import NewArchPipeline
from safety import DEBUG_WARNING_TEXT
from schemas import InputRecord

st.set_page_config(page_title="new_arch - Prosecutor/Defender/Judge Counter-Narrative Generator", layout="wide")
st.title("Culturally Grounded Counter-Narrative Generator (new_arch)")
# st.caption(
#     "Paper-inspired adaptation - not an exact replication of any single paper. "
#     "Prosecutor/Defender/Judge architecture: persona-driven only, no neutral/without-persona branch. "
#     "See README.md and ARCHITECTURE_MAPPING.md for details."
# )

CUSTOM_MODEL_OPTION = "Custom (type below)..."


def _model_options(backend: str) -> list:
    if backend == "ollama":
        installed = list_local_models()
        if installed:
            return installed
        return config.RECOMMENDED_OLLAMA_MODELS + config.LARGE_OLLAMA_MODELS_MANUAL_ONLY
    if backend == "hf-transformers":
        return config.RECOMMENDED_HF_TRANSFORMERS_MODELS
    if backend == "hf-inference":
        return config.RECOMMENDED_HF_INFERENCE_MODELS
    if backend == "openai":
        from model_api import DEFAULT_OPENAI_MODEL
        return [DEFAULT_OPENAI_MODEL]
    return list(dict.fromkeys(list_local_models() + [config.DEFAULT_MODEL]))


def _gpu_status() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            util, used, total = [x.strip() for x in result.stdout.strip().split(",")]
            return f"GPU {util}% - {used}/{total} MiB VRAM"
    except Exception:
        pass
    return ""


generate_tab, evaluation_tab = st.tabs(["Generate", "Evaluation"])

with generate_tab:
    with st.sidebar:
        st.header("Settings")
        method = st.selectbox("Method", baseline_methods.ALL_METHODS,
                               index=baseline_methods.ALL_METHODS.index("full_pipeline"),
                               help="full_pipeline = the Prosecutor/Defender/Judge architecture (default). "
                                    "zero_shot/few_shot/cot/few_shot_cot = simple non-agentic baselines with "
                                    "no RAG/personas/debate, kept only as evaluation comparison points.")
        num_few_shot = config.NUM_FEW_SHOT_EXAMPLES
        if method in ("few_shot", "few_shot_cot"):
            num_few_shot = st.number_input("Few-shot examples (drawn from TRAIN only)", min_value=0, max_value=10,
                                            value=config.NUM_FEW_SHOT_EXAMPLES)
        backend = st.selectbox("Backend", config.SUPPORTED_BACKENDS,
                                index=config.SUPPORTED_BACKENDS.index(config.DEFAULT_BACKEND))
        model_choices = _model_options(backend) + [CUSTOM_MODEL_OPTION]
        default_idx = model_choices.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in model_choices else 0
        model_pick = st.selectbox("Model", model_choices, index=default_idx)
        model = st.text_input("Custom model name", value=config.DEFAULT_MODEL) if model_pick == CUSTOM_MODEL_OPTION else model_pick

        language_choice = st.selectbox("Language", ["auto-detect"] + config.SUPPORTED_LANGUAGES)

        # RAG/region/persona/debate settings only apply to the full pipeline -
        # baselines ignore all of them (no RAG, no personas, no debate).
        region_choice, accept_suggestion = "Auto / Unknown", False
        rag_mode, rag_corpus, query_strategy = config.DEFAULT_RAG_MODE, "filtered", config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE
        persona_count, deep_dive_rounds = config.DEFAULT_PERSONA_CANDIDATE_COUNT, config.get_deep_dive_rounds()

        if method == "full_pipeline":
            st.caption("Cultural region - manual selection always takes priority over any suggestion.")
            region_choice = st.selectbox("Cultural region", ["Auto / Unknown"] + config.KNOWN_REGIONS)
            if region_choice == "Auto / Unknown":
                accept_suggestion = st.checkbox(
                    "Accept Case Analysis's region suggestion automatically (only applied if confident)",
                    value=False,
                )

            st.caption("RAG settings")
            rag_mode = st.selectbox("RAG mode", config.RAG_MODES,
                                     index=config.RAG_MODES.index(config.DEFAULT_RAG_MODE),
                                     help="no_rag = no retrieval. fact_rag = only factual/legal/historical evidence. "
                                          "cultural_rag = only culturally-relevant example evidence. "
                                          "dual_rag (default) = both, factual grounding never replaced by cultural context.")
            rag_corpus = st.selectbox("RAG corpus", ["filtered", "unfiltered"],
                                       help="filtered (recommended) = homophobia/transphobia-scoped evidence only. "
                                            "unfiltered = every hate-speech target group, kept only for ablation.")
            query_strategy = st.selectbox("Retrieval query language", config.RETRIEVAL_QUERY_LANGUAGE_MODES,
                                           index=config.RETRIEVAL_QUERY_LANGUAGE_MODES.index(
                                               config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE),
                                           help="both (default) retrieves using the input-language query AND an "
                                                "English-equivalent query, then merges/dedupes/reranks results.")

            st.caption("Persona / debate settings")
            persona_count = st.number_input("Persona candidate count", min_value=config.PERSONA_CANDIDATE_COUNT_MIN,
                                             max_value=config.PERSONA_CANDIDATE_COUNT_MAX,
                                             value=config.DEFAULT_PERSONA_CANDIDATE_COUNT)
            deep_dive_rounds = st.number_input("Deep-Dive rounds (K) - only used for implicit hate",
                                                min_value=1, max_value=6, value=config.get_deep_dive_rounds())

        if config.SHOW_INTERNAL_TRACES:
            st.caption("⚠ SHOW_INTERNAL_TRACES is enabled via configuration - a restricted developer "
                       "expander with internal Prosecutor excerpts will appear below after generation.")
        if method == "full_pipeline" and config.ENABLE_WEB_SEARCH:
            st.caption("🌐 ENABLE_WEB_SEARCH is enabled via configuration - the Defender will supplement "
                       "local RAG evidence with live web search results (Tavily) each round, each one "
                       "labeled with its source URL.")

    comment = st.text_area("Hate comment", height=120, placeholder="Paste a homophobic/transphobic comment here...")

    if st.button("Generate Counter-Narrative", type="primary"):
        if not comment.strip():
            st.warning("Please enter a comment first.")
        else:
            trace = None
            status_label = ("Running the Prosecutor/Defender/Judge pipeline... this can take a while on local models."
                             if method == "full_pipeline" else f"Running baseline method '{method}'...")
            with st.status(status_label, expanded=True) as status:
                def _report(stage: str, _status=status):
                    gpu = _gpu_status()
                    _status.update(label=stage)
                    _status.write(f"{stage}" + (f"  \n*{gpu}*" if gpu else ""))

                try:
                    client = get_client(backend=backend, model=model)
                    dataset_language = None if language_choice == "auto-detect" else language_choice

                    if method == "full_pipeline":
                        pipeline = NewArchPipeline(client, backend_name=backend, model_name=model)
                        region_hint = None if region_choice == "Auto / Unknown" else region_choice
                        record = InputRecord(text=comment, language_hint=dataset_language, region_hint=region_hint,
                                              rag_mode=rag_mode)
                        trace = pipeline.generate(
                            record, accept_region_suggestion=accept_suggestion,
                            persona_candidate_count=int(persona_count), deep_dive_rounds=int(deep_dive_rounds),
                            query_strategy=query_strategy, filter_target=(rag_corpus == "filtered"),
                            progress_callback=_report,
                        )
                    else:
                        train_df = None
                        if method in ("few_shot", "few_shot_cot") and dataset_language:
                            train_df, _, _ = dataset_loader.prepare_and_load(dataset_language, filter_target=True)
                        _report(f"Running baseline method '{method}'...")
                        trace = baseline_methods.run_baseline(
                            client, method, comment, language=dataset_language or "en", backend_name=backend,
                            model_name=model, train_df=train_df, num_examples=num_few_shot,
                        )
                    status.update(label="Pipeline finished.", state="complete")
                except Exception as exc:
                    status.update(label="Pipeline failed.", state="error")
                    st.error(f"Pipeline failed: {exc}")

            if trace:
                st.subheader("Final Counter-Narrative")
                st.success(trace.get("counter_narrative") or "(empty output)")
                if method != "full_pipeline":
                    st.caption(f"Generated via the '{method}' baseline - no RAG, personas, or debate involved.")
                meta = trace.get("metadata", {})

                if method == "full_pipeline":
                    st.markdown(f"**Explanation:** {trace.get('explanation') or '(none)'}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Detected language:** {config.LANGUAGE_NAMES.get(trace.get('language'), trace.get('language'))}")
                        st.markdown(f"**Hate type / track:** {trace.get('hate_type')} -> {trace.get('selected_track')}")
                        st.markdown(f"**Prosecutor persona (internal role, name only):** {meta.get('persona_prosecutor')}")
                        st.markdown(f"**Defender persona:** {meta.get('persona_defender')}")
                        region_suggestion = meta.get("region_suggestion") or {}
                        st.markdown(
                            f"**Region confirmed for this run:** {meta.get('region_confirmed') or 'Unknown (culturally neutral language used)'}"
                        )
                        if region_suggestion.get("region"):
                            st.caption(f"Case Analysis suggested: {region_suggestion.get('region')} "
                                       f"({region_suggestion.get('country')}), confidence {region_suggestion.get('confidence')} - "
                                       f"{region_suggestion.get('rationale')}. Not applied unless accepted/manually selected.")
                    with col2:
                        st.markdown("**Evidence trace (claim -> verdict):**")
                        st.json(trace.get("evidence_trace") or [])

                    persona_selection = trace.get("persona_selection") or {}
                    with st.expander("Why these personas were chosen"):
                        st.markdown(f"**Selection rationale:** {persona_selection.get('selection_rationale') or '(none given)'}")
                        pcol1, pcol2 = st.columns(2)
                        sp = persona_selection.get("selected_prosecutor") or {}
                        sd = persona_selection.get("selected_defender") or {}
                        with pcol1:
                            st.markdown(f"**Prosecutor - {sp.get('name', '(none)')}**")
                            st.markdown(f"Objective: {sp.get('objective') or '(none)'}")
                            if sp.get("boundaries"):
                                st.markdown("Boundaries: " + "; ".join(sp["boundaries"]))
                        with pcol2:
                            st.markdown(f"**Defender - {sd.get('name', '(none)')}**")
                            st.markdown(f"Objective: {sd.get('objective') or '(none)'}")
                            if sd.get("expertise"):
                                st.markdown("Expertise: " + ", ".join(sd["expertise"]))
                            if sd.get("cultural_guidance"):
                                st.markdown("Cultural guidance: " + "; ".join(sd["cultural_guidance"]))
                        candidates = persona_selection.get("candidate_personas") or []
                        if candidates:
                            st.markdown(f"**Full candidate pool ({len(candidates)}):**")
                            st.dataframe(pd.DataFrame(candidates), use_container_width=True)

                    with st.expander("Debate rounds (public view - Prosecutor content withheld)"):
                        st.json(trace.get("debate_rounds") or [])
                    with st.expander("Judge/Critic plan"):
                        st.json(trace.get("judge_plan") or {})

                    if "_debug" in trace:
                        with st.expander("⚠ Developer/debug view (restricted - internal Prosecutor excerpts)"):
                            st.warning(DEBUG_WARNING_TEXT)
                            st.json(trace["_debug"]["rounds"])

                if meta.get("errors"):
                    st.error(f"Errors: {meta['errors']}")

                with st.expander("Full trace (raw, Prosecutor content withheld)"):
                    st.json(trace)

                import json
                st.download_button(
                    "Download result (JSON, Prosecutor content withheld)",
                    data=json.dumps(trace, ensure_ascii=False, indent=2),
                    file_name=f"counter_narrative_{trace.get('input_id') or 'result'}.json",
                    mime="application/json",
                )

with evaluation_tab:
    st.subheader("Run full validation-set evaluation")
    st.caption("Runs the chosen model over the ENTIRE validation split (per language selected), then computes "
               "the final automatic metrics - no need to leave the UI and use the CLI. Can take a long time "
               "for a large model/split combination; watch the progress bar below.")

    with st.form("full_eval_form"):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            eval_backend = st.selectbox("Backend", config.SUPPORTED_BACKENDS,
                                         index=config.SUPPORTED_BACKENDS.index(config.DEFAULT_BACKEND),
                                         key="eval_backend")
            eval_model_choices = _model_options(eval_backend) + [CUSTOM_MODEL_OPTION]
            eval_default_idx = (eval_model_choices.index(config.DEFAULT_MODEL)
                                 if config.DEFAULT_MODEL in eval_model_choices else 0)
            eval_model_pick = st.selectbox("Model", eval_model_choices, index=eval_default_idx, key="eval_model_pick")
            eval_model = (st.text_input("Custom model name", value=config.DEFAULT_MODEL, key="eval_model_custom")
                          if eval_model_pick == CUSTOM_MODEL_OPTION else eval_model_pick)
            eval_method = st.selectbox("Method", baseline_methods.ALL_METHODS,
                                        index=baseline_methods.ALL_METHODS.index("full_pipeline"), key="eval_method")
            eval_languages = st.multiselect("Language(s)", config.SUPPORTED_LANGUAGES,
                                             default=config.SUPPORTED_LANGUAGES, key="eval_languages")
        with fcol2:
            eval_split = st.selectbox("Split", ["validation", "test"], key="eval_split")
            eval_limit = st.number_input("Limit per language (0 = entire split)", min_value=0, value=0,
                                          key="eval_limit")
            eval_rag_mode = st.selectbox("RAG mode", config.RAG_MODES,
                                          index=config.RAG_MODES.index(config.DEFAULT_RAG_MODE),
                                          help="Ignored by baseline methods.", key="eval_rag_mode")
            eval_rag_corpus = st.selectbox("RAG corpus", ["filtered", "unfiltered"],
                                            help="Ignored by baseline methods.", key="eval_rag_corpus")
            eval_llm_judge = st.checkbox("Also run the 19-dimension LLM-judge rubric (slower, uses this model)",
                                          key="eval_llm_judge")

        if eval_split == "test":
            st.warning("TEST should be used only once, for a final reported result - not repeatedly during development.")

        submitted = st.form_submit_button("Run full evaluation", type="primary")

    if submitted:
        if not eval_languages:
            st.warning("Pick at least one language.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            try:
                status_text.write(f"Loading model '{eval_model}' via backend '{eval_backend}'...")
                shared_client = get_client(backend=eval_backend, model=eval_model)

                for li, lang in enumerate(eval_languages):
                    run_args = argparse.Namespace(
                        language=lang, split=eval_split, limit=(eval_limit or None),
                        backend=eval_backend, model=eval_model, method=eval_method,
                        num_few_shot_examples=config.NUM_FEW_SHOT_EXAMPLES, rag_mode=eval_rag_mode,
                        rag_corpus=eval_rag_corpus, query_strategy=config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE,
                        persona_count=config.DEFAULT_PERSONA_CANDIDATE_COUNT,
                        deep_dive_rounds=config.get_deep_dive_rounds(),
                    )

                    def _on_row_done(done, total, row_id, _lang=lang, _li=li):
                        frac = (_li + done / total) / len(eval_languages)
                        progress_bar.progress(min(frac, 1.0))
                        status_text.write(f"[{_lang}] {done}/{total} rows generated (last id={row_id})")

                    main_cli.cmd_run_dataset(run_args, progress_callback=_on_row_done, client=shared_client)

                progress_bar.progress(1.0)
                status_text.write("Generation complete for all selected languages - computing metrics...")

                eval_args = argparse.Namespace(backend=eval_backend, model=eval_model, llm_judge=eval_llm_judge)
                main_cli.cmd_evaluate(eval_args)
                status_text.write("Done.")

                summary_path = config.EVALUATIONS_DIR / "all_languages_summary.csv"
                if summary_path.exists():
                    summary_df = pd.read_csv(summary_path)
                    result_rows = summary_df[summary_df["model"] == eval_model] if "model" in summary_df.columns else summary_df
                    st.subheader("Result")
                    st.dataframe(result_rows, use_container_width=True)
            except Exception as exc:
                st.error(f"Full evaluation run failed: {exc}")

    st.divider()
    st.subheader("Model performance across languages")
    summary_path = config.EVALUATIONS_DIR / "all_languages_summary.csv"
    if not summary_path.exists():
        st.info(
            "No evaluation summary found yet. Run generations then evaluate them first:\n\n"
            "```\npython main.py run-dataset --language en --rag-mode dual_rag\n"
            "python scripts_or_notebook_to_call_evaluation.py  # see evaluation.py\n```\n\n"
            "Evaluation/ablation runs must specify --rag-mode explicitly and support all 4 modes."
        )
    else:
        summary_df = pd.read_csv(summary_path)
        if "split" in summary_df.columns and (summary_df["split"] == "test").any():
            st.warning(
                "One or more rows below were evaluated on the TEST split. TEST should be used "
                "only once, for a final reported result - not repeatedly during development."
            )
        st.dataframe(summary_df, use_container_width=True)
        chart_metrics = [c for c in ["avg_bleu", "avg_rouge_l", "avg_bertscore_f1", "avg_distinct_2"]
                          if c in summary_df.columns]
        if chart_metrics and "language" in summary_df.columns:
            st.markdown("**Metric averages by language**")
            st.bar_chart(summary_df.groupby("language")[chart_metrics].mean())
