"""
new_arch CLI entry point. Subcommands:
  build-index    build the FAISS RAG index(es)
  generate       run the pipeline on one comment
  run-dataset    run the pipeline over one language's eval split, write a
                 generations CSV + a traces JSONL (mirrors the old project's
                 run_single_example.py / run_dataset.py, adapted)
  evaluate       compute automatic metrics (+ optional LLM judge, + new-arch
                 route/persona/evidence metrics) over generations CSVs
                 (mirrors the old project's evaluate_outputs.py, adapted)

Evaluation and ablation runs must specify --rag-mode explicitly - they never
silently rely on a UI default (see evaluation.py / README).
"""
import argparse

import pandas as pd

import baseline_methods
import config
import dataset_loader
import evaluation
import rag_pipeline
from model_api import get_client
from pipeline import NewArchPipeline
from schemas import InputRecord
from utils import get_logger, append_jsonl, RunTimer

logger = get_logger("new_arch.main")


def _add_common_model_args(p):
    p.add_argument("--backend", default=config.DEFAULT_BACKEND, choices=config.SUPPORTED_BACKENDS)
    p.add_argument("--model", default=config.DEFAULT_MODEL)


def _add_common_pipeline_args(p):
    p.add_argument("--method", default="full_pipeline", choices=baseline_methods.ALL_METHODS,
                    help="full_pipeline (default) = the Prosecutor/Defender/Judge architecture. "
                         "zero_shot/few_shot/cot/few_shot_cot = simple non-agentic baselines, no "
                         "RAG/personas/debate, for evaluation comparison only.")
    p.add_argument("--num-few-shot-examples", type=int, default=config.NUM_FEW_SHOT_EXAMPLES,
                    help="Only used by --method few_shot / few_shot_cot; examples drawn from TRAIN only.")
    p.add_argument("--rag-mode", default=config.DEFAULT_RAG_MODE, choices=config.RAG_MODES,
                    help="Must be specified explicitly for evaluation/ablation runs. Ignored by baseline methods.")
    p.add_argument("--rag-corpus", default="filtered", choices=["filtered", "unfiltered"],
                    help="Ignored by baseline methods.")
    p.add_argument("--query-strategy", default=config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE,
                    choices=config.RETRIEVAL_QUERY_LANGUAGE_MODES, help="Ignored by baseline methods.")
    p.add_argument("--persona-count", type=int, default=config.DEFAULT_PERSONA_CANDIDATE_COUNT,
                    help="Ignored by baseline methods.")
    p.add_argument("--deep-dive-rounds", type=int, default=None,
                    help=f"Defaults to config.get_deep_dive_rounds() (currently {config.get_deep_dive_rounds()}). "
                         "Ignored by baseline methods.")


def cmd_build_index(args):
    with RunTimer("build_index"):
        if not args.only_unfiltered:
            store = rag_pipeline.build_index(filter_target=True)
            print(f"Filtered index: {len(store.metadata)} records")
        if not args.only_filtered:
            store = rag_pipeline.build_index(filter_target=False)
            print(f"Unfiltered index: {len(store.metadata)} records")


def cmd_generate(args):
    client = get_client(backend=args.backend, model=args.model)

    def _report(stage):
        print(f"[stage] {stage}")

    if args.method == "full_pipeline":
        pipeline = NewArchPipeline(client, backend_name=args.backend, model_name=args.model)
        record = InputRecord(
            text=args.text, id=args.id, language_hint=args.language, region_hint=args.region, rag_mode=args.rag_mode,
        )
        trace = pipeline.generate(
            record, accept_region_suggestion=args.accept_region_suggestion,
            persona_candidate_count=args.persona_count, deep_dive_rounds=args.deep_dive_rounds,
            query_strategy=args.query_strategy, filter_target=(args.rag_corpus == "filtered"),
            progress_callback=_report,
        )
    else:
        language = args.language or "en"
        train_df = None
        if args.method in ("few_shot", "few_shot_cot"):
            if args.language:
                train_df, _, language = dataset_loader.prepare_and_load(args.language, filter_target=True)
            else:
                print("No --language given for a few-shot method - proceeding with zero examples "
                      "(pass --language to draw TRAIN examples for that language).")
        _report(f"Running baseline method '{args.method}'...")
        trace = baseline_methods.run_baseline(
            client, args.method, args.text, language=language, backend_name=args.backend, model_name=args.model,
            train_df=train_df, num_examples=args.num_few_shot_examples, input_id=args.id,
        )

    print("\nCounter-narrative:", trace["counter_narrative"])
    print("Explanation:", trace["explanation"])
    append_jsonl(trace, config.TRACES_DIR / "single_example.jsonl")


def cmd_run_dataset(args, progress_callback=None, client=None):
    """progress_callback(done: int, total: int, row_id) - optional, called after
    each row finishes. Added so app.py's Streamlit "full validation-set
    evaluation" tab can drive a progress bar without duplicating this
    function's generation/CSV-writing logic - the CLI path (progress_callback
    left as None) is unaffected.

    client - optional pre-built ModelClient. Added for the same UI tab: when
    running multiple languages back-to-back in one process, building a fresh
    client per language would reload the entire model from scratch each time
    (irrelevant for the lightweight Ollama/OpenAI/HF-Inference clients, but a
    real cost for hf-transformers with a large local model) - pass the same
    client through instead. The CLI path (client=None) is unaffected."""
    filter_target = (args.rag_corpus == "filtered")
    source = getattr(args, "source", None)
    train_df, eval_df, language = dataset_loader.prepare_and_load(args.language, eval_split=args.split,
                                                                    filter_target=filter_target, source=source)
    if args.limit:
        eval_df = eval_df.head(args.limit)
    dataset_name = (source if source else
                     {"en": "english_codabench", "ta": "tamil_codabench"}.get(language, "ml_mtconan_kn"))
    records = dataset_loader.to_records(eval_df, dataset_name, language, args.split)
    # Only tag the filename when --source overrides the default for this
    # language (currently just en -> ml_mtconan_kn) - every other case keeps
    # its exact existing filename untouched, since a live long-running Colab
    # job's resume logic depends on the filename never changing underneath it.
    filename_language = f"{language}-{source}" if source else language

    client = client or get_client(backend=args.backend, model=args.model)
    pipeline = NewArchPipeline(client, backend_name=args.backend, model_name=args.model) if args.method == "full_pipeline" else None

    is_baseline = args.method != "full_pipeline"
    effective_rag_mode = "no_rag" if is_baseline else args.rag_mode
    effective_rag_corpus = "n_a" if is_baseline else args.rag_corpus
    safe_model = args.model.replace(":", "-").replace("/", "-")
    # Model MUST be part of the traces filename, not just the generations CSV
    # filename - otherwise two models run for the same language/split/method
    # would append to the same shared traces.jsonl, corrupting the per-model
    # route/persona/evidence metrics computed by evaluation.summarize_traces().
    traces_filename = f"{filename_language}_{args.split}_{args.method}_{safe_model}_traces.jsonl"

    out_path = (config.GENERATIONS_DIR /
                f"{filename_language}_{args.split}_{args.method}_{safe_model}_{effective_rag_mode}_{effective_rag_corpus}.csv")

    # Resume support: a Colab disconnect mid-run used to be unrecoverable -
    # re-running the same command started `rows` from empty and overwrote
    # out_path, discarding every row already completed even though the
    # per-row incremental write (below) had genuinely saved them. Now: read
    # back any existing output for this exact CSV, treat rows that finished
    # without an error as done, and only re-process what's missing/errored.
    # Row order in the id column is intentionally NOT assumed - lookup is by
    # id, and ids are forced to str on both sides since a purely-numeric id
    # column gets read back as int64 by pandas otherwise, breaking equality
    # against the str ids `to_records()` produces.
    rows_by_id = {}
    if out_path.exists():
        existing_df = pd.read_csv(out_path, dtype={"id": str})
        rows_by_id = {row["id"]: row for row in existing_df.to_dict("records")}
    # pandas reads a written "" back as NaN, and bool(nan) is True in Python -
    # a naive `not row.get("error")` would treat every past success as NaN
    # (truthy) and needlessly redo it. pd.isna() catches the NaN case; the
    # "" check covers a freshly-written-this-run row that hasn't round-
    # tripped through a CSV read yet.
    done_ids = {rid for rid, row in rows_by_id.items()
                if pd.isna(row.get("error")) or row.get("error") == ""}
    total_records = len(records)
    records = [rec for rec in records if str(rec["id"]) not in done_ids]
    if done_ids:
        print(f"Resuming {out_path.name}: {len(done_ids)}/{total_records} rows already done, "
              f"{len(records)} remaining.")

    with RunTimer(f"run_dataset_{language}_{args.split}_{args.method}", language=language, split=args.split,
                  method=args.method, num_rows=len(records)):
        for rec in records:
            try:
                if args.method == "full_pipeline":
                    input_record = InputRecord(
                        text=rec["hate_speech"] or "", id=rec["id"], language_hint=language, rag_mode=args.rag_mode,
                        metadata={"dataset_region": rec.get("region")},
                    )
                    trace = pipeline.generate(
                        input_record, filter_target=filter_target, query_strategy=args.query_strategy,
                        persona_candidate_count=args.persona_count, deep_dive_rounds=args.deep_dive_rounds,
                    )
                else:
                    trace = baseline_methods.run_baseline(
                        client, args.method, rec["hate_speech"] or "", language=language,
                        backend_name=args.backend, model_name=args.model, train_df=train_df,
                        num_examples=args.num_few_shot_examples, input_id=rec["id"],
                    )
                error = ""
            except Exception as exc:
                logger.exception("run-dataset row failed (id=%s)", rec["id"])
                trace = {"counter_narrative": "", "explanation": "", "selected_track": None}
                error = str(exc)

            # Keyed by str(id) and overwritten-in-place (not appended) so a
            # retried previously-errored row replaces its old entry instead
            # of duplicating it in the output CSV.
            rows_by_id[str(rec["id"])] = {
                "id": rec["id"], "language": language, "split": args.split, "method": args.method,
                "hate_text": rec["hate_speech"], "reference_counter_narrative": rec["reference_counter_narrative"],
                "generated_counter_narrative": trace.get("counter_narrative", ""),
                "explanation": trace.get("explanation", ""),
                "selected_track": trace.get("selected_track"),
                "rag_mode": effective_rag_mode, "rag_corpus": effective_rag_corpus,
                "model": args.model, "backend": args.backend, "error": error,
                "traces_file": traces_filename,
            }
            append_jsonl(trace, config.TRACES_DIR / traces_filename)
            # Write incrementally, not just once at the end - found this gap
            # while scoping a multi-day unattended batch run: previously a
            # crash/interruption partway through a long language (e.g. 270
            # English rows) lost every row's work, since nothing hit disk
            # until the whole loop finished. The I/O cost of rewriting a
            # growing CSV each row is trivial next to LLM generation time.
            pd.DataFrame(list(rows_by_id.values())).to_csv(out_path, index=False)
            if progress_callback:
                progress_callback(len(rows_by_id), total_records, rec["id"])

    print(f"Wrote {len(rows_by_id)} rows to {out_path}")
    return out_path


def cmd_evaluate(args):
    generation_files = sorted(config.GENERATIONS_DIR.glob("*.csv"))
    if not generation_files:
        print(f"No generation CSVs found under {config.GENERATIONS_DIR} - run `main.py run-dataset` first.")
        return

    client = get_client(backend=args.backend, model=args.model) if args.llm_judge else None
    all_summaries = []

    for csv_path in generation_files:
        summary_path = config.EVALUATIONS_DIR / f"{csv_path.stem}_evaluation_summary.csv"
        judged_path = config.EVALUATIONS_DIR / f"{csv_path.stem}_llm_judged.csv"
        # Resume support: LLM-judge calls cost real money, and judge_dataframe()
        # holds a whole file's rows in memory with no per-row save - a crash or
        # API-budget cutoff mid-file loses every already-paid-for call in that
        # file. Re-running `evaluate` used to unconditionally re-judge every
        # file, including ones that had already finished and been paid for.
        # Now: if this file's expected output already exists, reuse it instead.
        if summary_path.exists() and (not args.llm_judge or judged_path.exists()):
            existing_summary = pd.read_csv(summary_path).to_dict("records")[0]
            all_summaries.append(existing_summary)
            pd.DataFrame(all_summaries).to_csv(config.EVALUATIONS_DIR / "all_languages_summary.csv", index=False)
            print(f"Skipping {csv_path.name} (already evaluated) - reusing {summary_path.name}")
            continue

        df = pd.read_csv(csv_path)
        summary = evaluation.summarize(df)
        for col in ("language", "split", "method", "rag_mode", "rag_corpus", "model", "backend"):
            if col in df.columns and df[col].nunique() == 1:
                summary[col] = df[col].iloc[0]

        # main.py run-dataset (since this file's addition of --method) records the
        # exact traces filename per row in a "traces_file" column - read it directly
        # rather than guessing it from the CSV filename (method/model names can both
        # contain underscores, e.g. "few_shot_cot", making positional parsing unreliable).
        if "traces_file" in df.columns and df["traces_file"].nunique() == 1:
            candidate_traces = config.TRACES_DIR / df["traces_file"].iloc[0]
            if candidate_traces.exists():
                summary.update(evaluation.summarize_traces(candidate_traces))

        if args.llm_judge:
            judged_df = evaluation.judge_dataframe(client, df)
            judged_df.to_csv(config.EVALUATIONS_DIR / f"{csv_path.stem}_llm_judged.csv", index=False)
            for col in judged_df.columns:
                if col.startswith("judge_") and pd.api.types.is_numeric_dtype(judged_df[col]):
                    summary[f"avg_{col}"] = round(judged_df[col].mean(), 4)

        summary["source_file"] = csv_path.name
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        all_summaries.append(summary)
        # Rewritten after every file (not just once at the end) so a crash or
        # API-budget cutoff mid-batch still leaves a genuine, if partial,
        # combined summary on disk - same rationale as run-dataset's
        # incremental CSV write.
        combined_path = config.EVALUATIONS_DIR / "all_languages_summary.csv"
        pd.DataFrame(all_summaries).to_csv(combined_path, index=False)
        print(f"Evaluated {csv_path.name}: {summary.get('total_samples')} samples")

    print(f"Wrote combined summary ({len(all_summaries)} rows) to {config.EVALUATIONS_DIR / 'all_languages_summary.csv'}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="new_arch - Prosecutor/Defender/Judge counter-narrative pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build the FAISS RAG index(es)")
    group = p_build.add_mutually_exclusive_group()
    group.add_argument("--only-filtered", action="store_true")
    group.add_argument("--only-unfiltered", action="store_true")
    p_build.set_defaults(func=cmd_build_index)

    p_gen = sub.add_parser("generate", help="Run the pipeline on one comment")
    p_gen.add_argument("--text", required=True)
    p_gen.add_argument("--id", default=None)
    p_gen.add_argument("--language", default=None, choices=config.SUPPORTED_LANGUAGES)
    p_gen.add_argument("--region", default=None, help='e.g. "Indian" / "European" / omit for Auto/Unknown')
    p_gen.add_argument("--accept-region-suggestion", action="store_true",
                        help="Accept Case Analysis's region suggestion as confirmed if confident enough.")
    _add_common_model_args(p_gen)
    _add_common_pipeline_args(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    p_run = sub.add_parser("run-dataset", help="Run the pipeline over one language's eval split")
    p_run.add_argument("--language", required=True, choices=config.SUPPORTED_LANGUAGES)
    p_run.add_argument("--split", default="validation", choices=["validation", "test"])
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--source", default=None, choices=["ml_mtconan_kn"],
                        help="Override the default per-language dataset source. Only 'en' has a real "
                             "second option: --source ml_mtconan_kn runs the European ml_mtconan_kn/en "
                             "subset instead of the default english_codabench (Indian). Leave unset for "
                             "every existing language/behavior - this only adds a new path, never changes "
                             "the default one.")
    _add_common_model_args(p_run)
    _add_common_pipeline_args(p_run)
    p_run.set_defaults(func=cmd_run_dataset)

    p_eval = sub.add_parser("evaluate", help="Compute metrics over all outputs/generations/*.csv files")
    p_eval.add_argument("--llm-judge", action="store_true", help="Also run the 19-dimension LLM-judge rubric.")
    _add_common_model_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "run-dataset" and args.split == "test":
        print("WARNING: running on the TEST split. TEST should be used only once, for a final "
              "reported result - not repeatedly during development.")
    args.func(args)


if __name__ == "__main__":
    main()
