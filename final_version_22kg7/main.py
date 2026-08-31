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
import prompts
import rag_pipeline
from model_api import get_client
from pipeline import NewArchPipeline
from schemas import InputRecord
from utils import get_logger, append_jsonl, RunTimer

logger = get_logger("new_arch.main")


def _tag_filename_for_final_cn_style(filename_language: str, final_cn_style: str) -> str:
    """v20 (the default) must keep the EXACT pre-existing filename for
    reproducibility and so resume-in-progress runs aren't affected; v22b
    must get a distinct one - without this, a v22b run for the same
    language/model/rag-mode as an already-completed v20 run would hit
    cmd_run_dataset's resume-from-existing-output logic, see every row ID
    already marked done, and silently skip regenerating them, producing a
    "v22b" output file that's actually 100% reused v20 rows."""
    if final_cn_style == "v20":
        return filename_language
    return f"{filename_language}-finalcn{final_cn_style}"


def _tag_filename_for_ablation_mode(filename_language: str, ablation_mode: str) -> str:
    """Keep full-mode filenames unchanged so an already-running full v22kg7
    job can continue/resume untouched; tag only the three ablation modes."""
    if (ablation_mode or "full") == "full":
        return filename_language
    return f"{filename_language}-ablation{ablation_mode}"


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
    p.add_argument("--final-cn-style", default="v20",
                    choices=["v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4", "v22b5", "v22b6", "v22kg",
                             "v22kg1", "v22kg2", "v22kg3", "v22kg4", "v22kg5", "v22kg6", "v22kg7"],
                    help="Which final counter-narrative prompt to use (see prompts.py section 7). "
                         "v20 (default, for reproducibility) = the original prompt, unchanged. "
                         "v22b = the experimental Mistral-quality-improvement prompt (2-3 sentence "
                         "structure: empathy -> direct rebuttal using Judge evidence -> respectful "
                         "closing; stricter anti-hallucination/anti-generic-slogan/anti-invented-"
                         "identity-term rules; simpler wording for eu/ta). v22b1 = v22b plus "
                         "trace-verified evidence-strictness rules (no citation-sounding phrases or "
                         "country/region legal claims beyond what approved_evidence exactly "
                         "supports, no narrowing hedged evidence into a specific claim, "
                         "cultural_guidance limited to tone only) and tighter eu/ta (exactly 2 "
                         "sentences)/it (max 3 sentences) limits. v22b2 = v22b1 plus an exact-"
                         "evidence-scope hotfix (mechanical, enumerated forbidden-phrase list for "
                         "country/region names not in approved_evidence; explicit source_type-aware "
                         "fact-vs-cultural-evidence usage rule; Basque rule against opposite-sex-"
                         "implying phrasing on same-sex-attraction claims). v22b3 = v22b2 plus an "
                         "explanation-safe hotfix (the explanation field gets its own detailed "
                         "method-level-only rules, so it can no longer relay an ungrounded claim "
                         "from the Judge's cultural_guidance/final_response_plan even when "
                         "counter_narrative stays clean) and EXACT source_type string matching "
                         "(only \"fact\"/\"web\" license factual claims; \"cultural\", \"culturale\", "
                         "missing, or malformed values are always tone-only). v22b4 = v22b3's "
                         "explanation rules unchanged, plus a stricter counter_narrative rule: a "
                         "\"cultural\"-tagged passage can no longer license a citation-style "
                         "\"science confirms...\" claim just because its own text mentions the word "
                         "\"science\" (fact/web source_type is now required, no exception), and an "
                         "explicit no-strengthening rule (a general claim like \"natural expression "
                         "of human diversity\" must not become a more specific one like \"not a "
                         "choice\"). v22b5 = v22b4's Italian rules and explanation rules unchanged, "
                         "plus a much stronger Tamil-specific note: must directly rebut the specific "
                         "hate-claim type (not just generic equality wording), bans a named list of "
                         "awkward/translated-sounding phrases, avoids listing identity categories not "
                         "in the input/evidence, caps unsupported theological claims, adds a Tamil-"
                         "named example to the explanation rules (no crediting \"scientific basis\"/"
                         "\"religious studies\" without exact fact/web evidence), and gives 4 safe "
                         "fallback templates matched to common hate-claim types. v22b6 = v22b5's "
                         "Italian rules and explanation rules unchanged, plus a rewritten Tamil note: "
                         "the 5 example responses are now explicit STYLE ANCHORS (illustrating tone/"
                         "directness, not mandatory templates to copy verbatim), bans more awkward/"
                         "translated-sounding phrases found in live review, requires silencing claims "
                         "to be named directly (not just generic dignity wording), avoids near-"
                         "theological claims about what God does/doesn't do unless evidence exactly "
                         "supports it, and explicitly forbids the Tamil explanation from switching to "
                         "English. v22kg = builds a lightweight, instance-specific Counter-Narrative "
                         "Knowledge Graph (see counter_narrative_kg.py - plain JSON nodes + typed "
                         "edges, no ontology/RDF/graph-DB, no LLM call) from this case's own Case "
                         "Analysis/Judge Plan/approved-evidence data, and layers it on top of v22b6's "
                         "unchanged prompt as an explicit grounding contract: required_rebuttal nodes "
                         "the response must address, forbidden_claim nodes (country/region, science/"
                         "research/statistics, legal, and religious/theological invention) it must "
                         "not make regardless of what the Judge's final_response_plan says, and "
                         "approved_factual_evidence vs approved_cultural_context nodes with strict "
                         "exact-source_type separation. Also stores the graph plus a non-blocking "
                         "kg_consistency diagnostic (does final_response_plan trip any forbidden_claim "
                         "node) in the trace JSONL. v22kg1 = builds on v22kg unchanged (reuses the same "
                         "KG builder, v22kg itself untouched) with 3 tightenings found necessary from a "
                         "real v22kg Tamil rerun: (1) a much more explicit, 13-rule KG-obedience prompt "
                         "block with concrete religion/theology and science/research ban-word lists "
                         "spelled out directly (not just the abstract forbidden_claim node name), "
                         "stating the KG overrides the Judge's plan on any conflict; (2) for Tamil only, "
                         "counter_narrative is exactly 1 short sentence instead of 2 (matching typical "
                         "reference length - the extra room in a second sentence is exactly where "
                         "unsupported religion/science claims kept appearing), with 5 new one-sentence "
                         "style anchors, plus an explicit no-code-mixing rule and a claim_type-vs-"
                         "disease-wording consistency rule; (3) kg_consistency is extended "
                         "(validate_kg_consistency_full, v22kg1 only - v22kg keeps the original "
                         "validate_kg_consistency unchanged) to check the actual generated "
                         "counter_narrative/explanation, not just final_response_plan, plus Tamil "
                         "code-mixing and disease/disorder-wording-vs-claim_type checks - trace gains "
                         "counter_narrative_violations/explanation_violations alongside "
                         "final_response_plan_violations. Diagnostic only, still non-blocking. "
                         "v22kg2 = corrects v22kg1's core framing: science/legal/religious/country/"
                         "cultural TERMS are never blindly banned, only UNSUPPORTED claims are - the "
                         "13-rule KG-obedience block is rewritten as 10 explicitly evidence-conditional "
                         "rules for every language (a claim is allowed whenever approved_factual_"
                         "evidence/approved_web_evidence explicitly supports it). Reuses v22kg1's proven "
                         "Tamil 1-sentence/anchor/ban handling unchanged (duplicated, not shared, so "
                         "v22kg1 itself stays untouched); adds a NEW Basque-specific fluency fix (avoids "
                         "unnatural \"dignitatea\"/\"berdintasunik gabeko arreta\"/\"harreman "
                         "homofobikoak\" wording found in real review, 3 new Basque style anchors, an "
                         "evidence-conditional science/psychology rule, and an explicit sexual-"
                         "orientation-immutability caution) without forcing Tamil's 1-sentence rule onto "
                         "Basque. kg_consistency is extended again (validate_kg_consistency_v2, v22kg2 "
                         "only - v22kg/v22kg1 keep their own validators unchanged) with full "
                         "multilingual (en/ta/eu/es/it) trigger coverage via has_supported_claim_family(), "
                         "plus a 5th claim family (orientation_immutability, not a KG node type - checked "
                         "directly against approved_factual_evidence text) - directly motivated by two "
                         "real v22kg1 Basque leaks: EU125's \"zientziaren eta psikologiaren arabera\" "
                         "(unsupported science claim, cultural-only evidence) and EU130's \"Ez da posible "
                         "sexualitatea aldatzea\" (unsupported orientation-immutability claim), neither "
                         "caught by v22kg1's Tamil/English-only trigger list. Only affects final CN "
                         "generation - ignored by baseline methods, which never call it. "
                         "v22kg3 = Basque one-sentence fluency fix + no-template-overfitting cleanup, on "
                         "top of v22kg2 (unchanged). Two parts: (1) a design-audit finding acted on here - "
                         "v22b6's shared EXPLANATION-FIELD RULES section names two rules after specific "
                         "validation-sample IDs (\"IT125-style case\", \"IT133-style case\") baked directly "
                         "into the runtime prompt text; v22kg3's own prompt replaces both with generic "
                         "claim-type descriptions (v20-v22b6/v22kg/v22kg1/v22kg2 keep the sample-ID wording "
                         "unchanged, per the explicit old-styles-unchanged constraint - this is a reported "
                         "finding, not a silent fix upstream). All newer style-specific examples "
                         "(Tamil/Basque anchors) are explicitly labelled by claim-type category and stated "
                         "as style anchors only, never sample-ID-labelled fixed templates - the model is "
                         "told directly not to copy one blindly or pick it by matching a sample ID. (2) "
                         "Basque gets the same one-sentence fix Tamil got under v22kg1 (extra room in a "
                         "second sentence is where broken grammar/unsupported claims kept appearing), an "
                         "expanded avoid-word list (\"Pertsonak guztiak\"/\"Pertsoak\"/\"sexuen "
                         "independentziak\"/etc., found unnatural in real review), 7 claim-type-labelled "
                         "style anchors, and a dedicated explanation-avoid-word list. kg_consistency is "
                         "extended again (validate_kg_consistency_v3, v22kg3 only - v22kg2 keeps its own "
                         "validator unchanged) with a real false-positive fix: Basque \"dio\" (an ordinary "
                         "auxiliary verb) was being matched against the Italian religion trigger "
                         "\"dio\"=God, because v22kg2 scanned every language's trigger phrases at once - "
                         "v22kg3 scans per-language instead, plus a negation guard so \"without science/"
                         "research...\" is no longer counted as an unsupported claim. Only affects final "
                         "CN generation - ignored by baseline methods, which never call it. "
                         "v22kg4 = tiny, prompt-only Basque fluency fallback on top of v22kg3 "
                         "(unchanged) - v22kg3 passed the structural KG gates but Basque real-review "
                         "output was still awkward/complex. KG architecture, validate_kg_consistency_v3, "
                         "RAG, Judge, Defender, tokenizer, model, and evaluation are all untouched - "
                         "v22kg4 reuses v22kg3's own validator unchanged. Adds an expanded Basque "
                         "avoid-list (\"Gizon-emakumezkoak diren edo ez\", \"orientazio sexualak\" "
                         "preferring \"sexu-orientazioa\", the \"pertsona guztiek ... merezi dituzte\" "
                         "verb-agreement error, \"naturaleko maitasuna\", \"iritziki gaizkileak\", "
                         "\"harremanen arteko aniztasuna\") plus a small set of claim-type FALLBACK style "
                         "anchors (exclusion, child/family rejection, conversion/change framing, "
                         "same-sex relationship stigma, generic dignity) and one explanation fallback, "
                         "explicitly framed as fallbacks to use only when unsure - not mandatory fixed "
                         "templates, not sample-specific. Only affects final CN generation - ignored by "
                         "baseline methods, which never call it. "
                         "v22kg5 = the final KG-grounded concise multilingual style, on top of v22kg4 "
                         "(unchanged - Tamil v22kg1 and Basque v22kg4 are already the best versions and "
                         "are NOT re-iterated here, just duplicated unchanged). Spanish/Italian/English "
                         "v22kg4 outputs were safe and fluent but too verbose (3-5 sentences reading like "
                         "mini-explanations) - v22kg5 tightens all three to 1-2 short sentences (prefer 1), "
                         "with new style anchors, and adds three English-specific guardrails found "
                         "necessary in real review: avoid unsupported biology/brain-development/hormone-"
                         "response/psychology framing unless approved factual/web evidence supports it, "
                         "avoid \"choose to love\" wording that could imply orientation is a choice, and do "
                         "not broaden the target from sexual orientation to gender identity unless the hate "
                         "comment itself targets both. The explanation-length rule is also tightened "
                         "universally (every language) from v22b6's original \"1-3 sentences\" to \"exactly "
                         "1 short, method-level sentence\". kg_consistency is extended once more "
                         "(validate_kg_consistency_v4, v22kg5 only - v22kg3/v22kg4 keep validate_kg_"
                         "consistency_v3 unchanged) with a negation-guard fix: an explanation saying "
                         "\"avoids unsupported scientific claims\" is no longer falsely flagged as making a "
                         "science claim. Only affects final CN generation - ignored by baseline methods, "
                         "which never call it. "
                         "v22kg6 = tiny final English cleanup on top of v22kg5 (unchanged - Spanish/Italian "
                         "already passed sanity validation and are out of scope; Tamil v22kg1/Basque v22kg4 "
                         "are explicitly not re-iterated). Fixes 3 real v22kg5 English issues found in "
                         "review: (1) \"choose to love\"/\"chooses to love\"/\"chosen love\"/\"who(m) they "
                         "choose to love\" are now hard-banned (can imply orientation is a choice) with "
                         "named alternatives; (2) mental-health/DSM/APA/science-style facts are now claim-"
                         "type-conditional - only used for disease/pathology, mental-illness, insanity, "
                         "defect/abnormality, or cure-framed conversion/change claims, never as a default "
                         "for child/family rejection, abortion, exclusion, slurs, relationship stigma, "
                         "silencing, or crime generalization, each of which gets its own claim-type-labelled "
                         "style anchor instead; (3) a new EXPLANATION DISCIPLINE paragraph strongly prefers "
                         "a generic method-level explanation (with a stated default and 3 alternates) and "
                         "names words to avoid unless exactly supported (\"scientific\", \"pseudoscientific\", "
                         "\"research\", \"biological\", \"medical\", \"legal\", \"religious\", \"cultural\", "
                         "country names, etc.) - fixes explanations that could trip kg_consistency even when "
                         "the counter_narrative itself was clean. Reuses validate_kg_consistency_v4 "
                         "unchanged (confirmed sufficient before implementing - no new validator was "
                         "written). Only affects final CN generation - ignored by baseline methods, which "
                         "never call it. "
                         "v22kg7 = v22kg6 + a small, deterministic, no-LLM-call KG safety fallback (English "
                         "only for now) - the prompt itself is byte-identical to v22kg6's (reused directly, "
                         "no new prompt code). Fixes 4 real v22kg6 English findings: cultural-only brain/"
                         "biology/hormone wording still leaked through; a child/abortion case used mental-"
                         "disorder evidence instead of a direct child-dignity rebuttal; unsupported \"not "
                         "supported by science\" wording appeared; and a genuinely relevant mental-disorder "
                         "rebuttal (hate comment said \"insane\", which the shared claim_type classifier "
                         "doesn't recognize as disease_or_pathology) was wrongly flagged. New "
                         "validate_kg_consistency_v5 (counter_narrative_kg.py, v22kg7 only - v4 untouched, "
                         "still serving v22kg5/v22kg6) adds English biology/brain/hormone/genetics triggers "
                         "and fixes the disease-mismatch check by also scanning the KG's own harmful_claim/"
                         "accusation node text (not just claim_type) for disease/insane/mental-illness "
                         "framing. After generation, apply_kg_safety_fallback validates the output: if only "
                         "the Judge's plan has a violation, output is unchanged; if only the explanation is "
                         "flagged, only the explanation is replaced with a generic method-level fallback; if "
                         "the counter_narrative itself is flagged, BOTH are replaced with a claim-type-safe "
                         "fallback pair, selected deterministically (never a sample ID) from 7 categories: "
                         "child/family rejection, conversion/change framing, trans biology/pseudoscience, "
                         "same-sex relationship stigma, disease/pathology (evidence-conditional wording), and "
                         "a generic dignity fallback - then revalidated. No repair loop (single pass, no "
                         "second model call). Trace gains a new kg_safety_fallback key (applied, reason, "
                         "original counter_narrative/explanation, fallback_type, post-fallback "
                         "kg_consistency) - pipeline.py copies it only when present, a no-op for every other "
                         "style. Only affects final CN generation - ignored by baseline methods, which never "
                         "call it.")
    p.add_argument("--ablation", default="full", choices=config.ABLATION_MODES,
                   help="Ablation mode for full_pipeline only: full, no_kg, no_debate, or no_persona. "
                        "Default full keeps the existing v22kg7 path unchanged.")


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
    prompts.FINAL_CN_STYLE = args.final_cn_style
    config.set_ablation_mode(getattr(args, "ablation", "full"))

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
    """Run one ablation mode, or the three ablation modes only.

    --run_all_ablations intentionally skips full because the full v22kg7 run may
    already be running in Colab; use --ablation full explicitly when needed.
    """
    if getattr(args, "run_all_ablations", False):
        shared_client = client or get_client(backend=args.backend, model=args.model)
        outputs = []
        for mode in config.ABLATION_BATCH_MODES:
            mode_args = argparse.Namespace(**vars(args))
            mode_args.run_all_ablations = False
            mode_args.ablation = mode
            outputs.append(_run_dataset_for_ablation(mode_args, progress_callback=progress_callback, client=shared_client))
        return outputs
    return _run_dataset_for_ablation(args, progress_callback=progress_callback, client=client)


def _run_dataset_for_ablation(args, progress_callback=None, client=None):
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
    # getattr with a "v20" default (not args.final_cn_style directly) - same
    # reasoning as `source` above: app.py builds its own argparse.Namespace
    # for the Streamlit UI path, which won't have this CLI-only attribute
    # unless app.py is also updated. Defaulting to "v20" keeps that path
    # exactly as before with zero changes needed there.
    final_cn_style = getattr(args, "final_cn_style", "v20")
    prompts.FINAL_CN_STYLE = final_cn_style
    ablation_mode = getattr(args, "ablation", "full")
    config.set_ablation_mode(ablation_mode)
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
    filename_language = _tag_filename_for_final_cn_style(filename_language, final_cn_style)
    filename_language = _tag_filename_for_ablation_mode(filename_language, ablation_mode)

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
                "ablation_mode": ablation_mode,
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

    needs_client = args.llm_judge or args.language_quality
    client = get_client(backend=args.backend, model=args.model) if needs_client else None
    all_summaries = []

    for csv_path in generation_files:
        summary_path = config.EVALUATIONS_DIR / f"{csv_path.stem}_evaluation_summary.csv"
        judged_path = config.EVALUATIONS_DIR / f"{csv_path.stem}_llm_judged.csv"
        lq_path = config.EVALUATIONS_DIR / f"{csv_path.stem}_language_quality.csv"
        # Resume support: LLM-judge (and language-quality) calls cost real
        # money/time, and hold a whole file's rows in memory with no per-row
        # save - a crash or API-budget cutoff mid-file loses every already-
        # paid-for call in that file. Re-running `evaluate` used to
        # unconditionally re-judge every file, including ones that had
        # already finished and been paid for. Now: if this file's expected
        # output(s) already exist for whichever flags were passed, reuse
        # them instead.
        if (summary_path.exists() and (not args.llm_judge or judged_path.exists())
                and (not args.language_quality or lq_path.exists())):
            existing_summary = pd.read_csv(summary_path).to_dict("records")[0]
            all_summaries.append(existing_summary)
            pd.DataFrame(all_summaries).to_csv(config.EVALUATIONS_DIR / "all_languages_summary.csv", index=False)
            print(f"Skipping {csv_path.name} (already evaluated) - reusing {summary_path.name}")
            continue

        df = pd.read_csv(csv_path)
        summary = evaluation.summarize(df)
        for col in ("language", "split", "method", "ablation_mode", "rag_mode", "rag_corpus", "model", "backend"):
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

        if args.language_quality:
            lq_df = evaluation.score_language_quality(client, df)
            lq_df.to_csv(lq_path, index=False)
            for col in ("lq_language_quality_score", "lq_fluency_score", "lq_terminology_score",
                        "lq_clarity_score", "lq_confidence_score"):
                if col in lq_df.columns:
                    vals = pd.to_numeric(lq_df[col], errors="coerce").dropna()
                    summary[f"avg_{col}"] = round(vals.mean(), 4) if len(vals) else None
            if "lq_quality_flag" in lq_df.columns:
                flagged = (lq_df["lq_quality_flag"] == "needs_review").sum()
                summary["language_quality_flagged_count"] = int(flagged)
                summary["language_quality_flagged_rate"] = round(flagged / len(lq_df), 4) if len(lq_df) else None

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
    p_run.add_argument("--run_all_ablations", action="store_true",
                       help="Run only the three ablation modes: no_kg, no_debate, no_persona. "
                            "Does not rerun full mode.")
    p_run.set_defaults(func=cmd_run_dataset)

    p_eval = sub.add_parser("evaluate", help="Compute metrics over all outputs/generations/*.csv files")
    p_eval.add_argument("--llm-judge", action="store_true", help="Also run the 12-dimension LLM-judge rubric.")
    p_eval.add_argument("--language-quality", action="store_true",
                         help="Also run the language-quality diagnostic (fluency/terminology/clarity/"
                              "target-language-match, with problem_spans + confidence_score) - "
                              "diagnostic only, never rewrites anything. Writes "
                              "<file>_language_quality.csv alongside the summary.")
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
