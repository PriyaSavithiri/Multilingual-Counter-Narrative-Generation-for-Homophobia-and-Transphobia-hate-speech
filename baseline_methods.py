"""
Simple, non-agentic baseline methods - zero-shot, few-shot, CoT, and
few-shot+CoT - evaluated alongside the full Prosecutor/Defender/Judge
pipeline (method="full_pipeline", handled by pipeline.py) as comparison
points for the thesis's evaluation tables. None of these use RAG, personas,
or debate; each is a single LLM call (or, for CoT variants, one call whose
final line is extracted via utils.extract_final_counter_narrative()).

Kept in a separate file from pipeline.py/agents/ since these are simple
prompting strategies, not architecture stages - mixing them into the
Prosecutor/Defender/Judge orchestration would blur what's actually being
compared against what.
"""
import time

import config
import dataset_loader
import prompts
from schemas import FinalOutput
from utils import extract_final_counter_narrative, get_logger

logger = get_logger("new_arch.baseline_methods")

BASELINE_METHODS = ["zero_shot", "few_shot", "cot", "few_shot_cot"]
ALL_METHODS = BASELINE_METHODS + ["full_pipeline"]

# Maps a language to the dataset_name to_records()/get_few_shot_examples()
# need - same routing as dataset_loader.prepare_and_load / main.py's run-dataset.
_DATASET_NAME_FOR_LANGUAGE = {"en": "english_codabench", "ta": "tamil_codabench"}


def dataset_name_for_language(language: str) -> str:
    return _DATASET_NAME_FOR_LANGUAGE.get(language, "ml_mtconan_kn")


def _get_examples(train_df, language: str, num_examples: int) -> list:
    if train_df is None or num_examples <= 0:
        return []
    return dataset_loader.get_few_shot_examples(train_df, dataset_name_for_language(language), language, n=num_examples)


def run_baseline(client, method: str, comment: str, language: str = "en", backend_name: str = "",
                  model_name: str = "", train_df=None, num_examples: int = 3, input_id: str = None) -> dict:
    """Returns the same dict shape as pipeline.NewArchPipeline.generate()
    (a FinalOutput.to_dict(), with hate_type/selected_track/rag_mode/
    evidence_trace left as None/[] since baselines don't classify hate type,
    route, retrieve, or verify evidence)."""
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method '{method}'. Supported: {BASELINE_METHODS}")

    started = time.time()
    errors = []
    text = ""
    try:
        if method == "zero_shot":
            raw = client.generate(prompts.build_prompt_zero(comment), temperature=config.DEFAULT_TEMPERATURE,
                                   max_tokens=config.DEFAULT_MAX_TOKENS)
            text = raw.strip()
        elif method == "few_shot":
            examples = _get_examples(train_df, language, num_examples)
            raw = client.generate(prompts.build_prompt_few(comment, examples), temperature=config.DEFAULT_TEMPERATURE,
                                   max_tokens=config.DEFAULT_MAX_TOKENS)
            text = raw.strip()
        elif method == "cot":
            raw = client.generate(prompts.build_prompt_cot(comment), temperature=config.DEFAULT_TEMPERATURE,
                                   max_tokens=config.DEFAULT_MAX_TOKENS)
            text = extract_final_counter_narrative(raw)
        elif method == "few_shot_cot":
            examples = _get_examples(train_df, language, num_examples)
            raw = client.generate(prompts.build_prompt_few_shot_cot(comment, examples),
                                   temperature=config.DEFAULT_TEMPERATURE, max_tokens=config.DEFAULT_MAX_TOKENS)
            text = extract_final_counter_narrative(raw)
    except Exception as exc:
        logger.exception("Baseline method '%s' failed", method)
        errors.append(str(exc))

    output = FinalOutput(
        input_id=input_id, language=language, hate_type=None, selected_track=None, rag_mode="no_rag",
        counter_narrative=text, explanation="",
        metadata={
            "backend": backend_name, "model": model_name, "method": method,
            "duration_seconds": round(time.time() - started, 2), "errors": errors,
        },
    )
    return output.to_dict()
