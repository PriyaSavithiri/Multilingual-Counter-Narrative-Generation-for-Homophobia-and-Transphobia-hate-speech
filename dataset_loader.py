"""
new_arch dataset loading - clean-room re-implementation, reading directly
and read-only from data_orig/ (never modified/renamed/overwritten/copied;
original column names are never changed on disk - any renaming below
happens only in an in-memory dict/record, per-row).

This does not import the old project's src/data_loading/*.py - it is an
independent implementation of the same verified rules (per the "no runtime
dependency on old application code" restriction), re-deriving the exact
on-disk column names by reading the actual CSV headers rather than trusting
memory:

  english_codabench   train: Id,text,span,counter_narrative,label
                      test:  Id,text,class,span_words,CN
  tamil_codabench     train: V4_id,augmented_text,entityt_tokens,counter_narrative,label
                      test:  Id,text,class,extracted_span,counter_speech
  ml_mtconan_kn       MTCONAN_ID,HS,KN,KN_CN,PAIR_ID,SPLIT,LANG,TARGET,ID  (HF Hub - not
                      present locally under data_orig/ML_MTCONAN_KN)
  multitarget_conan   INDEX,HATE_SPEECH,COUNTER_NARRATIVE,TARGET,VERSION
  kn_grounded_cn      hate_speech,knowledge_sentence,counter_narrative,target
  conan_mt_sp         hate-speech,Human cn,...,target  (annotation-quality columns
                      in between are preserved in original_row, unused otherwise)

Only ml_mtconan_kn (eu/es/it, and en) and the two CodaBench sets (en/ta) are
used as per-language train/eval sources (via prepare_and_load). These are
also the ONLY sources for the RAG knowledge corpus (via iter_corpus_records):
english_codabench (Indian) + ml_mtconan_kn/en (European) for en, tamil_codabench
(Indian) for ta, and ml_mtconan_kn (European) for eu/es/it. multitarget_conan,
kn_grounded_cn, and conan_mt_sp are loaded/tested by name elsewhere (see
test_dataset_integrity.py) but are NOT part of the RAG corpus or any
language's train/eval set - the project's scope doesn't need them.

Target-group filtering (filter_target=True, the default) restricts the 4
multi-target datasets to homophobia/transphobia/LGBT+-relevant rows using
the same allow-list as before. The two CodaBench sets are already
homophobia/transphobia-scoped (LT-EDI shared task data) and are never
filtered.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from utils import require_file, get_logger

logger = get_logger("new_arch.dataset_loader")

LGBT_TARGET_VALUES = {"lgbt+", "lgbt", "homophobia", "transphobia"}

# ---------------------------------------------------------------------------
# Paths (read-only; re-resolved here rather than importing old config/dataset_paths.py)
# ---------------------------------------------------------------------------
_CODABENCH_DIR = config.EXTERNAL_DATASETS_DIR / "Codabench"
ENGLISH_TRAIN = _CODABENCH_DIR / "CN_English_Train.csv"
ENGLISH_TEST = _CODABENCH_DIR / "CN_English_Test.csv"
TAMIL_TRAIN = _CODABENCH_DIR / "CN_Tamil_Train.csv"
TAMIL_TEST = _CODABENCH_DIR / "CN_Tamil_Test.csv"
MULTITARGET_CONAN_CSV = config.EXTERNAL_DATASETS_DIR / "CONAN" / "Multitarget-CONAN" / "Multitarget-CONAN.csv"
KN_GROUNDED_CN_CSV = config.EXTERNAL_DATASETS_DIR / "CONAN" / "multitarget_KN_grounded_CN" / "multitarget_KN_grounded_CN.csv"
CONAN_MT_SP_CSV = config.EXTERNAL_DATASETS_DIR / "CONAN-MT-SP" / "CONAN-MT-SP.csv"
ML_MTCONAN_KN_DIR = config.EXTERNAL_DATASETS_DIR / "ML_MTCONAN_KN"


# ---------------------------------------------------------------------------
# Split helpers - purely in-memory, random_state=42, NEVER written to disk.
# Validation is carved from TRAIN only; TEST is never touched by carving.
# ---------------------------------------------------------------------------
def carve_validation_from_train(train_df: pd.DataFrame, frac: float = None):
    frac = config.VALIDATION_HOLDOUT_FRACTION if frac is None else frac
    train_eff, val = train_test_split(train_df, test_size=frac, random_state=config.RANDOM_STATE)
    return train_eff.reset_index(drop=True), val.reset_index(drop=True)


def make_reproducible_split(df: pd.DataFrame):
    """Datasets with no official split get a 70/15/15 train/val/test split,
    or 70/30 train/test (val=None) if there are fewer than
    MIN_ROWS_FOR_VALIDATION_SPLIT rows post-filter."""
    if len(df) >= config.MIN_ROWS_FOR_VALIDATION_SPLIT:
        train, temp = train_test_split(df, test_size=0.3, random_state=config.RANDOM_STATE)
        val, test = train_test_split(temp, test_size=0.5, random_state=config.RANDOM_STATE)
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.3, random_state=config.RANDOM_STATE)
    return train.reset_index(drop=True), None, test.reset_index(drop=True)


def _apply_target_filter(df: pd.DataFrame, target_col: str, filter_target: bool) -> pd.DataFrame:
    if not filter_target or target_col not in df.columns:
        return df
    mask = df[target_col].astype(str).str.strip().str.lower().isin(LGBT_TARGET_VALUES)
    return df[mask].reset_index(drop=True)


def _normalize_record(row: dict, dataset_name: str, language: str, split: str, row_id,
                       hate_speech, reference_cn, target, knowledge_text, original_row_index) -> dict:
    """The one normalized record shape used throughout new_arch. Original
    row/columns are preserved verbatim under original_row for traceability -
    nothing here mutates the source dataframe or the file on disk."""
    return {
        "id": str(row_id) if row_id is not None else None,
        "dataset_name": dataset_name,
        "language": language,
        "region": config.DATASET_CULTURAL_REGION.get(dataset_name),
        "split": split,
        "hate_speech": hate_speech,
        "reference_counter_narrative": reference_cn,
        "target": target,
        "knowledge_text": knowledge_text,
        "original_row_index": original_row_index,
        "original_row": row,
    }


# ---------------------------------------------------------------------------
# 1. English CodaBench (en) - already homophobia/transphobia-scoped
# ---------------------------------------------------------------------------
def _load_english_codabench_raw():
    train = pd.read_csv(require_file(ENGLISH_TRAIN, "English CodaBench train"))
    test = pd.read_csv(require_file(ENGLISH_TEST, "English CodaBench test"))
    return train, test


def load_english_codabench_splits():
    train, test = _load_english_codabench_raw()
    train_eff, val = carve_validation_from_train(train)
    return {"train": train_eff, "validation": val, "test": test}


def _english_codabench_to_records(df: pd.DataFrame, split: str) -> list:
    records = []
    for idx, row in df.iterrows():
        row_d = row.to_dict()
        if split == "test":
            hate, cn, row_id = row_d.get("text"), row_d.get("CN"), row_d.get("Id")
        else:
            hate, cn, row_id = row_d.get("text"), row_d.get("counter_narrative"), row_d.get("Id")
        records.append(_normalize_record(row_d, "english_codabench", "en", split, row_id, hate, cn, None, None, idx))
    return records


# ---------------------------------------------------------------------------
# 2. Tamil CodaBench (ta) - already homophobia/transphobia-scoped
# ---------------------------------------------------------------------------
def _load_tamil_codabench_raw():
    train = pd.read_csv(require_file(TAMIL_TRAIN, "Tamil CodaBench train"))
    test = pd.read_csv(require_file(TAMIL_TEST, "Tamil CodaBench test"))
    return train, test


def load_tamil_codabench_splits():
    train, test = _load_tamil_codabench_raw()
    train_eff, val = carve_validation_from_train(train)
    return {"train": train_eff, "validation": val, "test": test}


def _tamil_codabench_to_records(df: pd.DataFrame, split: str) -> list:
    records = []
    for idx, row in df.iterrows():
        row_d = row.to_dict()
        if split == "test":
            hate, cn, row_id = row_d.get("text"), row_d.get("counter_speech"), row_d.get("Id")
        else:
            hate, cn, row_id = row_d.get("augmented_text"), row_d.get("counter_narrative"), row_d.get("V4_id")
        records.append(_normalize_record(row_d, "tamil_codabench", "ta", split, row_id, hate, cn, None, None, idx))
    return records


# ---------------------------------------------------------------------------
# 3. ML_MTCONAN_KN (eu, es, it, + en) - multi-target, HF Hub fallback
# ---------------------------------------------------------------------------
_HUB_CACHE = None
_SPLIT_VALUE_MAP = {
    "train": "train", "training": "train",
    "dev": "validation", "validation": "validation", "val": "validation",
    "test": "test", "testing": "test",
}


def _load_ml_mtconan_kn_from_hub() -> pd.DataFrame:
    global _HUB_CACHE
    if _HUB_CACHE is not None:
        return _HUB_CACHE
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FileNotFoundError(
            "ML_MTCONAN_KN is not present locally under data_orig/ML_MTCONAN_KN and the "
            "`datasets` package is not installed to fetch it from the Hugging Face Hub "
            "(pip install datasets). This project never fabricates dataset content."
        ) from exc
    logger.info("Fetching LanD-FBK/ML_MTCONAN_KN from the Hugging Face Hub (cached for this process)...")
    hub_dataset = load_dataset("LanD-FBK/ML_MTCONAN_KN")
    frames = []
    for split_name, split_ds in hub_dataset.items():
        df = split_ds.to_pandas()
        if "SPLIT" not in df.columns:
            df["SPLIT"] = split_name
        frames.append(df)
    _HUB_CACHE = pd.concat(frames, ignore_index=True)
    return _HUB_CACHE


def _load_ml_mtconan_kn_raw(language: str) -> pd.DataFrame:
    per_language_csv = ML_MTCONAN_KN_DIR / f"{language}.csv"
    if per_language_csv.exists():
        df = pd.read_csv(per_language_csv)
    elif ML_MTCONAN_KN_DIR.exists() and any(ML_MTCONAN_KN_DIR.glob("*.csv")):
        combined = sorted(ML_MTCONAN_KN_DIR.glob("*.csv"))[0]
        df = pd.read_csv(combined)
    else:
        df = _load_ml_mtconan_kn_from_hub()
    if "LANG" in df.columns:
        df = df[df["LANG"].astype(str).str.lower().str.startswith(language.lower())].reset_index(drop=True)
    return df


def load_ml_mtconan_kn_splits(language: str, filter_target: bool = True) -> dict:
    df = _load_ml_mtconan_kn_raw(language)
    df = _apply_target_filter(df, "TARGET", filter_target)
    if "SPLIT" not in df.columns or df.empty:
        train, val, test = make_reproducible_split(df)
        return {"train": train, "validation": val, "test": test}
    mapped = df["SPLIT"].astype(str).str.lower().map(lambda v: _SPLIT_VALUE_MAP.get(v, v))
    train = df[mapped == "train"].reset_index(drop=True)
    val = df[mapped == "validation"].reset_index(drop=True)
    test = df[mapped == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        train, val, test = make_reproducible_split(df)
        return {"train": train, "validation": val, "test": test}
    return {"train": train, "validation": (val if not val.empty else None), "test": test}


def _ml_mtconan_kn_to_records(df: pd.DataFrame, language: str, split: str) -> list:
    records = []
    for idx, row in df.iterrows():
        row_d = row.to_dict()
        records.append(_normalize_record(
            row_d, "ml_mtconan_kn", language, split, row_d.get("ID"),
            row_d.get("HS"), row_d.get("KN_CN"), row_d.get("TARGET"), row_d.get("KN"), idx,
        ))
    return records


# ---------------------------------------------------------------------------
# 4. Multitarget-CONAN (en) - RAG corpus source only, no official split
# ---------------------------------------------------------------------------
def load_multitarget_conan_raw(filter_target: bool = True) -> pd.DataFrame:
    df = pd.read_csv(require_file(MULTITARGET_CONAN_CSV, "Multitarget-CONAN"))
    return _apply_target_filter(df, "TARGET", filter_target)


def _multitarget_conan_to_records(df: pd.DataFrame, split: str = "all") -> list:
    records = []
    for idx, row in df.iterrows():
        row_d = row.to_dict()
        records.append(_normalize_record(
            row_d, "multitarget_conan", "en", split, row_d.get("INDEX"),
            row_d.get("HATE_SPEECH"), row_d.get("COUNTER_NARRATIVE"), row_d.get("TARGET"), None, idx,
        ))
    return records


# ---------------------------------------------------------------------------
# 5. KN-grounded-CN (en) - RAG corpus source only, no official split
# ---------------------------------------------------------------------------
def load_kn_grounded_cn_raw(filter_target: bool = True) -> pd.DataFrame:
    df = pd.read_csv(require_file(KN_GROUNDED_CN_CSV, "KN-grounded-CN"))
    return _apply_target_filter(df, "target", filter_target)


def _kn_grounded_cn_to_records(df: pd.DataFrame, split: str = "all") -> list:
    records = []
    for idx, row in df.iterrows():
        row_d = row.to_dict()
        records.append(_normalize_record(
            row_d, "kn_grounded_cn", "en", split, None,
            row_d.get("hate_speech"), row_d.get("counter_narrative"), row_d.get("target"),
            row_d.get("knowledge_sentence"), idx,
        ))
    return records


# ---------------------------------------------------------------------------
# 6. CONAN-MT-SP (es) - RAG corpus source only, no official split
# ---------------------------------------------------------------------------
def load_conan_mt_sp_raw(filter_target: bool = True) -> pd.DataFrame:
    df = pd.read_csv(require_file(CONAN_MT_SP_CSV, "CONAN-MT-SP"))
    return _apply_target_filter(df, "target", filter_target)


def _conan_mt_sp_to_records(df: pd.DataFrame, split: str = "all") -> list:
    records = []
    for idx, row in df.iterrows():
        row_d = row.to_dict()
        records.append(_normalize_record(
            row_d, "conan_mt_sp", "es", split, None,
            row_d.get("hate-speech"), row_d.get("Human cn"), row_d.get("target"), None, idx,
        ))
    return records


# ---------------------------------------------------------------------------
# Per-language train/eval registry - ONLY these 3 dataset families ever serve
# as a language's own train/eval source, and they are also the only sources
# used for the RAG corpus (see iter_corpus_records).
# ---------------------------------------------------------------------------
def prepare_and_load(language: str, eval_split: str = "validation", filter_target: bool = True,
                      source: str = None):
    """Returns (train_df, eval_df, language). Raises ValueError for an
    unsupported language or an eval_split that isn't available (e.g. a
    dataset small enough that make_reproducible_split returned no
    validation slice).

    source=None keeps every existing language's default routing untouched
    (en -> english_codabench, ta -> tamil_codabench, eu/es/it -> ml_mtconan_kn).
    source="ml_mtconan_kn" is currently only meaningful for language="en" -
    it routes English to the European ml_mtconan_kn/en subset instead of the
    default Indian english_codabench, verified to carry 596 raw rows the
    same as eu/es/it. Ignored for languages that already default to
    ml_mtconan_kn (eu/es/it) since there's nothing to override there."""
    if language not in config.SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Supported: {config.SUPPORTED_LANGUAGES}")
    if eval_split not in ("validation", "test"):
        raise ValueError("eval_split must be 'validation' or 'test' (test is opt-in and should be used only once).")

    if source == "ml_mtconan_kn" and language in ("en", "eu", "es", "it"):
        splits = load_ml_mtconan_kn_splits(language, filter_target=filter_target)
    elif language == "en":
        splits = load_english_codabench_splits()
    elif language == "ta":
        splits = load_tamil_codabench_splits()
    else:  # eu, es, it
        splits = load_ml_mtconan_kn_splits(language, filter_target=filter_target)

    eval_df = splits.get(eval_split)
    if eval_df is None or len(eval_df) == 0:
        raise ValueError(f"No '{eval_split}' data available for language '{language}'.")
    return splits["train"], eval_df, language


def to_records(df: pd.DataFrame, dataset_name: str, language: str, split: str) -> list:
    """Dispatches to the right per-dataset normalizer."""
    if dataset_name == "english_codabench":
        return _english_codabench_to_records(df, split)
    if dataset_name == "tamil_codabench":
        return _tamil_codabench_to_records(df, split)
    if dataset_name == "ml_mtconan_kn":
        return _ml_mtconan_kn_to_records(df, language, split)
    if dataset_name == "multitarget_conan":
        return _multitarget_conan_to_records(df, split)
    if dataset_name == "kn_grounded_cn":
        return _kn_grounded_cn_to_records(df, split)
    if dataset_name == "conan_mt_sp":
        return _conan_mt_sp_to_records(df, split)
    raise ValueError(f"Unknown dataset_name '{dataset_name}'")


def get_few_shot_examples(train_df: pd.DataFrame, dataset_name: str, language: str, n: int = 3) -> list:
    """(hate_speech, reference_cn) pairs drawn from TRAIN only, first n
    non-empty rows (no shuffling - matches the old project's behaviour)."""
    records = to_records(train_df.head(max(n * 3, n)), dataset_name, language, "train")
    pairs = [(r["hate_speech"], r["reference_counter_narrative"]) for r in records
             if r["hate_speech"] and r["reference_counter_narrative"]]
    return pairs[:n]


# ---------------------------------------------------------------------------
# RAG corpus source - gathers TRAIN rows from ONLY the 3 in-scope dataset
# families (english_codabench, tamil_codabench, ml_mtconan_kn). TEST and
# carved VALIDATION are NEVER included here, to avoid the retriever ever
# surfacing a held-out gold reference as "evidence" for that same example.
#
# en gets 2 corpus sources - english_codabench (Indian) + ml_mtconan_kn/en
# (European); ta gets 1 - tamil_codabench (Indian); eu/es/it get 1 each -
# ml_mtconan_kn (European). multitarget_conan, kn_grounded_cn, and
# conan_mt_sp are intentionally excluded - out of this project's scope.
# ---------------------------------------------------------------------------
def iter_corpus_records(filter_target: bool = True) -> list:
    records = []

    en_splits = load_english_codabench_splits()
    records += to_records(en_splits["train"], "english_codabench", "en", "train")

    ta_splits = load_tamil_codabench_splits()
    records += to_records(ta_splits["train"], "tamil_codabench", "ta", "train")

    for lang in ("en", "eu", "it", "es"):
        try:
            mlk_splits = load_ml_mtconan_kn_splits(lang, filter_target=filter_target)
        except FileNotFoundError as exc:
            logger.warning("Skipping ml_mtconan_kn/%s for corpus build: %s", lang, exc)
            continue
        records += to_records(mlk_splits["train"], "ml_mtconan_kn", lang, "train")

    return records
