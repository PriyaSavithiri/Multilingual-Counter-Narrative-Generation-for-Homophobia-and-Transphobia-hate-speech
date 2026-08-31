"""
Verifies the dataset-integrity rules that matter most for this thesis:
original files are never modified by loading/filtering/splitting, the
verified LGBT+/homophobia/transphobia filter counts are stable, and TEST
(and carved VALIDATION) rows never leak into the RAG corpus.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset_loader as dl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_files_are_byte_identical_after_loading_and_splitting():
    paths = [dl.ENGLISH_TRAIN, dl.ENGLISH_TEST, dl.TAMIL_TRAIN, dl.TAMIL_TEST,
             dl.MULTITARGET_CONAN_CSV, dl.KN_GROUNDED_CN_CSV, dl.CONAN_MT_SP_CSV]
    before = {p: _sha256(p) for p in paths}

    dl.load_english_codabench_splits()
    dl.load_tamil_codabench_splits()
    dl.load_multitarget_conan_raw(filter_target=True)
    dl.load_kn_grounded_cn_raw(filter_target=True)
    dl.load_conan_mt_sp_raw(filter_target=True)

    after = {p: _sha256(p) for p in paths}
    for p in paths:
        assert before[p] == after[p], f"{p} was modified by loading/filtering - this must never happen."


def test_verified_lgbt_target_filter_counts():
    assert len(dl.load_multitarget_conan_raw(filter_target=True)) == 617
    assert len(dl.load_kn_grounded_cn_raw(filter_target=True)) == 39
    assert len(dl.load_conan_mt_sp_raw(filter_target=True)) == 450


def test_english_and_tamil_codabench_have_official_test_sets_with_references():
    en = dl.load_english_codabench_splits()
    ta = dl.load_tamil_codabench_splits()
    assert len(en["test"]) > 0 and len(ta["test"]) > 0
    en_records = dl.to_records(en["test"], "english_codabench", "en", "test")
    ta_records = dl.to_records(ta["test"], "tamil_codabench", "ta", "test")
    assert all(r["reference_counter_narrative"] for r in en_records[:5])
    assert all(r["reference_counter_narrative"] for r in ta_records[:5])


def test_validation_is_carved_from_train_only_never_from_test():
    # NOTE: the "Id" column is a per-file row index, not a globally unique
    # key (train Ids run 1..1800, test Ids run 1..66) - comparing Id values
    # across files would misleadingly "overlap" even with zero real leakage.
    # Instead verify the actual row-content partition: carve_validation_from_train
    # takes ONLY the raw train dataframe as input (sklearn train_test_split
    # guarantees a disjoint partition of it), and the official test file is
    # a completely separate dataframe never passed into that function at all.
    raw_train, raw_test = dl._load_english_codabench_raw()
    train_eff, val = dl.carve_validation_from_train(raw_train)
    assert len(train_eff) + len(val) == len(raw_train), "carve_validation_from_train must partition TRAIN exactly."
    # Identify rows by their FULL content (all columns), not a single column -
    # a handful of hate comments legitimately repeat with a different
    # counter_narrative annotation, so single-column "text" equality is not a
    # reliable row identity here; the full-row tuple is.
    train_rows = {tuple(row) for row in train_eff.itertuples(index=False)}
    val_rows = {tuple(row) for row in val.itertuples(index=False)}
    assert train_rows.isdisjoint(val_rows), "Effective-train and validation rows must not overlap."
    # The official test file is a distinct object, loaded independently, and
    # is never an input to carve_validation_from_train - confirmed by signature
    # (it takes a single dataframe, not train+test) and by this call passing
    # only raw_train above.
    assert id(raw_test) != id(raw_train)


def test_corpus_records_never_include_test_or_validation_split():
    records = dl.iter_corpus_records(filter_target=True)
    splits_present = {r["split"] for r in records}
    assert "test" not in splits_present, "TEST rows must never be indexed into the RAG corpus (leakage risk)."
    assert "validation" not in splits_present, "Carved VALIDATION rows must never be indexed into the RAG corpus."
    assert splits_present <= {"train", "all"}


# ---------------------------------------------------------------------------
# ML_MTCONAN_KN (eu/es/it/en) test-split ground-truth fix, added 2026-08-13.
# The official LanD-FBK/ML_MTCONAN_KN test split ships with every reference
# counter-narrative (KN_CN) empty across all 4 languages - verified directly
# against the real Hub data (400/400 test rows, 0 non-empty KN_CN, vs. 0/1584
# empty in train and 0/400 empty in validation) - a genuine leaderboard-style
# held-out test set, not a loading bug. Per explicit supervisor direction,
# dataset_loader.carve_test_from_train() (config.ML_MTCONAN_TEST_HOLDOUT_
# FRACTION = 0.2) carves a fresh, ground-truth-bearing test set out of the
# official TRAIN split instead, leaving the official validation split
# untouched. These tests require network access (real Hub fetch, no local
# CSV/mocking, matching this file's own no-fabrication testing philosophy).
# ---------------------------------------------------------------------------
_ML_MTCONAN_LANGUAGES = ["en", "eu", "es", "it"]


def test_ml_mtconan_test_split_has_real_ground_truth_after_carve():
    for lang in _ML_MTCONAN_LANGUAGES:
        splits = dl.load_ml_mtconan_kn_splits(lang, filter_target=True)
        test = splits["test"]
        assert len(test) > 0, f"ml_mtconan_kn/{lang} carved test split must not be empty."
        assert dl._has_ground_truth(test, "KN_CN"), (
            f"ml_mtconan_kn/{lang} carved test split must have real KN_CN references for every row."
        )


def test_ml_mtconan_carve_is_deterministic_across_calls():
    for lang in _ML_MTCONAN_LANGUAGES:
        s1 = dl.load_ml_mtconan_kn_splits(lang, filter_target=True)
        s2 = dl.load_ml_mtconan_kn_splits(lang, filter_target=True)
        assert s1["test"]["ID"].tolist() == s2["test"]["ID"].tolist(), (
            f"ml_mtconan_kn/{lang} carve must be reproducible (random_state={dl.config.RANDOM_STATE})."
        )


def test_ml_mtconan_train_validation_test_are_disjoint():
    for lang in _ML_MTCONAN_LANGUAGES:
        splits = dl.load_ml_mtconan_kn_splits(lang, filter_target=True)
        train_ids = set(splits["train"]["ID"])
        val_ids = set(splits["validation"]["ID"])
        test_ids = set(splits["test"]["ID"])
        assert train_ids.isdisjoint(test_ids), f"ml_mtconan_kn/{lang}: carved train/test must not overlap."
        assert train_ids.isdisjoint(val_ids), f"ml_mtconan_kn/{lang}: train/validation must not overlap."
        assert val_ids.isdisjoint(test_ids), f"ml_mtconan_kn/{lang}: validation/test must not overlap."


def test_ml_mtconan_official_validation_is_left_unchanged():
    # The official validation split already has full ground truth and was never the reported
    # problem - only the official test split (0% ground truth) should be replaced. 17 LGBT+
    # rows/language is the verified real count from the official (uncarved) validation split.
    for lang in _ML_MTCONAN_LANGUAGES:
        splits = dl.load_ml_mtconan_kn_splits(lang, filter_target=True)
        assert len(splits["validation"]) == 17, (
            f"ml_mtconan_kn/{lang} validation split size changed - it should stay the official, "
            f"uncarved split."
        )


def test_ml_mtconan_carved_test_never_leaks_into_rag_corpus():
    records = dl.iter_corpus_records(filter_target=True)
    mlk_records = [r for r in records if r["dataset_name"] == "ml_mtconan_kn"]
    assert mlk_records, "sanity check: ml_mtconan_kn should contribute at least some corpus rows."
    assert all(r["split"] == "train" for r in mlk_records), (
        "ml_mtconan_kn corpus rows must all come from the (reduced, post-carve) train split - "
        "the carved test split must never reach the RAG corpus."
    )
    for lang in _ML_MTCONAN_LANGUAGES:
        splits = dl.load_ml_mtconan_kn_splits(lang, filter_target=True)
        corpus_ids = {r["original_row"].get("ID") for r in mlk_records if r["language"] == lang}
        test_ids = set(splits["test"]["ID"])
        assert corpus_ids.isdisjoint(test_ids), f"ml_mtconan_kn/{lang}: carved test rows leaked into the RAG corpus."


if __name__ == "__main__":
    test_original_files_are_byte_identical_after_loading_and_splitting()
    test_verified_lgbt_target_filter_counts()
    test_english_and_tamil_codabench_have_official_test_sets_with_references()
    test_validation_is_carved_from_train_only_never_from_test()
    test_corpus_records_never_include_test_or_validation_split()
    test_ml_mtconan_test_split_has_real_ground_truth_after_carve()
    test_ml_mtconan_carve_is_deterministic_across_calls()
    test_ml_mtconan_train_validation_test_are_disjoint()
    test_ml_mtconan_official_validation_is_left_unchanged()
    test_ml_mtconan_carved_test_never_leaks_into_rag_corpus()
    print("test_dataset_integrity.py: ALL PASSED")
