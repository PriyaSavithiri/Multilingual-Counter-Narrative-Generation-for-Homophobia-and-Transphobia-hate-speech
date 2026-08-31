"""
Covers main.py's _tag_filename_for_final_cn_style() - v22b's generations-CSV/
traces-JSONL filename tagging. Real bug found while documenting v22b's Colab
run instructions: without this, a --final-cn-style v22b run for the same
language/model/rag-mode as an already-completed v20 run would hit
cmd_run_dataset's resume-from-existing-output logic, see every row ID
already marked done, and silently skip regenerating them - producing a
"v22b" output file that's actually 100% reused v20 rows, corrupting the
exact before/after comparison this flag exists for.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _tag_filename_for_final_cn_style


def test_v20_keeps_filename_unchanged():
    """v20 is the default and must produce the EXACT pre-existing filename -
    reproducibility, and so a Colab run started before this flag existed can
    still be resumed without its output file suddenly moving."""
    assert _tag_filename_for_final_cn_style("eu", "v20") == "eu"
    assert _tag_filename_for_final_cn_style("en-ml_mtconan_kn", "v20") == "en-ml_mtconan_kn"


def test_v22b_gets_a_distinct_filename():
    assert _tag_filename_for_final_cn_style("eu", "v22b") == "eu-finalcnv22b"


def test_v22b_and_v20_never_collide():
    v20_name = _tag_filename_for_final_cn_style("es", "v20")
    v22b_name = _tag_filename_for_final_cn_style("es", "v22b")
    assert v20_name != v22b_name


def test_tagging_composes_with_source_override_tagging():
    """--source (e.g. en -> en-ml_mtconan_kn) is applied before this - both
    tags must survive together, not overwrite each other."""
    base = "en-ml_mtconan_kn"
    assert _tag_filename_for_final_cn_style(base, "v22b") == "en-ml_mtconan_kn-finalcnv22b"


if __name__ == "__main__":
    test_v20_keeps_filename_unchanged()
    test_v22b_gets_a_distinct_filename()
    test_v22b_and_v20_never_collide()
    test_tagging_composes_with_source_override_tagging()
    print("test_main_final_cn_style.py: ALL PASSED")
