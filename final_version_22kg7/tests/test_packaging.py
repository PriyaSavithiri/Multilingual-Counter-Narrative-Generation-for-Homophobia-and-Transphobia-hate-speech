"""
Verifies new_arch/requirements.txt is standalone - installable from a
new_arch-only checkout (e.g. the colab_transfer_*.zip bundles, which contain
new_arch/ alone) without needing a parent ../requirements.txt to exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent / "requirements.txt"


def test_requirements_txt_does_not_reference_parent_file():
    content = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    for line in lines:
        assert not line.startswith("-r "), (
            f"requirements.txt has an active -r include ({line!r}) - this breaks in a "
            "new_arch-only checkout where the referenced file doesn't exist."
        )


def test_requirements_txt_lists_core_dependencies_directly():
    content = REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
    for package in ["pandas", "torch", "transformers", "langchain-core", "faiss-cpu", "sentence-transformers"]:
        assert package in content, f"requirements.txt is missing a direct entry for {package!r}."


if __name__ == "__main__":
    test_requirements_txt_does_not_reference_parent_file()
    test_requirements_txt_lists_core_dependencies_directly()
    print("test_packaging.py: ALL PASSED")
