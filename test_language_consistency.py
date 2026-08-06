import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from schemas import InputRecord, InputValidationError, validate_input
from utils import detect_language


def test_trusted_hint_wins_over_everything_else():
    assert detect_language("some ambiguous text", dataset_language="eu") == "eu"


def test_tamil_script_detected():
    assert detect_language("இது ஒரு சோதனை வாக்கியம்") == "ta"


def test_basque_markers_detected():
    text = "gizonak eta emakumeak dira berdinak, hori da egia"
    assert detect_language(text) == "eu"


def test_falls_back_to_english_when_undetectable():
    assert detect_language("") == "en"


def test_all_five_supported_languages_are_declared():
    assert set(config.SUPPORTED_LANGUAGES) == {"en", "ta", "eu", "es", "it"}


def test_validate_input_rejects_empty_text():
    try:
        validate_input(InputRecord(text="   "))
        assert False, "empty text should have raised InputValidationError"
    except InputValidationError:
        pass


def test_validate_input_rejects_unsupported_language_hint():
    try:
        validate_input(InputRecord(text="hello", language_hint="fr"))
        assert False, "unsupported language_hint should have raised InputValidationError"
    except InputValidationError:
        pass


def test_validate_input_accepts_supported_language_hint():
    validate_input(InputRecord(text="hello", language_hint="ta"))  # must not raise


if __name__ == "__main__":
    test_trusted_hint_wins_over_everything_else()
    test_tamil_script_detected()
    test_basque_markers_detected()
    test_falls_back_to_english_when_undetectable()
    test_all_five_supported_languages_are_declared()
    test_validate_input_rejects_empty_text()
    test_validate_input_rejects_unsupported_language_hint()
    test_validate_input_accepts_supported_language_hint()
    print("test_language_consistency.py: ALL PASSED")
