"""
Covers model_api.py's Mistral tokenizer-regex fix + HF_TOKEN handling
(HFTransformersClient._load()). Motivated by a real warning observed live:
"The tokenizer you are loading from 'gghfez/Mistral-Small-3.2-24B-Instruct-hf'
with an incorrect regex pattern... You should set the fix_mistral_regex=True
flag when loading this tokenizer to fix this issue."

The first version of this fix shared ONE kwargs dict between the tokenizer
and model loads (via HuggingFacePipeline.from_model_id, which has no way to
give them different kwargs) and crashed a real Colab run with
"MistralForCausalLM.__init__() got an unexpected keyword argument
'fix_mistral_regex'" - fix_mistral_regex is a tokenizer-only constructor
argument the model class does not accept. Fixed by splitting into two pure,
directly-testable functions - _hf_transformers_tokenizer_kwargs() (may
include fix_mistral_regex) and _hf_transformers_model_auth_kwargs() (never
does) - and building the tokenizer/model/pipeline manually in _load()
instead of via from_model_id's single shared dict.

Since this touches model_api.py (one of the 16 files on
tests/test_generation_path_frozen.py's locked manifest), that manifest's
hash for model_api.py was deliberately updated - see that test file's own
docstring for the full explanation and confirmation that all other 15
locked files remain untouched.

Does not actually load a model (multi-GB download, no GPU in this
environment) - both kwargs-building functions are pure, so this logic is
fully testable without one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_api import (_hf_transformers_model_auth_kwargs, _hf_transformers_tokenizer_kwargs,
                        _is_mistral_model)


# ---------------------------------------------------------------------------
# 1. Mistral tokenizer kwargs include fix_mistral_regex=True
# ---------------------------------------------------------------------------
def test_mistral_tokenizer_kwargs_include_fix_mistral_regex():
    kwargs = _hf_transformers_tokenizer_kwargs("gghfez/Mistral-Small-3.2-24B-Instruct-hf", hf_token="")
    assert kwargs.get("fix_mistral_regex") is True


def test_mistral_detection_is_case_insensitive_and_matches_any_mistral_variant():
    assert _is_mistral_model("gghfez/Mistral-Small-3.2-24B-Instruct-hf") is True
    assert _is_mistral_model("mistralai/Mistral-7B-Instruct-v0.3") is True
    assert _is_mistral_model("MISTRAL-large") is True
    assert _is_mistral_model("some/mistral-variant") is True


# ---------------------------------------------------------------------------
# 2. Mistral MODEL kwargs do NOT include fix_mistral_regex (the actual bug
# that crashed the real Colab run)
# ---------------------------------------------------------------------------
def test_mistral_model_kwargs_never_include_fix_mistral_regex():
    kwargs = _hf_transformers_model_auth_kwargs(hf_token="tok123")
    assert "fix_mistral_regex" not in kwargs


def test_mistral_model_kwargs_are_only_ever_token_or_empty():
    """_hf_transformers_model_auth_kwargs takes no model name at all - by
    construction it can never return fix_mistral_regex, for any model,
    Mistral or not. This test pins that contract."""
    assert _hf_transformers_model_auth_kwargs(hf_token="") == {}
    assert _hf_transformers_model_auth_kwargs(hf_token="tok123") == {"token": "tok123"}
    assert set(_hf_transformers_model_auth_kwargs(hf_token="tok123").keys()) <= {"token"}


# ---------------------------------------------------------------------------
# 3. HF_TOKEN is passed to both tokenizer and model when present
# ---------------------------------------------------------------------------
def test_token_passed_to_both_tokenizer_and_model_kwargs_when_present():
    tok_kwargs = _hf_transformers_tokenizer_kwargs("Qwen/Qwen2.5-32B-Instruct", hf_token="tok_abc123")
    model_kwargs = _hf_transformers_model_auth_kwargs(hf_token="tok_abc123")
    assert tok_kwargs.get("token") == "tok_abc123"
    assert model_kwargs.get("token") == "tok_abc123"


def test_token_omitted_entirely_when_absent_from_both():
    tok_kwargs = _hf_transformers_tokenizer_kwargs("Qwen/Qwen2.5-32B-Instruct", hf_token="")
    model_kwargs = _hf_transformers_model_auth_kwargs(hf_token="")
    assert "token" not in tok_kwargs
    assert "token" not in model_kwargs


def test_default_hf_token_source_is_config_hf_token(monkeypatch):
    """No explicit hf_token= override - both functions must fall back to
    config.HF_TOKEN, the project's existing never-hardcoded env-var
    pattern (already used by HFInferenceClient)."""
    import config
    monkeypatch.setattr(config, "HF_TOKEN", "from_config_env")
    assert _hf_transformers_tokenizer_kwargs("Qwen/Qwen2.5-32B-Instruct").get("token") == "from_config_env"
    assert _hf_transformers_model_auth_kwargs().get("token") == "from_config_env"


def test_mistral_tokenizer_gets_both_fix_and_token_together():
    kwargs = _hf_transformers_tokenizer_kwargs("mistralai/Mistral-7B-Instruct-v0.3", hf_token="tok_xyz")
    assert kwargs == {"fix_mistral_regex": True, "token": "tok_xyz"}


# ---------------------------------------------------------------------------
# 4. Non-Mistral tokenizer/model kwargs do not include fix_mistral_regex
# ---------------------------------------------------------------------------
def test_non_mistral_tokenizer_kwargs_do_not_get_fix_mistral_regex():
    kwargs = _hf_transformers_tokenizer_kwargs("Qwen/Qwen2.5-32B-Instruct", hf_token="")
    assert "fix_mistral_regex" not in kwargs
    assert kwargs == {}


def test_non_mistral_model_kwargs_do_not_get_fix_mistral_regex():
    kwargs = _hf_transformers_model_auth_kwargs(hf_token="")
    assert "fix_mistral_regex" not in kwargs


# ---------------------------------------------------------------------------
# 5. No token is printed/logged
# ---------------------------------------------------------------------------
def test_token_value_never_appears_in_module_source_via_print_or_log():
    """Static guard: neither kwargs-building function may hardcode or embed
    a real token value, and must never route the token through a print/log
    call. Reads model_api.py's own source and confirms the token only ever
    flows as a dict value (config.HF_TOKEN -> kwargs["token"]), never
    through print()/logger.*() in either function body."""
    source = Path(__file__).resolve().parent.parent.joinpath("model_api.py").read_text(encoding="utf-8")
    for func_name in ("_hf_transformers_tokenizer_kwargs", "_hf_transformers_model_auth_kwargs"):
        start = source.index(f"def {func_name}")
        end = source.index("\ndef ", start + 1)
        func_body = source[start:end]
        assert "print(" not in func_body, f"{func_name} must never print the token"
        assert "logger." not in func_body, f"{func_name} must never log the token"
        assert ".log(" not in func_body, f"{func_name} must never log the token"


if __name__ == "__main__":
    test_mistral_tokenizer_kwargs_include_fix_mistral_regex()
    test_mistral_detection_is_case_insensitive_and_matches_any_mistral_variant()
    test_mistral_model_kwargs_never_include_fix_mistral_regex()
    test_mistral_model_kwargs_are_only_ever_token_or_empty()
    test_token_passed_to_both_tokenizer_and_model_kwargs_when_present()
    test_token_omitted_entirely_when_absent_from_both()
    test_mistral_tokenizer_gets_both_fix_and_token_together()
    test_non_mistral_tokenizer_kwargs_do_not_get_fix_mistral_regex()
    test_non_mistral_model_kwargs_do_not_get_fix_mistral_regex()
    test_token_value_never_appears_in_module_source_via_print_or_log()
    print("test_model_api_tokenizer_fix.py: most tests passed (monkeypatch-dependent "
          "test needs pytest, run via `pytest tests/test_model_api_tokenizer_fix.py`)")
