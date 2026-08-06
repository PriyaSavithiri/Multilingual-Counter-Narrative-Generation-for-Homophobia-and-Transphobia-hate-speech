import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import baseline_methods
from model_api import ModelClient


class _StubClient(ModelClient):
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is not None:
            raise AssertionError("Baseline methods never request structured JSON output.")
        if "Step 1" in messages:
            return ("Step 1 (Claim): x\nStep 2 (Target): y\nStep 3 (Strategy): z\nStep 4: w\n"
                     "Counter-narrative: The CoT final response.")
        return "The plain zero/few-shot response."


class _ExplodingClient(ModelClient):
    backend_name = "exploding"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        raise RuntimeError("simulated backend failure")


def test_all_four_baseline_methods_are_registered():
    assert baseline_methods.BASELINE_METHODS == ["zero_shot", "few_shot", "cot", "few_shot_cot"]
    assert "full_pipeline" in baseline_methods.ALL_METHODS


def test_zero_shot_returns_plain_text():
    result = baseline_methods.run_baseline(_StubClient(), "zero_shot", "Being gay is unnatural.", language="en")
    assert result["counter_narrative"] == "The plain zero/few-shot response."
    assert result["hate_type"] is None and result["selected_track"] is None and result["rag_mode"] == "no_rag"
    assert result["metadata"]["method"] == "zero_shot"


def test_cot_extracts_only_the_final_marker_line():
    result = baseline_methods.run_baseline(_StubClient(), "cot", "Being gay is unnatural.", language="en")
    assert result["counter_narrative"] == "The CoT final response."
    assert "Step 1" not in result["counter_narrative"]


def test_few_shot_examples_drawn_from_train_only():
    import dataset_loader as dl
    train_df = dl.load_english_codabench_splits()["train"]
    result = baseline_methods.run_baseline(_StubClient(), "few_shot", "Being gay is unnatural.", language="en",
                                            train_df=train_df, num_examples=2)
    assert result["counter_narrative"] == "The plain zero/few-shot response."


def test_few_shot_with_no_train_df_uses_zero_examples_without_crashing():
    result = baseline_methods.run_baseline(_StubClient(), "few_shot", "Being gay is unnatural.", language="en",
                                            train_df=None, num_examples=3)
    assert result["counter_narrative"]


def test_unknown_method_raises():
    try:
        baseline_methods.run_baseline(_StubClient(), "full_pipeline", "x", language="en")
        assert False, "full_pipeline must not be handled by run_baseline (it's the agentic pipeline)."
    except ValueError:
        pass


def test_backend_failure_is_caught_and_returns_empty_output_with_error_recorded():
    result = baseline_methods.run_baseline(_ExplodingClient(), "zero_shot", "x", language="en")
    assert result["counter_narrative"] == ""
    assert result["metadata"]["errors"]


if __name__ == "__main__":
    test_all_four_baseline_methods_are_registered()
    test_zero_shot_returns_plain_text()
    test_cot_extracts_only_the_final_marker_line()
    test_few_shot_examples_drawn_from_train_only()
    test_few_shot_with_no_train_df_uses_zero_examples_without_crashing()
    test_unknown_method_raises()
    test_backend_failure_is_caught_and_returns_empty_output_with_error_recorded()
    print("test_baseline_methods.py: ALL PASSED")
