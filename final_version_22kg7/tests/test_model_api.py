import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_api import ModelClient

_SCHEMA = ["answer"]


class _NoChatModelClient(ModelClient):
    """_build_chat_model() is left unimplemented (raises NotImplementedError,
    same as the base class) - simulates exactly what broke
    test_fast_track_explicit_hate_smoke: a client that only implements
    generate() directly (as every test stub in this suite does) and never
    overrides _build_chat_model() at all, since it doesn't go through
    LangChain's plumbing to produce a result."""
    backend_name = "no-chat-model"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        # messages is a plain string for every prompt in these tests.
        return {"answer": f"reply to: {messages}"}


class _BatchFailsClient(ModelClient):
    """_build_chat_model() works, but llm.batch() itself would fail - covers
    the OTHER failure point inside generate_batch(), not just model
    construction."""
    backend_name = "batch-fails"

    def _build_chat_model(self, temperature, max_tokens):
        class _ExplodingLLM:
            def batch(self, lc_batches):
                raise RuntimeError("simulated batch failure")
        return _ExplodingLLM()

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        return {"answer": f"reply to: {messages}"}


def test_generate_batch_falls_back_when_build_chat_model_fails():
    client = _NoChatModelClient()
    prompts = ["prompt one", "prompt two", "prompt three"]
    results = client.generate_batch(prompts, response_schema=_SCHEMA)
    assert results == [{"answer": "reply to: prompt one"}, {"answer": "reply to: prompt two"},
                        {"answer": "reply to: prompt three"}]


def test_generate_batch_falls_back_when_llm_batch_fails():
    client = _BatchFailsClient()
    prompts = ["prompt A", "prompt B"]
    results = client.generate_batch(prompts, response_schema=_SCHEMA)
    assert results == [{"answer": "reply to: prompt A"}, {"answer": "reply to: prompt B"}]


def test_generate_batch_preserves_input_order_on_fallback():
    client = _NoChatModelClient()
    prompts = [f"prompt {i}" for i in range(5)]
    results = client.generate_batch(prompts, response_schema=_SCHEMA)
    assert [r["answer"] for r in results] == [f"reply to: prompt {i}" for i in range(5)]


def test_generate_batch_empty_input_returns_empty_list():
    assert _NoChatModelClient().generate_batch([], response_schema=_SCHEMA) == []


if __name__ == "__main__":
    test_generate_batch_falls_back_when_build_chat_model_fails()
    test_generate_batch_falls_back_when_llm_batch_fails()
    test_generate_batch_preserves_input_order_on_fallback()
    test_generate_batch_empty_input_returns_empty_list()
    print("test_model_api.py: ALL PASSED")
