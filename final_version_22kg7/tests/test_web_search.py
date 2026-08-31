"""
Covers: web search is fully opt-in (config.ENABLE_WEB_SEARCH, off by
default), missing TAVILY_API_KEY raises a clear configuration error rather
than silently guessing/skipping, evidence_verifier merges local+web
evidence with explicit origin/url labeling when enabled, no_rag mode skips
web search entirely (same as local retrieval), and a web search failure
degrades gracefully to local-only evidence instead of crashing the round.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import evidence_verifier
import rag_pipeline
import web_search
from model_api import ModelClient
from schemas import EvidenceItem

_FAKE_RAG_RECORD = {
    "corpus_id": 1, "dataset_name": "mock-corpus", "language": "en", "_score": 0.9,
    "knowledge_text": "Mock factual evidence for testing.", "reference_counter_narrative": "",
    "_query_used": "mock query", "_query_language": "input_language",
}


def _fake_retrieve_bilingual(**kwargs):
    return [dict(_FAKE_RAG_RECORD)]


def test_web_search_disabled_by_default():
    assert config.ENABLE_WEB_SEARCH is False


def test_missing_api_key_raises_clear_configuration_error():
    original = config.TAVILY_API_KEY
    config.TAVILY_API_KEY = ""
    try:
        try:
            web_search.search("some query")
            assert False, "search() should have raised WebSearchError with no TAVILY_API_KEY set"
        except web_search.WebSearchError as exc:
            assert "TAVILY_API_KEY" in str(exc)
    finally:
        config.TAVILY_API_KEY = original


def test_empty_query_returns_empty_list_without_calling_the_api():
    # Empty query short-circuits before _get_client() is ever reached, so
    # this must not raise even with no key configured.
    assert web_search.search("") == []
    assert web_search.search("   ") == []


class _StubQueryClient(ModelClient):
    backend_name = "stub"

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        if response_schema is None:
            return "n/a"
        return {"input_language_query": "claim query", "english_query": "claim query en"}


def test_no_rag_mode_skips_web_search_entirely_even_when_enabled():
    original = config.ENABLE_WEB_SEARCH
    config.ENABLE_WEB_SEARCH = True
    try:
        items, block = evidence_verifier.gather_evidence_for_claim(
            _StubQueryClient(), "some claim", "en", None, "no_rag", True, "both"
        )
        assert items == []
        assert "no_rag" in block
    finally:
        config.ENABLE_WEB_SEARCH = original


def test_web_search_failure_degrades_to_local_only_not_a_crash(monkeypatch):
    # Local RAG retrieval is mocked here too, not just web search - this test
    # used to require a real local FAISS index + a working sentence-transformers
    # install (which pulls in transformers.PreTrainedModel), breaking on any
    # environment where that import chain is broken for reasons unrelated to
    # this project's own code (same issue fixed in test_pipeline.py's smoke
    # test). Mocking it removes that dependency without losing what this test
    # actually checks: that a web-search failure degrades gracefully rather
    # than crashing the round, and that local evidence still comes through.
    monkeypatch.setattr(rag_pipeline, "retrieve_bilingual", _fake_retrieve_bilingual)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(web_search, "search_for_claim", _boom)
    original_enabled = config.ENABLE_WEB_SEARCH
    config.ENABLE_WEB_SEARCH = True
    try:
        items, block = evidence_verifier.gather_evidence_for_claim(
            _StubQueryClient(), "being gay is unnatural", "en", None, "dual_rag", True, "both"
        )
        assert isinstance(items, list)  # did not crash
        assert len(items) > 0, "mocked local RAG evidence should still be returned despite the web search failure"
        assert all(it.origin == "local_rag" for it in items)  # no web items made it through
    finally:
        config.ENABLE_WEB_SEARCH = original_enabled


class _RecordingTavilyClient:
    """Stands in for tavily.TavilyClient - records the kwargs search() was
    called with instead of hitting the real API, so we can verify
    include_domains is (or isn't) threaded through correctly."""

    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": []}


def test_search_restricts_to_trusted_domains_by_default():
    original_get_client = web_search._get_client
    original_restrict = config.WEB_SEARCH_RESTRICT_DOMAINS
    fake_client = _RecordingTavilyClient()
    web_search._get_client = lambda: fake_client
    config.WEB_SEARCH_RESTRICT_DOMAINS = True
    try:
        web_search.search("is homosexuality a mental illness")
        assert len(fake_client.calls) == 1
        assert fake_client.calls[0].get("include_domains") == config.WEB_SEARCH_TRUSTED_DOMAINS
    finally:
        web_search._get_client = original_get_client
        config.WEB_SEARCH_RESTRICT_DOMAINS = original_restrict


def test_search_omits_domain_restriction_when_disabled():
    original_get_client = web_search._get_client
    original_restrict = config.WEB_SEARCH_RESTRICT_DOMAINS
    fake_client = _RecordingTavilyClient()
    web_search._get_client = lambda: fake_client
    config.WEB_SEARCH_RESTRICT_DOMAINS = False
    try:
        web_search.search("is homosexuality a mental illness")
        assert len(fake_client.calls) == 1
        assert "include_domains" not in fake_client.calls[0]
    finally:
        web_search._get_client = original_get_client
        config.WEB_SEARCH_RESTRICT_DOMAINS = original_restrict


def test_evidence_item_has_origin_and_url_fields_defaulting_to_local_rag():
    item = EvidenceItem(source_id="1", title="t", passage="p", retrieval_score=0.5, source_type="fact")
    assert item.origin == "local_rag"
    assert item.url == ""


def test_format_combined_evidence_block_labels_web_items_with_url():
    web_item = EvidenceItem(source_id="https://example.org/a", title="Example", passage="some content",
                             retrieval_score=0.9, source_type="web", origin="web_search",
                             url="https://example.org/a")
    block = evidence_verifier.format_combined_evidence_block([web_item])
    assert "web_search" in block and "https://example.org/a" in block


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_web_search_disabled_by_default()
    test_missing_api_key_raises_clear_configuration_error()
    test_empty_query_returns_empty_list_without_calling_the_api()
    test_no_rag_mode_skips_web_search_entirely_even_when_enabled()
    test_web_search_failure_degrades_to_local_only_not_a_crash(_FakeMonkeypatch())
    test_search_restricts_to_trusted_domains_by_default()
    test_search_omits_domain_restriction_when_disabled()
    test_evidence_item_has_origin_and_url_fields_defaulting_to_local_rag()
    test_format_combined_evidence_block_labels_web_items_with_url()
    print("test_web_search.py: ALL PASSED")
