"""
Optional web search - a supplementary evidence source alongside local RAG,
via the Tavily Search API. Entirely gated by config.ENABLE_WEB_SEARCH
(default False, env-driven) - when disabled, this module's search
functions are never called and no network request beyond local retrieval
ever happens. When enabled, TAVILY_API_KEY must be set (never hardcoded,
never guessed) - missing key raises a clear error the first time a search
is actually attempted, not at import time.

When enabled, web search runs ALONGSIDE local RAG (not only as a fallback
when local RAG is empty) - see evidence_verifier.py. Every result carries
its source URL explicitly, since the Defender/Judge are instructed (see
prompts.build_defender_prompt) to only treat web-sourced claims as reliable
when the source is identifiable and looks credible - never blended in
anonymously with local corpus evidence.

Skipped entirely in "no_rag" mode, same as local retrieval - "no_rag" means
no retrieval of any kind, local or web.

Results are restricted to config.WEB_SEARCH_TRUSTED_DOMAINS by default
(config.WEB_SEARCH_RESTRICT_DOMAINS) - a curated allowlist of domains
considered reputable for this project's specific claim domain (homophobia/
transphobia-related medical, psychological, legal, and human-rights facts),
applied via Tavily's include_domains at the API level. This is a technical
guardrail on top of (not instead of) the Defender's own prompt-level source
judgement - see config.py for the domain list and rationale.
"""
import config
from utils import get_logger

logger = get_logger("new_arch.web_search")


class WebSearchError(RuntimeError):
    """Raised for a missing key/package - a configuration problem, not a
    transient network failure. Callers should let this propagate (it means
    ENABLE_WEB_SEARCH is on but not usable) rather than silently swallowing it,
    but should catch broad Exceptions separately for real API/network
    failures and degrade to local-only evidence for that claim."""


def _get_client():
    if not config.TAVILY_API_KEY:
        raise WebSearchError(
            "ENABLE_WEB_SEARCH is true but TAVILY_API_KEY is not set. Get a key at "
            "https://tavily.com and set it as an environment variable. This project "
            "never inserts or guesses API keys for you."
        )
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise WebSearchError(
            "ENABLE_WEB_SEARCH is true but the `tavily-python` package is not installed. "
            "Run: pip install tavily-python"
        ) from exc
    return TavilyClient(api_key=config.TAVILY_API_KEY)


def search(query: str, max_results: int = None) -> list:
    """Returns a list of dicts: {title, url, passage, score}. Raises
    WebSearchError for a missing key/package. Raises the underlying
    exception for actual API failures (network, rate limit, etc) - callers
    should catch broadly and degrade to local-only evidence.

    When config.WEB_SEARCH_RESTRICT_DOMAINS is True (default), results are
    restricted to config.WEB_SEARCH_TRUSTED_DOMAINS via Tavily's
    include_domains - see config.py for the rationale. This means a query
    that only has an answer outside that domain set returns fewer/no
    results rather than a broader but less vetted set; that's an accepted
    precision-over-recall tradeoff for this project's sensitive claim
    domain, not an oversight."""
    if not query or not query.strip():
        return []
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    client = _get_client()
    search_kwargs = {"query": query, "max_results": max_results, "search_depth": "basic"}
    if config.WEB_SEARCH_RESTRICT_DOMAINS and config.WEB_SEARCH_TRUSTED_DOMAINS:
        search_kwargs["include_domains"] = config.WEB_SEARCH_TRUSTED_DOMAINS
    response = client.search(**search_kwargs)
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "passage": (r.get("content") or "")[:500],
            "score": r.get("score", 0.0),
        })
    return results


def search_for_claim(input_query: str, english_query: str, query_strategy: str, max_results: int = None) -> list:
    """Bilingual-aware, mirroring rag_pipeline.retrieve_bilingual: runs
    whichever query variant(s) query_strategy calls for ("input_language" /
    "english" / "both"), tags each result with which query/language found
    it, and dedupes by URL. Returns raw dicts for evidence_verifier to wrap
    into EvidenceItem objects - never raises for an individual query
    variant coming back empty, only for a genuine configuration problem
    (see _get_client)."""
    query_strategy = query_strategy or config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE
    combined = []
    if query_strategy in ("input_language", "both") and input_query:
        for r in search(input_query, max_results=max_results):
            r["_query_used"], r["_query_language"] = input_query, "input_language"
            combined.append(r)
    if query_strategy in ("english", "both") and english_query:
        for r in search(english_query, max_results=max_results):
            r["_query_used"], r["_query_language"] = english_query, "english"
            combined.append(r)

    seen_urls = set()
    deduped = []
    for r in combined:
        if r["url"] and r["url"] in seen_urls:
            continue
        if r["url"]:
            seen_urls.add(r["url"])
        deduped.append(r)
    return deduped
