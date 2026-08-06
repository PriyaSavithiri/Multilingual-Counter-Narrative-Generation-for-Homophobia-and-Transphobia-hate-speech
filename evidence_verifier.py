"""
Evidence verification helper - called from INSIDE the Defender's turn
(defender_agent.py), each round. There is no standalone post-hoc
verification stage anywhere in new_arch (the old project's separate
Tool-MAD/checker_manager pipeline stage is deliberately not reproduced -
this logic is redistributed here and invoked per-round instead).

Bilingual retrieval (config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE = "both" by
default): for each claim, an input-language query and a semantically
equivalent English query are both formulated (one LLM call), then both are
retrieved, merged, deduplicated, and reranked by rag_pipeline.retrieve_bilingual.
In "no_rag" mode, NO queries are generated at all - not even to skip them
later, the query-formulation call itself is never made.

Optional web search (config.ENABLE_WEB_SEARCH, off by default): when
enabled, every claim's evidence gathering runs BOTH local RAG AND a live
web search (via web_search.py, Tavily) - not a fallback used only when
local RAG is empty, always alongside it. Web results are never blended in
anonymously: every item's origin ("local_rag" | "web_search") and, for web
results, the source URL, are carried on the EvidenceItem and rendered
explicitly in the evidence block so the Defender/Judge can judge
reliability rather than trusting it by default. A web search failure
(missing key, network error) degrades gracefully to local-only evidence
for that claim rather than failing the whole Defender turn.
"""
import config
import prompts
import rag_pipeline
import web_search
from schemas import EvidenceItem
from utils import get_logger

logger = get_logger("new_arch.evidence_verifier")


def formulate_bilingual_queries(client, claim: str, language: str):
    prompt = prompts.build_claim_query_prompt(claim, language)
    parsed = client.generate(prompt, response_schema=["input_language_query", "english_query"],
                              temperature=0.2, max_tokens=200)
    if not parsed:
        return claim, None
    return parsed.get("input_language_query") or claim, parsed.get("english_query")


def _local_rag_items(input_q: str, english_q: str, language: str, region: str, rag_mode: str,
                      filter_target: bool, query_strategy: str) -> list:
    records = rag_pipeline.retrieve_bilingual(
        input_query=input_q, english_query=english_q, query_strategy=query_strategy,
        language=language, region=region, rag_mode=rag_mode, filter_target=filter_target,
    )
    return [
        EvidenceItem(
            source_id=str(r.get("corpus_id")),
            title=r.get("dataset_name") or "",
            passage=(r.get("reference_counter_narrative") or r.get("knowledge_text") or ""),
            retrieval_score=r.get("_score", 0.0),
            source_type=("fact" if (r.get("knowledge_text") or "").strip() else "cultural"),
            language=r.get("language") or "",
            query_used=r.get("_query_used") or "",
            query_language=r.get("_query_language") or "input_language",
            origin="local_rag",
        )
        for r in records
    ]


def _web_search_items(input_q: str, english_q: str, query_strategy: str) -> list:
    try:
        results = web_search.search_for_claim(input_q, english_q, query_strategy)
    except web_search.WebSearchError:
        raise  # configuration problem (missing key/package) - let the caller see it clearly
    except Exception as exc:
        logger.warning("Web search failed, continuing with local RAG evidence only: %s", exc)
        return []
    return [
        EvidenceItem(
            source_id=r.get("url") or "", title=r.get("title") or "", passage=r.get("passage") or "",
            retrieval_score=r.get("score", 0.0), source_type="web", language="",
            query_used=r.get("_query_used") or "", query_language=r.get("_query_language") or "input_language",
            origin="web_search", url=r.get("url") or "",
        )
        for r in results
    ]


def format_combined_evidence_block(items: list) -> str:
    if not items:
        return "No relevant retrieved evidence available."
    lines = []
    for i, it in enumerate(items[:8], 1):
        if it.origin == "web_search":
            lines.append(
                f'[{i}] (origin=web_search, url={it.url or "unknown"}, query_lang={it.query_language}) '
                f'title="{it.title}" content="{(it.passage or "")[:300]}"'
            )
        else:
            lines.append(
                f'[{i}] (origin=local_rag, source={it.title}, lang={it.language}, query_lang={it.query_language}) '
                f'passage="{(it.passage or "")[:300]}"'
            )
    return "\n".join(lines)


def gather_evidence_for_claim(client, claim: str, language: str, region: str, rag_mode: str,
                               filter_target: bool, query_strategy: str = None):
    """Returns (evidence_items: list[EvidenceItem], evidence_block: str).
    "no_rag" skips ALL retrieval, local and web - no queries formulated at all."""
    if rag_mode == "no_rag":
        return [], "No relevant retrieved evidence available. (RAG mode set to 'no_rag'.)"

    query_strategy = query_strategy or config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE
    input_q, english_q = formulate_bilingual_queries(client, claim, language)

    items = _local_rag_items(input_q, english_q, language, region, rag_mode, filter_target, query_strategy)
    if config.ENABLE_WEB_SEARCH:
        items = items + _web_search_items(input_q, english_q, query_strategy)

    return items, format_combined_evidence_block(items)
