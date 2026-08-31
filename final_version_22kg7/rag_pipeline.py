"""
new_arch RAG pipeline - clean-room re-implementation: corpus build, FAISS
retrieval, reranking, and bilingual query merging, all in one file (the old
project split this across 6 files under src/retrieval/; consolidated here
per new_arch's "one module per file" tree). Does not import src/retrieval/*.

Corpus: two variants, exactly as before - "filtered" (default, restricted to
homophobia/transphobia/LGBT+-targeted rows) and "unfiltered" (every row,
ablation-only). TRAIN (or "all", for the 3 corpus-only datasets) rows only -
TEST and carved VALIDATION are NEVER indexed, to avoid ever surfacing a
held-out gold reference as "evidence" for that same example.

Similarity: FAISS IndexFlatIP over L2-normalized intfloat/multilingual-e5-small
embeddings (cosine similarity via normalized inner product).

4 RAG modes, ALL preserved: no_rag / fact_rag / cultural_rag / dual_rag.
no_rag is a hard early return - zero embedding calls, zero corpus reads, and
(per the Defender/evidence_verifier contract) zero retrieval queries are ever
generated for it - it must never pretend evidence was consulted.

Bilingual retrieval (default query_strategy="both"): each claim is looked up
using the input-language query text AND a semantically-equivalent English
query text (supplied by the caller, since generating that translation is an
LLM job - see evidence_verifier.py); results are merged, deduplicated, and
reranked together. The language of the retrieved SOURCE is always preserved
in the returned record and is never altered - only the query language varies.
"""
import numpy as np

import config
import dataset_loader
from utils import get_logger, write_jsonl, read_jsonl, ensure_dir

logger = get_logger("new_arch.rag_pipeline")

CORPUS_PATH = config.KNOWLEDGE_DIR / "corpus.jsonl"
CORPUS_PATH_UNFILTERED = config.KNOWLEDGE_DIR / "corpus_unfiltered.jsonl"

MIN_USEFUL_TEXT_LEN = 5
SAME_LANGUAGE_BONUS = 0.05
SAME_REGION_BONUS = 0.03   # soft additive bonus only - NEVER a hard filter
DIVERSITY_PENALTY_STEP = 0.02

# ---------------------------------------------------------------------------
# Corpus build
# ---------------------------------------------------------------------------
def _quality_filter(records: list) -> list:
    """Drop unusably short/duplicate/unsafe records - a coarse text-level
    filter, independent of embedding similarity scores.

    Deliberately does NOT call utils.contains_unsafe_keywords() - that
    function is the v22 evaluation-only unsafe-keyword redesign (language-
    gated, Optional[bool]-returning, word-boundary matching) and must never
    be allowed to change what gets indexed for retrieval. This inlines the
    original v20 corpus-build check verbatim (unconditional bare-substring
    match against the English keyword list, regardless of a record's actual
    language) so the RAG corpus - and therefore retrieval and generation -
    stays behaviorally identical to v20, even as the evaluation-side keyword
    logic keeps evolving in later stages. See utils.UNSAFE_KEYWORDS_EN."""
    from utils import UNSAFE_KEYWORDS_EN, normalize_whitespace

    seen_texts = set()
    kept = []
    for r in records:
        cn = normalize_whitespace(r.get("reference_counter_narrative") or "")
        kn = normalize_whitespace(r.get("knowledge_text") or "")
        useful_text = cn or kn
        if len(useful_text) < MIN_USEFUL_TEXT_LEN:
            continue
        key = useful_text.lower()
        if key in seen_texts:
            continue
        if any(kw in cn.lower() for kw in UNSAFE_KEYWORDS_EN):
            continue
        seen_texts.add(key)
        kept.append(r)
    return kept


def build_corpus(filter_target: bool = True) -> list:
    records = dataset_loader.iter_corpus_records(filter_target=filter_target)
    records = _quality_filter(records)
    for idx, r in enumerate(records):
        r["corpus_id"] = idx
    path = CORPUS_PATH if filter_target else CORPUS_PATH_UNFILTERED
    ensure_dir(config.KNOWLEDGE_DIR)
    write_jsonl(records, path)
    logger.info("Built RAG corpus (filter_target=%s): %d records -> %s", filter_target, len(records), path)
    return records


def load_corpus(filter_target: bool = True) -> list:
    path = CORPUS_PATH if filter_target else CORPUS_PATH_UNFILTERED
    if not path.exists():
        raise FileNotFoundError(f"RAG corpus not found at {path}. Build it first (see main.py build-index).")
    return read_jsonl(path)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


def get_embedding_model(model_name: str = None):
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def _prefix_for_e5(texts: list, is_query: bool) -> list:
    prefix = "query: " if is_query else "passage: "
    return [f"{prefix}{t}" for t in texts]


def embed_texts(texts: list, model_name: str = None, is_query: bool = False) -> np.ndarray:
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    model = get_embedding_model(model_name)
    prepped = _prefix_for_e5(texts, is_query) if "e5" in model_name.lower() else texts
    return model.encode(prepped, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# FAISS vector store
# ---------------------------------------------------------------------------
class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.dim = None

    def build(self, embeddings: np.ndarray, metadata: list):
        import faiss
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype("float32"))
        self.metadata = metadata
        return self

    def save(self, index_path, metadata_path):
        import faiss
        ensure_dir(index_path.parent)
        faiss.write_index(self.index, str(index_path))
        write_jsonl(self.metadata, metadata_path)

    def load(self, index_path, metadata_path):
        import faiss
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"FAISS index/metadata not found at {index_path} - build it first.")
        self.index = faiss.read_index(str(index_path))
        self.metadata = read_jsonl(metadata_path)
        self.dim = self.index.d
        return self

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        if self.index is None:
            raise RuntimeError("FaissVectorStore not built/loaded yet.")
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["_score"] = float(score)
            results.append(entry)
        return results


_STORES = {}  # {filter_target(bool): FaissVectorStore} - dual cache, both variants queryable per process


def build_index(filter_target: bool = True) -> FaissVectorStore:
    records = build_corpus(filter_target=filter_target)
    texts = [f"{r.get('hate_speech') or ''} {r.get('reference_counter_narrative') or ''}".strip() for r in records]
    embeddings = embed_texts(texts, is_query=False)
    store = FaissVectorStore().build(embeddings, records)
    index_path = config.FAISS_INDEX_PATH if filter_target else config.FAISS_INDEX_PATH_UNFILTERED
    metadata_path = config.FAISS_METADATA_PATH if filter_target else config.FAISS_METADATA_PATH_UNFILTERED
    store.save(index_path, metadata_path)
    _STORES[filter_target] = store
    logger.info("Built FAISS index (filter_target=%s): %d vectors", filter_target, len(records))
    return store


def _get_store(filter_target: bool = True) -> FaissVectorStore:
    if filter_target not in _STORES:
        index_path = config.FAISS_INDEX_PATH if filter_target else config.FAISS_INDEX_PATH_UNFILTERED
        metadata_path = config.FAISS_METADATA_PATH if filter_target else config.FAISS_METADATA_PATH_UNFILTERED
        _STORES[filter_target] = FaissVectorStore().load(index_path, metadata_path)
    return _STORES[filter_target]


# ---------------------------------------------------------------------------
# RAG-mode filtering + reranking
# ---------------------------------------------------------------------------
def _matches_rag_mode(record: dict, rag_mode: str) -> bool:
    has_knowledge = bool((record.get("knowledge_text") or "").strip())
    has_cn = bool((record.get("reference_counter_narrative") or "").strip())
    if rag_mode == "fact_rag":
        return has_knowledge
    if rag_mode == "cultural_rag":
        return (not has_knowledge) and has_cn
    return True  # dual_rag (and any unrecognized value) - no type restriction


def rerank(records: list, target_language: str = None, target_region: str = None, top_k: int = 5) -> list:
    """Additive soft bonuses (language match, region match) + a diversity
    penalty across repeated source datasets. Region match is NEVER a hard
    filter - a record failing both checks is still eligible."""
    scored = []
    for r in records:
        score = r.get("_score", 0.0)
        if target_language and r.get("language") == target_language:
            score += SAME_LANGUAGE_BONUS
        if target_region and r.get("region") == target_region:
            score += SAME_REGION_BONUS
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    seen_sources = {}
    final = []
    for score, r in scored:
        source = r.get("dataset_name")
        penalty = seen_sources.get(source, 0) * DIVERSITY_PENALTY_STEP
        final.append((score - penalty, r))
        seen_sources[source] = seen_sources.get(source, 0) + 1
        if len(final) >= top_k * 2:
            break
    final.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in final[:top_k]]


# ---------------------------------------------------------------------------
# Retrieval entry points
# ---------------------------------------------------------------------------
def retrieve(query_text: str, query_language: str = "input_language", language: str = None,
             region: str = None, rag_mode: str = "dual_rag", filter_target: bool = True,
             top_k: int = None) -> list:
    """Single-query retrieval. rag_mode="no_rag" must be checked by the
    CALLER before ever calling this (evidence_verifier.py) - but as a second
    line of defense this also returns [] immediately for "no_rag" without
    touching the embedding model or FAISS index."""
    if rag_mode == "no_rag":
        return []
    top_k = top_k or config.RAG_TOP_K
    store = _get_store(filter_target=filter_target)
    query_emb = embed_texts([query_text], is_query=True)[0]
    pool = store.search(query_emb, top_k=max(top_k * 5, top_k))
    pool = [r for r in pool if _matches_rag_mode(r, rag_mode)]
    results = rerank(pool, target_language=language, target_region=region, top_k=top_k)
    for r in results:
        r["_query_used"] = query_text
        r["_query_language"] = query_language
    return results


def retrieve_bilingual(input_query: str, english_query: str = None, query_strategy: str = None,
                        language: str = None, region: str = None, rag_mode: str = "dual_rag",
                        filter_target: bool = True, top_k: int = None) -> list:
    """The Defender's main retrieval entry point. query_strategy in
    {"input_language","english","both"} (default config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE
    = "both"). Runs both query variants (when requested and available),
    merges, deduplicates by passage text, and reranks by score (language/
    region bonus + diversity penalty already applied per-query before
    merging). Each returned record carries which query/language retrieved it
    (_query_used/_query_language) and its own preserved source `language` -
    the source's language is never translated or altered."""
    if rag_mode == "no_rag":
        return []
    query_strategy = query_strategy or config.DEFAULT_RETRIEVAL_QUERY_LANGUAGE
    top_k = top_k or config.RAG_TOP_K

    results = []
    if query_strategy in ("input_language", "both"):
        results += retrieve(input_query, "input_language", language, region, rag_mode, filter_target, top_k)
    if query_strategy in ("english", "both") and english_query:
        results += retrieve(english_query, "english", language, region, rag_mode, filter_target, top_k)

    merged = {}
    for r in results:
        key = (r.get("hate_speech") or "", r.get("reference_counter_narrative") or "", r.get("knowledge_text") or "")
        if key not in merged or r.get("_score", 0.0) > merged[key].get("_score", 0.0):
            merged[key] = r
    ordered = sorted(merged.values(), key=lambda r: r.get("_score", 0.0), reverse=True)
    return ordered[:top_k]


def format_evidence_block(records: list, max_items: int = 5) -> str:
    if not records:
        return "No relevant retrieved evidence available."
    lines = []
    for i, r in enumerate(records[:max_items], 1):
        hate = (r.get("hate_speech") or "")[:200]
        cn = (r.get("reference_counter_narrative") or "")[:300]
        kn = (r.get("knowledge_text") or "")[:300]
        lines.append(
            f"[{i}] (source={r.get('dataset_name')}, lang={r.get('language')}, "
            f"query_lang={r.get('_query_language')}) similar_comment=\"{hate}\" "
            f"counter_narrative=\"{cn}\" knowledge=\"{kn}\""
        )
    return "\n".join(lines)
