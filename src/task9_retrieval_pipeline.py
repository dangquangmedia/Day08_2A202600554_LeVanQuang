"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

try:
    from .task5_semantic_search import load_index, semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import load_index, semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search

import re
import unicodedata


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def normalize_text(text: str) -> str:
    """Chuẩn hóa text để match điều luật ổn định hơn."""
    normalized = text.lower().replace("đ", "d").replace("Đ", "d")
    decomposed = unicodedata.normalize("NFD", normalized)
    no_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", no_accents)).strip()


def legal_article_targets(query: str) -> list[str]:
    """Map câu hỏi pháp lý phổ biến tới điều luật về ma túy trong BLHS."""
    normalized = normalize_text(query)
    if "ma tuy" not in normalized:
        return []

    targets = []
    if "249" in normalized or "tang tru" in normalized:
        targets.extend(["dieu 249", "toi tang tru trai phep chat ma tuy"])
    if "250" in normalized or "van chuyen" in normalized:
        targets.extend(["dieu 250", "toi van chuyen trai phep chat ma tuy"])
    if "251" in normalized or "mua ban" in normalized:
        targets.extend(["dieu 251", "toi mua ban trai phep chat ma tuy"])
    if "252" in normalized or "chiem doat" in normalized:
        targets.extend(["dieu 252", "toi chiem doat chat ma tuy"])
    return targets


def legal_article_boost(query: str, top_k: int) -> list[dict]:
    """
    Ưu tiên chunk chứa đúng điều luật khi query hỏi tội danh cụ thể.

    Dense/BM25 trên PDF OCR đôi khi kéo nhầm đoạn chú thích hoặc đoạn lỗi font.
    Lớp boost này chỉ kích hoạt khi query có intent pháp lý rõ ràng.
    """
    targets = legal_article_targets(query)
    if not targets:
        return []

    boosted = []
    for chunk in load_index():
        content = chunk.get("content", "")
        normalized = normalize_text(content)
        match_count = sum(1 for target in targets if target in normalized)
        if match_count == 0:
            continue

        metadata = dict(chunk.get("metadata", {}))
        if "doc_type" not in metadata and "type" in metadata:
            metadata["doc_type"] = metadata["type"]
        boosted.append({
            "content": content,
            "score": 1.0 + match_count,
            "metadata": metadata,
            "source": "hybrid",
        })

    boosted.sort(key=lambda item: item["score"], reverse=True)
    return boosted[:top_k]


def merge_priority_results(priority: list[dict], regular: list[dict], top_k: int) -> list[dict]:
    """Merge priority legal hits before regular retrieval, deduplicating content."""
    merged = []
    seen = set()
    for result in priority + regular:
        key = result.get("content", "")[:300]
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= top_k:
            break
    return merged


def tag_retrieval_results(results: list[dict], retrieval_source: str) -> list[dict]:
    """Gắn metadata nguồn retrieval trước khi fusion."""
    tagged = []
    for result in results:
        item = result.copy()
        metadata = dict(item.get("metadata", {}))
        sources = set(metadata.get("retrieval_sources", []))
        sources.add(retrieval_source)
        metadata["retrieval_sources"] = sorted(sources)
        item["metadata"] = metadata
        tagged.append(item)
    return tagged


def mark_hybrid_results(results: list[dict]) -> list[dict]:
    """Chuẩn hóa output hybrid để Task 9/10 dùng cùng schema."""
    marked = []
    for result in results:
        item = result.copy()
        metadata = dict(item.get("metadata", {}))
        metadata.setdefault("retrieval_sources", ["hybrid"])
        item["metadata"] = metadata
        item["source"] = "hybrid"
        marked.append(item)
    return marked


def fallback_to_pageindex(query: str, top_k: int) -> list[dict]:
    """PageIndex vectorless fallback, luôn giữ schema source='pageindex'."""
    fallback = pageindex_search(query, top_k=top_k)
    normalized = []
    for result in fallback[:top_k]:
        item = result.copy()
        item["source"] = "pageindex"
        item["metadata"] = dict(item.get("metadata", {}))
        normalized.append(item)
    return normalized


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if top_k <= 0 or not query.strip():
        return []

    retrieval_k = max(top_k * 2, top_k)
    dense_results = tag_retrieval_results(
        semantic_search(query, top_k=retrieval_k),
        "semantic",
    )
    sparse_results = tag_retrieval_results(
        lexical_search(query, top_k=retrieval_k),
        "lexical",
    )

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    merged = mark_hybrid_results(merged)

    if use_reranking and merged:
        final_results = rerank(
            query,
            merged,
            top_k=top_k,
            method=RERANK_METHOD,
        )
        final_results = mark_hybrid_results(final_results)
    else:
        final_results = merged[:top_k]

    priority_results = legal_article_boost(query, top_k)
    if priority_results:
        final_results = merge_priority_results(priority_results, final_results, top_k)

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({best_score:.3f}) < threshold "
            f"({score_threshold}). Fallback → PageIndex"
        )
        return fallback_to_pageindex(query, top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
