"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import math
import re
import unicodedata

try:
    from src.task4_chunking_indexing import create_local_embedding
except ModuleNotFoundError:
    from task4_chunking_indexing import create_local_embedding


def normalize_text(text: str) -> str:
    """Lowercase và bỏ dấu để query 'ma tuý' khớp document 'ma túy'."""
    text = text.lower()
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, đồng bộ tinh thần với Task 6 lexical search."""
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def cosine_sim(vector_a: list[float], vector_b: list[float]) -> float:
    """Cosine similarity an toàn với zero vector hoặc vector lệch chiều."""
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_overlap(query: str, content: str) -> float:
    """Tỷ lệ query tokens xuất hiện trong content, dùng làm relevance local."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    content_terms = set(tokenize(content))
    return len(query_terms.intersection(content_terms)) / len(query_terms)


def normalized_base_score(candidate: dict, max_abs_score: float) -> float:
    """Đưa retrieval score về khoảng ổn định để trộn với overlap."""
    if max_abs_score <= 0:
        return 0.0
    return float(candidate.get("score", 0.0)) / max_abs_score


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if top_k <= 0 or not query.strip() or not candidates:
        return []

    max_abs_score = max(
        abs(float(candidate.get("score", 0.0))) for candidate in candidates
    ) or 1.0

    reranked = []
    for candidate in candidates:
        item = candidate.copy()
        content = item.get("content", "")
        overlap_score = lexical_overlap(query, content)
        base_score = normalized_base_score(item, max_abs_score)
        rerank_score = 0.75 * overlap_score + 0.25 * base_score
        item["original_score"] = float(candidate.get("score", 0.0))
        item["rerank_score"] = rerank_score
        item["score"] = rerank_score
        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []

    lambda_param = min(1.0, max(0.0, lambda_param))
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    candidate_embeddings = [
        candidate.get("embedding") or create_local_embedding(candidate.get("content", ""))
        for candidate in candidates
    ]
    mmr_scores: dict[int, float] = {}

    for _ in range(min(top_k, len(candidates))):
        best_idx = remaining[0]
        best_score = float("-inf")

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidate_embeddings[idx])
            max_sim_to_selected = 0.0
            for selected_idx in selected:
                sim = cosine_sim(
                    candidate_embeddings[idx],
                    candidate_embeddings[selected_idx],
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        mmr_scores[best_idx] = best_score

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["mmr_score"] = mmr_scores[idx]
        item["score"] = mmr_scores[idx]
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0 or not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map.setdefault(key, item)

    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["original_score"] = item.get("score", 0.0)
        item["rrf_score"] = score
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        query_embedding = create_local_embedding(query)
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
