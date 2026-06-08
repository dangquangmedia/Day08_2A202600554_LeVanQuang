"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
import math

try:
    from src.task4_chunking_indexing import INDEX_PATH, create_local_embedding
except ModuleNotFoundError:
    from task4_chunking_indexing import INDEX_PATH, create_local_embedding


def load_index() -> list[dict]:
    """Đọc index JSONL đã tạo ở Task 4."""
    if not INDEX_PATH.exists():
        return []

    chunks = []
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Tính cosine similarity, an toàn với vector rỗng hoặc zero vector."""
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    query_embedding = create_local_embedding(query)
    results = []

    for chunk in load_index():
        content = chunk.get("content", "")
        embedding = chunk.get("embedding", [])
        metadata = dict(chunk.get("metadata", {}))
        if "doc_type" not in metadata and "type" in metadata:
            metadata["doc_type"] = metadata["type"]

        score = cosine_similarity(query_embedding, embedding)
        results.append({
            "content": content,
            "score": score,
            "metadata": metadata,
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
