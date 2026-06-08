"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import re
import unicodedata

try:
    from src.task4_chunking_indexing import INDEX_PATH
except ModuleNotFoundError:
    from task4_chunking_indexing import INDEX_PATH


CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def normalize_text(text: str) -> str:
    """Lowercase và bỏ dấu để 'ma túy' và 'ma tuý' cùng match."""
    text = text.lower()
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, phù hợp dữ liệu chunk tiếng Việt đã normalize."""
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def load_corpus() -> list[dict]:
    """Đọc corpus chunks từ index JSONL đã tạo ở Task 4."""
    if not INDEX_PATH.exists():
        return []

    corpus = []
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            metadata = dict(chunk.get("metadata", {}))
            if "doc_type" not in metadata and "type" in metadata:
                metadata["doc_type"] = metadata["type"]
            corpus.append({
                "content": chunk.get("content", ""),
                "metadata": metadata,
            })
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_overlap_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Fallback nhỏ khi BM25 có IDF bằng 0 trên corpus rất nhỏ."""
    query_terms = set(query_tokens)
    if not query_terms:
        return 0.0
    overlap = query_terms.intersection(doc_tokens)
    return len(overlap) / len(query_terms)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus = CORPUS or load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    scores = [
        max(float(score), 0.0) + lexical_overlap_score(query_tokens, doc_tokens)
        for score, doc_tokens in zip(bm25_scores, tokenized_corpus)
    ]

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append({
            "content": corpus[index]["content"],
            "score": score,
            "metadata": corpus[index]["metadata"],
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
