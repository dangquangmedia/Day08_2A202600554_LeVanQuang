from src import task7_reranking


def test_default_rerank_boosts_query_matching_candidate():
    candidates = [
        {"content": "Python programming tutorial", "score": 0.9, "metadata": {}},
        {"content": "Hình phạt tàng trữ trái phép chất ma túy", "score": 0.4, "metadata": {}},
    ]

    results = task7_reranking.rerank("hình phạt ma tuý", candidates, top_k=2)

    assert results[0]["content"].startswith("Hình phạt")
    assert "rerank_score" in results[0]
    assert results[0]["score"] == results[0]["rerank_score"]


def test_rerank_respects_top_k():
    candidates = [
        {"content": f"Document {i} ma túy", "score": 1.0 - i * 0.1, "metadata": {}}
        for i in range(6)
    ]

    results = task7_reranking.rerank("ma túy", candidates, top_k=3)

    assert len(results) == 3


def test_rerank_rrf_fuses_duplicate_candidates():
    semantic = [
        {"content": "A", "score": 0.9, "metadata": {"source": "a.md"}},
        {"content": "B", "score": 0.8, "metadata": {"source": "b.md"}},
    ]
    lexical = [
        {"content": "B", "score": 10.0, "metadata": {"source": "b.md"}},
        {"content": "A", "score": 5.0, "metadata": {"source": "a.md"}},
    ]

    results = task7_reranking.rerank_rrf([semantic, lexical], top_k=2, k=60)

    assert len(results) == 2
    assert all("score" in result for result in results)
    assert all("rrf_score" in result for result in results)


def test_rerank_mmr_returns_diverse_top_k():
    query_embedding = [1.0, 0.0]
    candidates = [
        {"content": "A", "score": 0.9, "embedding": [1.0, 0.0], "metadata": {}},
        {"content": "B", "score": 0.8, "embedding": [0.9, 0.1], "metadata": {}},
        {"content": "C", "score": 0.7, "embedding": [0.0, 1.0], "metadata": {}},
    ]

    results = task7_reranking.rerank_mmr(
        query_embedding,
        candidates,
        top_k=2,
        lambda_param=0.5,
    )

    assert len(results) == 2
    assert results[0]["content"] == "A"
    assert "mmr_score" in results[0]
