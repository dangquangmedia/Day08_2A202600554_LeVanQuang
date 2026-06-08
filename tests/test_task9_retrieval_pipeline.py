from src import task9_retrieval_pipeline


def test_retrieve_merges_reranks_and_marks_hybrid(monkeypatch):
    monkeypatch.setattr(
        task9_retrieval_pipeline,
        "semantic_search",
        lambda query, top_k: [
            {"content": "semantic ma túy", "score": 0.8, "metadata": {"source": "s.md"}}
        ],
    )
    monkeypatch.setattr(
        task9_retrieval_pipeline,
        "lexical_search",
        lambda query, top_k: [
            {"content": "lexical hình phạt ma túy", "score": 2.0, "metadata": {"source": "l.md"}}
        ],
    )
    monkeypatch.setattr(
        task9_retrieval_pipeline,
        "pageindex_search",
        lambda query, top_k: [],
    )

    results = task9_retrieval_pipeline.retrieve("hình phạt ma túy", top_k=2)

    assert results
    assert len(results) <= 2
    assert all(result["source"] == "hybrid" for result in results)
    assert all("retrieval_sources" in result["metadata"] for result in results)


def test_retrieve_falls_back_to_pageindex_when_hybrid_score_low(monkeypatch):
    monkeypatch.setattr(task9_retrieval_pipeline, "semantic_search", lambda query, top_k: [])
    monkeypatch.setattr(task9_retrieval_pipeline, "lexical_search", lambda query, top_k: [])
    monkeypatch.setattr(
        task9_retrieval_pipeline,
        "pageindex_search",
        lambda query, top_k: [
            {
                "content": "fallback result",
                "score": 0.5,
                "metadata": {},
                "source": "pageindex",
            }
        ],
    )

    results = task9_retrieval_pipeline.retrieve(
        "obscure",
        top_k=3,
        score_threshold=0.99,
    )

    assert results[0]["source"] == "pageindex"


def test_retrieve_respects_top_k_without_reranking(monkeypatch):
    monkeypatch.setattr(
        task9_retrieval_pipeline,
        "semantic_search",
        lambda query, top_k: [
            {"content": f"semantic {i}", "score": 1.0, "metadata": {}}
            for i in range(5)
        ],
    )
    monkeypatch.setattr(task9_retrieval_pipeline, "lexical_search", lambda query, top_k: [])
    monkeypatch.setattr(task9_retrieval_pipeline, "pageindex_search", lambda query, top_k: [])

    results = task9_retrieval_pipeline.retrieve(
        "ma túy",
        top_k=2,
        use_reranking=False,
        score_threshold=0.0,
    )

    assert len(results) == 2


def test_legal_article_boost_prioritizes_specific_drug_crime_article(monkeypatch):
    monkeypatch.setattr(
        task9_retrieval_pipeline,
        "load_index",
        lambda: [
            {
                "content": "Điều 249. Tội tàng trữ trái phép chất ma túy. Người nào tàng trữ thì bị phạt tù.",
                "metadata": {"source": "bo-luat-hinh-su-2015-sua-doi-2017.md", "type": "legal"},
            },
            {
                "content": "Đoạn OCR lỗi có nhiều chữ hình phạt nhưng không phải điều luật.",
                "metadata": {"source": "bad.md", "type": "legal"},
            },
        ],
    )

    boosted = task9_retrieval_pipeline.legal_article_boost(
        "Tội tàng trữ ma túy bị xử phạt như thế nào?",
        top_k=2,
    )

    assert boosted
    assert "Điều 249" in boosted[0]["content"]
    assert boosted[0]["source"] == "hybrid"
