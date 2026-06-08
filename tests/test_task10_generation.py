from src import task10_generation


def test_prohibited_procurement_query_returns_refusal(monkeypatch):
    monkeypatch.setattr(task10_generation, "OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(
        task10_generation,
        "retrieve",
        lambda query, top_k: [
            {
                "content": "Luật cấm mua bán, sử dụng trái phép chất ma túy.",
                "score": 0.9,
                "source": "hybrid",
                "metadata": {"source": "luat-phong-chong-ma-tuy-2021.md", "type": "legal"},
            }
        ],
    )

    result = task10_generation.generate_with_citation("mua ma túy ở đâu rẻ")

    assert result["generation_mode"] == "safety_refusal"
    assert "không thể hỗ trợ" in result["answer"].lower()
    assert "mua" in result["answer"].lower()
    assert result["sources"][0]["metadata"]["source"] == "luat-phong-chong-ma-tuy-2021.md"


def test_reorder_for_llm_keeps_best_at_edges():
    chunks = [
        {"content": f"Chunk {i}", "score": 1.0 - i * 0.1}
        for i in range(5)
    ]

    reordered = task10_generation.reorder_for_llm(chunks)

    assert len(reordered) == 5
    assert reordered[0]["content"] == "Chunk 0"
    assert reordered[-1]["content"] == "Chunk 1"


def test_format_context_includes_source_type_score_and_content():
    chunks = [
        {
            "content": "Nội dung pháp luật",
            "score": 0.9,
            "source": "hybrid",
            "metadata": {
                "source": "luat-phong-chong-ma-tuy-2021.md",
                "type": "legal",
            },
        }
    ]

    context = task10_generation.format_context(chunks)

    assert "luat-phong-chong-ma-tuy-2021.md" in context
    assert "legal" in context
    assert "Nội dung pháp luật" in context


def test_generate_with_citation_uses_local_fallback_when_no_api_key(monkeypatch):
    monkeypatch.setattr(task10_generation, "OPENAI_API_KEY", "")
    monkeypatch.setattr(
        task10_generation,
        "retrieve",
        lambda query, top_k: [
            {
                "content": "Tàng trữ trái phép chất ma túy có thể bị phạt tù.",
                "score": 0.9,
                "source": "hybrid",
                "metadata": {"source": "bo-luat-hinh-su-2015-sua-doi-2017.md", "type": "legal"},
            }
        ],
    )

    result = task10_generation.generate_with_citation("Hình phạt tàng trữ ma túy?")

    assert "answer" in result
    assert "bo-luat-hinh-su-2015-sua-doi-2017.md" in result["answer"]
    assert result["retrieval_source"] == "hybrid"
    assert len(result["sources"]) == 1
