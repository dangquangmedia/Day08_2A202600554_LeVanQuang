import json

from src import task6_lexical_search


def write_index(path, chunks):
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def test_lexical_search_reads_task4_index_and_sorts_results(monkeypatch, tmp_path):
    index_path = tmp_path / "chunks.jsonl"
    write_index(
        index_path,
        [
            {
                "content": "Tội tàng trữ trái phép chất ma túy bị xử lý theo pháp luật.",
                "metadata": {"source": "law.md", "type": "legal", "chunk_index": 0},
            },
            {
                "content": "Một bài báo về hoạt động biểu diễn nghệ thuật.",
                "metadata": {"source": "news.md", "type": "news", "chunk_index": 0},
            },
        ],
    )
    monkeypatch.setattr(task6_lexical_search, "INDEX_PATH", index_path)

    results = task6_lexical_search.lexical_search("tàng trữ ma tuý", top_k=2)

    assert results
    assert results[0]["score"] >= results[-1]["score"]
    assert "tàng trữ" in results[0]["content"]
    assert results[0]["metadata"]["doc_type"] == "legal"


def test_lexical_search_respects_top_k(monkeypatch, tmp_path):
    index_path = tmp_path / "chunks.jsonl"
    write_index(
        index_path,
        [
            {"content": f"ma túy chunk {i}", "metadata": {"source": f"{i}.md"}}
            for i in range(5)
        ],
    )
    monkeypatch.setattr(task6_lexical_search, "INDEX_PATH", index_path)

    results = task6_lexical_search.lexical_search("ma tuý", top_k=2)

    assert len(results) == 2


def test_lexical_search_returns_empty_when_index_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(task6_lexical_search, "INDEX_PATH", tmp_path / "missing.jsonl")

    assert task6_lexical_search.lexical_search("ma tuý") == []
