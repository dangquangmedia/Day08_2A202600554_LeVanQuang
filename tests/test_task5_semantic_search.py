import json

from src import task5_semantic_search
from src.task4_chunking_indexing import create_local_embedding


def test_semantic_search_reads_task4_index_and_sorts_results(monkeypatch, tmp_path):
    index_path = tmp_path / "chunks.jsonl"
    chunks = [
        {
            "content": "hình phạt ma túy theo pháp luật",
            "metadata": {"source": "law.md", "type": "legal", "chunk_index": 0},
        },
        {
            "content": "nghệ sĩ tham gia chương trình truyền hình",
            "metadata": {"source": "news.md", "type": "news", "chunk_index": 0},
        },
    ]
    with index_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps({
                **chunk,
                "embedding": create_local_embedding(chunk["content"]),
            }, ensure_ascii=False) + "\n")

    monkeypatch.setattr(task5_semantic_search, "INDEX_PATH", index_path)

    results = task5_semantic_search.semantic_search("ma túy pháp luật", top_k=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["content"] == "hình phạt ma túy theo pháp luật"
    assert results[0]["metadata"]["doc_type"] == "legal"


def test_semantic_search_respects_top_k(monkeypatch, tmp_path):
    index_path = tmp_path / "chunks.jsonl"
    with index_path.open("w", encoding="utf-8") as file:
        for i in range(3):
            content = f"chunk {i} ma túy"
            file.write(json.dumps({
                "content": content,
                "metadata": {"source": f"{i}.md", "type": "legal", "chunk_index": i},
                "embedding": create_local_embedding(content),
            }, ensure_ascii=False) + "\n")

    monkeypatch.setattr(task5_semantic_search, "INDEX_PATH", index_path)

    results = task5_semantic_search.semantic_search("ma túy", top_k=2)

    assert len(results) == 2


def test_semantic_search_returns_empty_list_when_index_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(task5_semantic_search, "INDEX_PATH", tmp_path / "missing.jsonl")

    assert task5_semantic_search.semantic_search("ma túy") == []
