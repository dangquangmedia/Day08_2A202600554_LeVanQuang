import json

from src import task8_pageindex_vectorless


def write_index(path, chunks):
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def test_pageindex_search_uses_local_index_fallback(monkeypatch, tmp_path):
    index_path = tmp_path / "chunks.jsonl"
    write_index(
        index_path,
        [
            {
                "content": "# Luật\n\nHình phạt tàng trữ trái phép chất ma túy.",
                "metadata": {"source": "law.md", "path": "legal/law.md", "type": "legal"},
            },
            {
                "content": "# Tin tức\n\nNghệ sĩ biểu diễn trên sân khấu.",
                "metadata": {"source": "news.md", "path": "news/news.md", "type": "news"},
            },
        ],
    )
    monkeypatch.setattr(task8_pageindex_vectorless, "INDEX_PATH", index_path)

    results = task8_pageindex_vectorless.pageindex_search("hình phạt ma tuý", top_k=1)

    assert len(results) == 1
    assert results[0]["source"] == "pageindex"
    assert "Hình phạt" in results[0]["content"]
    assert results[0]["metadata"]["retrieval_mode"] == "local_pageindex_fallback"


def test_pageindex_search_returns_empty_for_missing_index(monkeypatch, tmp_path):
    monkeypatch.setattr(task8_pageindex_vectorless, "INDEX_PATH", tmp_path / "missing.jsonl")

    assert task8_pageindex_vectorless.pageindex_search("ma tuý") == []


def test_upload_documents_skips_api_when_no_key(monkeypatch, tmp_path):
    standardized_dir = tmp_path / "standardized"
    (standardized_dir / "legal").mkdir(parents=True)
    (standardized_dir / "legal" / "law.md").write_text("content", encoding="utf-8")
    monkeypatch.setattr(task8_pageindex_vectorless, "STANDARDIZED_DIR", standardized_dir)
    monkeypatch.setattr(task8_pageindex_vectorless, "PAGEINDEX_API_KEY", "")

    assert task8_pageindex_vectorless.upload_documents() == []
