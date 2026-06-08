from src import task4_chunking_indexing


def test_load_documents_reads_markdown_with_metadata(monkeypatch, tmp_path):
    standardized_dir = tmp_path / "standardized"
    legal_dir = standardized_dir / "legal"
    news_dir = standardized_dir / "news"
    legal_dir.mkdir(parents=True)
    news_dir.mkdir(parents=True)
    (legal_dir / "law.md").write_text("# Law\n\nNội dung pháp luật.", encoding="utf-8")
    (news_dir / "article.md").write_text("# News\n\nNội dung bài báo.", encoding="utf-8")

    monkeypatch.setattr(task4_chunking_indexing, "STANDARDIZED_DIR", standardized_dir)

    docs = task4_chunking_indexing.load_documents()

    assert len(docs) == 2
    assert {doc["metadata"]["type"] for doc in docs} == {"legal", "news"}
    assert all(doc["metadata"]["source"].endswith(".md") for doc in docs)


def test_chunk_documents_preserves_metadata_and_size_limit():
    docs = [
        {
            "content": "A" * 300 + "\n\n" + "B" * 300,
            "metadata": {"source": "sample.md", "type": "legal"},
        }
    ]

    chunks = task4_chunking_indexing.chunk_documents(docs)

    assert len(chunks) >= 2
    assert all(len(chunk["content"]) <= int(task4_chunking_indexing.CHUNK_SIZE * 1.1) for chunk in chunks)
    assert chunks[0]["metadata"]["source"] == "sample.md"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_embed_and_index_chunks_locally(monkeypatch, tmp_path):
    index_path = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(task4_chunking_indexing, "INDEX_PATH", index_path)
    chunks = [
        {"content": "ma túy pháp luật", "metadata": {"source": "a.md", "type": "legal", "chunk_index": 0}},
        {"content": "nghệ sĩ bài báo", "metadata": {"source": "b.md", "type": "news", "chunk_index": 0}},
    ]

    embedded = task4_chunking_indexing.embed_chunks(chunks)
    task4_chunking_indexing.index_to_vectorstore(embedded)

    assert index_path.exists()
    assert len(embedded[0]["embedding"]) == task4_chunking_indexing.EMBEDDING_DIM
    assert index_path.read_text(encoding="utf-8").count("\n") == 2
