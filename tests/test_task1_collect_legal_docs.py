from src import task1_collect_legal_docs


def test_list_legal_files_returns_supported_documents(monkeypatch, tmp_path):
    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    (legal_dir / "law.pdf").write_bytes(b"x" * 2048)
    (legal_dir / "decree.docx").write_bytes(b"x" * 2048)
    (legal_dir / ".gitkeep").write_text("", encoding="utf-8")
    (legal_dir / "note.txt").write_text("ignore", encoding="utf-8")

    monkeypatch.setattr(task1_collect_legal_docs, "DATA_DIR", legal_dir)

    files = task1_collect_legal_docs.list_legal_files()

    assert [file.name for file in files] == ["decree.docx", "law.pdf"]


def test_validate_legal_files_reports_ready(monkeypatch, tmp_path):
    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    for filename in ["a.pdf", "b.pdf", "c.docx"]:
        (legal_dir / filename).write_bytes(b"x" * 2048)

    monkeypatch.setattr(task1_collect_legal_docs, "DATA_DIR", legal_dir)

    result = task1_collect_legal_docs.validate_legal_files()

    assert result["is_ready"] is True
    assert result["file_count"] == 3
    assert result["small_files"] == []


def test_validate_legal_files_reports_missing_or_too_small(monkeypatch, tmp_path):
    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    (legal_dir / "tiny.pdf").write_bytes(b"x")

    monkeypatch.setattr(task1_collect_legal_docs, "DATA_DIR", legal_dir)

    result = task1_collect_legal_docs.validate_legal_files()

    assert result["is_ready"] is False
    assert result["file_count"] == 1
    assert result["missing_count"] == 2
    assert result["small_files"] == ["tiny.pdf"]
