import json

from src import task3_convert_markdown


def test_convert_legal_docs_writes_markdown(monkeypatch, tmp_path):
    landing_dir = tmp_path / "landing"
    output_dir = tmp_path / "standardized"
    legal_dir = landing_dir / "legal"
    legal_dir.mkdir(parents=True)
    pdf_path = legal_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF sample")

    class FakeMarkItDown:
        def convert(self, path):
            assert path == str(pdf_path)
            return type("Result", (), {"text_content": "# Legal Doc\n\nNội dung pháp luật."})()

    monkeypatch.setattr(task3_convert_markdown, "LANDING_DIR", landing_dir)
    monkeypatch.setattr(task3_convert_markdown, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(task3_convert_markdown, "MarkItDown", FakeMarkItDown)

    task3_convert_markdown.convert_legal_docs()

    output_path = output_dir / "legal" / "sample.md"
    assert output_path.exists()
    assert "Nội dung pháp luật" in output_path.read_text(encoding="utf-8")


def test_convert_legal_docs_writes_fallback_when_pdf_has_no_text(monkeypatch, tmp_path):
    landing_dir = tmp_path / "landing"
    output_dir = tmp_path / "standardized"
    legal_dir = landing_dir / "legal"
    legal_dir.mkdir(parents=True)
    pdf_path = legal_dir / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF scanned")

    class FakeMarkItDown:
        def convert(self, path):
            assert path == str(pdf_path)
            return type("Result", (), {"text_content": "   "})()

    monkeypatch.setattr(task3_convert_markdown, "LANDING_DIR", landing_dir)
    monkeypatch.setattr(task3_convert_markdown, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(task3_convert_markdown, "MarkItDown", FakeMarkItDown)

    task3_convert_markdown.convert_legal_docs()

    content = (output_dir / "legal" / "scanned.md").read_text(encoding="utf-8")
    assert "Không trích xuất được text" in content
    assert len(content) > 200


def test_convert_news_articles_writes_markdown_with_metadata(monkeypatch, tmp_path):
    landing_dir = tmp_path / "landing"
    output_dir = tmp_path / "standardized"
    news_dir = landing_dir / "news"
    news_dir.mkdir(parents=True)
    article_path = news_dir / "article_01.json"
    article_path.write_text(
        json.dumps(
            {
                "url": "https://example.com/article",
                "title": "Tiêu đề bài báo",
                "date_crawled": "2026-06-08T00:00:00",
                "content_markdown": "Nội dung bài báo về ma túy.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(task3_convert_markdown, "LANDING_DIR", landing_dir)
    monkeypatch.setattr(task3_convert_markdown, "OUTPUT_DIR", output_dir)

    task3_convert_markdown.convert_news_articles()

    output_path = output_dir / "news" / "article_01.md"
    content = output_path.read_text(encoding="utf-8")
    assert "# Tiêu đề bài báo" in content
    assert "**Source:** https://example.com/article" in content
    assert "Nội dung bài báo về ma túy." in content
