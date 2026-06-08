import sys
from types import SimpleNamespace

import pytest

from src import task2_crawl_news


def test_article_urls_are_configured():
    assert len(task2_crawl_news.ARTICLE_URLS) >= 5
    assert all(url.startswith("https://") for url in task2_crawl_news.ARTICLE_URLS)


@pytest.mark.asyncio
async def test_crawl_article_returns_metadata_and_markdown(monkeypatch):
    class FakeCrawler:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def arun(self, url):
            return SimpleNamespace(
                metadata={},
                markdown="# Title From Markdown\n\nNội dung bài báo.",
            )

    monkeypatch.setitem(
        sys.modules,
        "crawl4ai",
        SimpleNamespace(AsyncWebCrawler=FakeCrawler),
    )

    article = await task2_crawl_news.crawl_article("https://example.com/news")

    assert article["url"] == "https://example.com/news"
    assert article["title"] == "Title From Markdown"
    assert "date_crawled" in article
    assert article["content_markdown"].startswith("# Title From Markdown")
