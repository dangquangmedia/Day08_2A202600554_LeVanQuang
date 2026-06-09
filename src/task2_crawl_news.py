import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://tuoitre.vn/nu-dien-vien-tung-thu-vai-hoai-thatcher-bi-bat-vi-mua-ban-ma-tuy-20230423174834021.htm",
    "https://thanhnien.vn/dien-vien-hai-tran-huu-tin-lanh-7-nam-6-thang-tu-185230428134549434.htm",
    "https://vietnamnet.vn/chi-dan-an-tay-truc-phuong-la-nhung-mat-xich-cuoi-trong-duong-day-ma-tuy-2342032.html",
    "https://tuoitre.vn/nha-thiet-ke-cong-tri-lien-quan-ma-tuy-nguoi-noi-tieng-cung-la-cong-dan-deu-bi-xu-ly-nghiem-20250724192919372.htm",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
]


def extract_title_from_markdown(markdown: str) -> str:
    """Lấy title từ dòng markdown đầu tiên dạng # Title."""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    return "Unknown"


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Output:
    {
        "url": str,
        "title": str,
        "date_crawled": str,
        "content_markdown": str
    }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    markdown = getattr(result, "markdown", "") or ""
    metadata = getattr(result, "metadata", {}) or {}

    title = metadata.get("title") or extract_title_from_markdown(markdown)

    if len(markdown.strip()) < 500:
        raise ValueError(
            f"Crawl content quá ngắn cho URL: {url}. "
            "Có thể website chặn crawl hoặc chưa load đủ nội dung."
        )

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

        try:
            article = await crawl_article(url)

            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename

            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            print(f"  ✓ Saved: {filepath}")

        except Exception as e:
            print(f"  ✗ Failed: {url}")
            print(f"    Error: {e}")

        await asyncio.sleep(1)


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        asyncio.run(crawl_all())