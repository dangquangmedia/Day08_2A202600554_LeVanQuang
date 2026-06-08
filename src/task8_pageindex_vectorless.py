"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "chunks.jsonl"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "pageindex_manifest.json"


def normalize_text(text: str) -> str:
    """Normalize tiếng Việt để query 'ma tuý' khớp content 'ma túy'."""
    text = text.lower()
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho fallback vectorless local."""
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def load_manifest() -> dict:
    """Đọc manifest PageIndex upload nếu đã có."""
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict):
    """Lưu manifest doc_id để tránh upload lặp không cần thiết."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_markdown_files() -> list[Path]:
    """Liệt kê markdown documents đã chuẩn hóa từ Task 3."""
    if not STANDARDIZED_DIR.exists():
        return []
    return sorted(STANDARDIZED_DIR.rglob("*.md"))


def load_local_chunks() -> list[dict]:
    """Đọc chunks từ index Task 4 để làm PageIndex fallback local."""
    if not INDEX_PATH.exists():
        return []

    chunks = []
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def structural_score(query: str, chunk: dict) -> float:
    """
    Chấm điểm vectorless local dựa trên cấu trúc + lexical overlap.

    PageIndex thật dùng structural understanding của document. Fallback này giữ
    cùng tinh thần ở mức local: heading/path/type/source được bonus nhẹ, còn
    content overlap là tín hiệu chính.
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    content = chunk.get("content", "")
    metadata = chunk.get("metadata", {})
    content_tokens = set(tokenize(content))
    source_tokens = set(tokenize(metadata.get("source", "")))
    path_tokens = set(tokenize(metadata.get("path", "")))
    type_tokens = set(tokenize(metadata.get("type", "")))

    content_overlap = len(query_tokens.intersection(content_tokens)) / len(query_tokens)
    source_overlap = len(query_tokens.intersection(source_tokens)) / len(query_tokens)
    path_overlap = len(query_tokens.intersection(path_tokens)) / len(query_tokens)
    type_overlap = len(query_tokens.intersection(type_tokens)) / len(query_tokens)

    heading_bonus = 0.0
    first_line = content.splitlines()[0] if content.splitlines() else ""
    if first_line.startswith("#"):
        heading_tokens = set(tokenize(first_line))
        heading_bonus = len(query_tokens.intersection(heading_tokens)) / len(query_tokens)

    return (
        1.0 * content_overlap
        + 0.2 * heading_bonus
        + 0.15 * source_overlap
        + 0.1 * path_overlap
        + 0.05 * type_overlap
    )


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("! Bỏ qua upload PageIndex: thiếu PAGEINDEX_API_KEY.")
        return []

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    manifest = load_manifest()
    uploaded = []

    for md_file in iter_markdown_files():
        relative_path = str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/")
        existing = manifest.get(relative_path)
        current_size = md_file.stat().st_size
        if existing and existing.get("size") == current_size:
            uploaded.append(existing)
            print(f"  = Cached: {relative_path}")
            continue

        response = client.submit_document(str(md_file))
        doc_id = (
            response.get("doc_id")
            or response.get("document_id")
            or response.get("id")
        )
        record = {
            "doc_id": doc_id,
            "path": relative_path,
            "filename": md_file.name,
            "type": md_file.parent.name,
            "size": current_size,
            "response": response,
        }
        manifest[relative_path] = record
        uploaded.append(record)
        print(f"  ✓ Uploaded: {relative_path}")

    save_manifest(manifest)
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0 or not query.strip():
        return []

    scored_chunks = []
    for chunk in load_local_chunks():
        score = structural_score(query, chunk)
        if score <= 0:
            continue
        metadata = dict(chunk.get("metadata", {}))
        metadata["retrieval_mode"] = "local_pageindex_fallback"
        scored_chunks.append({
            "content": chunk.get("content", ""),
            "score": float(score),
            "metadata": metadata,
            "source": "pageindex",
        })

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


if __name__ == "__main__":
    if "--upload" in sys.argv:
        if not PAGEINDEX_API_KEY:
            print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
            print("  Đăng ký tại: https://pageindex.ai/")
            raise SystemExit(1)

        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
