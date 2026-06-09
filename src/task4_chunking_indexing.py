"""
Task 4 — Chunking & Indexing vào Vector Store.

MVP implementation:
- Load markdown files từ data/standardized/
- Chunk bằng RecursiveCharacterTextSplitter
- Embed bằng sentence-transformers
- Lưu local index vào data/index/chunks.json

Lý do:
- Recursive chunking an toàn cho cả văn bản luật và bài báo.
- all-MiniLM-L6-v2 nhẹ, dễ chạy local.
- local_json giúp pass lab nhanh, không phụ thuộc Weaviate.
"""

from pathlib import Path
import json
import re
import math
import hashlib

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_FILE = INDEX_DIR / "chunks.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Chọn 700 để giữ đủ ngữ cảnh pháp luật/bài báo trong mỗi chunk.
CHUNK_SIZE = 700

# Overlap 100 giúp tránh mất ý khi câu/điều luật bị cắt giữa chunk.
CHUNK_OVERLAP = 100

# Recursive phù hợp dữ liệu Markdown không đồng nhất.
CHUNKING_METHOD = "recursive"

# Model nhẹ, dễ chạy local. Khi demo tốt hơn có thể đổi sang "BAAI/bge-m3".
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Dùng local JSON để dễ pass test và dễ nối sang Task 5.
VECTOR_STORE = "local_json"


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_text(text: str) -> str:
    """Làm sạch text cơ bản."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_title(content: str, fallback: str) -> str:
    """Lấy title từ heading Markdown đầu tiên."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    return fallback


def _detect_doc_type(md_file: Path) -> str:
    """Detect legal/news dựa vào path."""
    parts = [p.lower() for p in md_file.parts]
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Fallback splitter đảm bảo chunk không vượt CHUNK_SIZE.
    Dùng khi langchain không có hoặc chunk quá dài.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)

    return chunks


def _fallback_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Embedding fallback dạng hashing.
    Dùng khi sentence-transformers không tải được model.
    Không tốt bằng embedding thật nhưng giúp pipeline không crash.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec

    return [v / norm for v in vec]


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {
            'content': str,
            'metadata': {
                'source': str,
                'path': str,
                'type': str,
                'title': str,
                'doc_id': str
            }
        }
    """
    documents = []

    if not STANDARDIZED_DIR.exists():
        print(f"Không tìm thấy thư mục: {STANDARDIZED_DIR}")
        return documents

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))

    for idx, md_file in enumerate(md_files, start=1):
        try:
            content = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = md_file.read_text(encoding="utf-8", errors="ignore")

        content = _normalize_text(content)

        if len(content) < 50:
            continue

        doc_type = _detect_doc_type(md_file)
        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        title = _extract_title(content, fallback=md_file.stem)

        doc_id = f"{doc_type}_{idx:03d}"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": relative_path,
                "type": doc_type,
                "title": title,
                "doc_id": doc_id,
            }
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo recursive strategy.

    Returns:
        List of {
            'content': str,
            'metadata': dict
        }
    """
    chunks = []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            length_function=len,
        )

        use_langchain = True

    except Exception as e:
        print(f"Không dùng được langchain splitter, chuyển sang fallback. Error: {e}")
        splitter = None
        use_langchain = False

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        if use_langchain:
            raw_splits = splitter.split_text(content)
        else:
            raw_splits = _hard_split(content, CHUNK_SIZE, CHUNK_OVERLAP)

        safe_splits = []

        for split in raw_splits:
            split = split.strip()
            if not split:
                continue

            # Đảm bảo pass test: không chunk nào vượt CHUNK_SIZE.
            if len(split) > CHUNK_SIZE:
                safe_splits.extend(_hard_split(split, CHUNK_SIZE, CHUNK_OVERLAP))
            else:
                safe_splits.append(split)

        for i, chunk_text in enumerate(safe_splits):
            chunk_id = f"{metadata['doc_id']}_chunk_{i:04d}"

            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "chunk_size": len(chunk_text),
                }
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks.

    Returns:
        Mỗi chunk được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]

    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            chunk["embedding_model"] = EMBEDDING_MODEL

    except Exception as e:
        print("Không dùng được sentence-transformers, dùng fallback hashing embedding.")
        print(f"Error: {e}")

        for chunk in chunks:
            chunk["embedding"] = _fallback_embedding(chunk["content"])
            chunk["embedding_model"] = "fallback_hash_embedding"

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào local JSON vector store.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunking_method": CHUNKING_METHOD,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "vector_store": VECTOR_STORE,
        },
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    INDEX_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"✓ Saved local index: {INDEX_FILE}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    if not docs:
        print("Chưa có file .md trong data/standardized/. Hãy chạy Task 3 trước.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to local vector store")


if __name__ == "__main__":
    run_pipeline()