"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=True)

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


def normalize_text(text: str) -> str:
    """Chuẩn hóa tiếng Việt để detect intent ổn định hơn."""
    normalized = text.lower().replace("đ", "d").replace("Đ", "d")
    decomposed = unicodedata.normalize("NFD", normalized)
    no_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", no_accents)).strip()


def is_prohibited_procurement_query(query: str) -> bool:
    """
    Detect câu hỏi tìm cách mua/bán/nguồn/giá chất ma túy.

    Các câu hỏi pháp lý như "mua bán trái phép chất ma túy bị phạt thế nào"
    vẫn được phép trả lời nếu có intent hỏi luật/hình phạt rõ ràng.
    """
    normalized = normalize_text(query)
    drug_terms = {
        "ma tuy", "can sa", "heroin", "cocaine", "thuoc lac",
        "ketamine", "meth", "amphetamine", "hang trang",
    }
    procurement_terms = {
        "mua", "ban", "o dau", "gia", "re", "ship", "dat hang",
        "nguon", "cho nao", "kiem", "lay hang", "order",
    }
    legal_intent_terms = {
        "toi", "hinh phat", "phat", "luat", "dieu", "khoan",
        "quy dinh", "xu ly", "trach nhiem", "vi pham",
    }

    has_drug = any(term in normalized for term in drug_terms)
    has_procurement = any(term in normalized for term in procurement_terms)
    has_legal_intent = any(term in normalized for term in legal_intent_terms)
    return has_drug and has_procurement and not has_legal_intent


def build_safety_refusal(query: str, chunks: list[dict]) -> str:
    """Trả lời an toàn cho câu hỏi tìm cách mua/bán chất cấm."""
    citation = source_label(chunks[0], 1) if chunks else "nguồn pháp luật hiện có"
    return (
        "Tôi không thể hỗ trợ tìm nơi mua, giá bán, nguồn cung hoặc cách giao dịch "
        f"chất ma túy. Đây là nội dung liên quan đến hành vi bị pháp luật cấm [{citation}].\n\n"
        "Tôi có thể hỗ trợ theo hướng an toàn hơn, ví dụ: giải thích hành vi mua bán, "
        f"tàng trữ, vận chuyển trái phép chất ma túy bị xử lý thế nào; hoặc cung cấp "
        f"thông tin về cai nghiện, phòng chống ma túy và tác hại pháp lý [{citation}]."
    )


def safe_error_message(exc: Exception) -> str:
    """Rút gọn lỗi API, tránh đưa secret/key vào UI hoặc log dài."""
    message = str(exc)
    if "invalid_api_key" in message or "Incorrect API key" in message:
        return "OpenAI API key không hợp lệ hoặc đã bị revoke."
    if "insufficient_quota" in message:
        return "OpenAI API key hết quota."
    if len(message) > 240:
        message = message[:237].rstrip() + "..."
    return message


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = [chunks[0]]
    middle = chunks[2:]
    reordered.extend(middle)
    reordered.append(chunks[1])
    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type") or metadata.get("doc_type", "unknown")
        retrieval_source = chunk.get("source", "unknown")
        score = float(chunk.get("score", 0.0))
        label = (
            f"[Document {i} | Source: {source} | Type: {doc_type} | "
            f"Retrieval: {retrieval_source} | Score: {score:.3f}]"
        )
        context_parts.append(f"{label}\n{chunk.get('content', '')}\n")
    return "\n---\n".join(context_parts)


def build_user_message(query: str, context: str) -> str:
    """Tạo user prompt chứa context đã gắn nhãn nguồn."""
    return f"""Context:
{context}

---

Question: {query}"""


def source_label(chunk: dict, fallback_index: int = 1) -> str:
    """Lấy label citation ngắn từ metadata chunk."""
    metadata = chunk.get("metadata", {})
    return metadata.get("source") or metadata.get("path") or f"Source {fallback_index}"


def build_extractive_answer(query: str, chunks: list[dict]) -> str:
    """
    Fallback local: dùng các chunk retrieved làm câu trả lời có citation.

    Không tự suy diễn ngoài context; mỗi đoạn đều gắn citation source.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_parts = []
    for i, chunk in enumerate(chunks[:3], 1):
        content = " ".join(chunk.get("content", "").split())
        if len(content) > 450:
            content = content[:447].rstrip() + "..."
        citation = source_label(chunk, i)
        answer_parts.append(f"{content} [{citation}]")

    return "\n\n".join(answer_parts)


def call_openai(user_message: str) -> str:
    """Gọi OpenAI nếu có key thật; lỗi sẽ để caller fallback local."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if is_prohibited_procurement_query(query):
        safety_query = (
            "Luật Phòng chống ma túy cấm mua bán sử dụng trái phép chất ma túy "
            "và Bộ luật Hình sự xử lý tội mua bán tàng trữ vận chuyển ma túy"
        )
        chunks = retrieve(safety_query, top_k=top_k)
        return {
            "answer": build_safety_refusal(query, chunks),
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
            "generation_mode": "safety_refusal",
            "generation_error": "",
        }

    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = build_user_message(query, context)

    answer = ""
    generation_mode = "local_extractive"
    generation_error = ""
    if OPENAI_API_KEY:
        try:
            answer = call_openai(user_message)
            generation_mode = f"openai:{OPENAI_MODEL}"
        except Exception as exc:
            generation_error = safe_error_message(exc)
            print(f"! OpenAI generation failed, using local fallback: {generation_error}")

    if not answer.strip():
        answer = build_extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "generation_mode": generation_mode,
        "generation_error": generation_error,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
