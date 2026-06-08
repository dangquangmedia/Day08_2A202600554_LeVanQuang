"""Streamlit RAG chatbot for the group project."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env", override=True)

from src.task10_generation import generate_with_citation  # noqa: E402

SUGGESTED_QUESTIONS = [
    "Hình phạt cho tội tàng trữ ma túy?",
    "Các hình thức cai nghiện theo Luật 2021?",
    "Ca sĩ Chi Dân bị bắt ở đâu, dùng chất gì?",
    "Thời gian cai nghiện bắt buộc là bao lâu?",
]


def build_contextual_query(messages: list[dict], question: str) -> str:
    """Attach recent conversation turns so follow-up questions have context."""
    recent = messages[-6:]
    if not recent:
        return question

    history_lines = []
    for message in recent:
        role = "User" if message["role"] == "user" else "Assistant"
        content = " ".join(str(message["content"]).split())
        if len(content) > 700:
            content = content[:697].rstrip() + "..."
        history_lines.append(f"{role}: {content}")

    return (
        "Ngữ cảnh hội thoại gần đây:\n"
        + "\n".join(history_lines)
        + f"\n\nCâu hỏi mới: {question}"
    )


def count_unique_documents(sources: list[dict]) -> int:
    """Count unique source documents from retrieved chunks."""
    document_names = set()
    for source in sources:
        metadata = source.get("metadata", {})
        title = metadata.get("source") or metadata.get("path")
        if title:
            document_names.add(str(title))
    return len(document_names)


def render_sources(sources: list[dict]) -> None:
    """Display source chunks used by the answer."""
    if not sources:
        st.info("Không có source document được trả về.")
        return

    st.caption(
        f"Đã truy xuất {len(sources)} chunks từ "
        f"{count_unique_documents(sources)} tài liệu nguồn."
    )
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = metadata.get("source") or metadata.get("path") or f"Source {index}"
        score = float(source.get("score", 0.0))
        with st.expander(f"{index}. {title} - score {score:.3f}"):
            st.caption(
                f"type={metadata.get('type') or metadata.get('doc_type', 'unknown')} | "
                f"retrieval={source.get('source', 'unknown')}"
            )
            st.write(source.get("content", ""))


def init_state() -> None:
    """Initialize Streamlit session state."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_meta", {})
    st.session_state.setdefault("pending_question", "")


def submit_question(question: str, top_k: int) -> None:
    """Run one question through Task 10 and update chat state."""
    st.session_state.messages.append({"role": "user", "content": question})
    contextual_query = build_contextual_query(st.session_state.messages[:-1], question)
    result = generate_with_citation(contextual_query, top_k=top_k)
    answer = result["answer"]
    sources = result.get("sources", [])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = sources
    st.session_state.last_meta = {
        "retrieval_source": result.get("retrieval_source", "unknown"),
        "generation_mode": result.get("generation_mode", "unknown"),
        "generation_error": result.get("generation_error", ""),
        "retrieved_chunks": len(sources),
        "retrieved_documents": count_unique_documents(sources),
    }


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="DrugLaw RAG Chatbot", layout="wide")
    init_state()

    with st.sidebar:
        st.header("Cấu Hình RAG Pipeline")
        top_k = st.slider("Số lượng tài liệu truy xuất (Top-K Chunks)", min_value=3, max_value=8, value=5)
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_meta = {}
            st.session_state.pending_question = ""
            st.rerun()

        st.divider()
        st.subheader("Gợi Ý Câu Hỏi")
        for suggestion in SUGGESTED_QUESTIONS:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

        st.divider()
        st.write("Pipeline")
        st.caption("Task 9 Retrieval -> Task 10 Generation with citation")
        if st.session_state.last_meta:
            col_a, col_b = st.columns(2)
            col_a.metric("Chunks", st.session_state.last_meta.get("retrieved_chunks", 0))
            col_b.metric("Docs", st.session_state.last_meta.get("retrieved_documents", 0))
            st.metric("Retrieval", st.session_state.last_meta.get("retrieval_source", "unknown"))
            st.metric("Generation", st.session_state.last_meta.get("generation_mode", "unknown"))
            if st.session_state.last_meta.get("generation_error"):
                st.warning(st.session_state.last_meta["generation_error"])

    st.title("DrugLaw RAG Chatbot")
    st.caption("Hỏi đáp về pháp luật ma tuý và tin tức liên quan, có citation và source documents.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = ""
        with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
            submit_question(question, top_k)
        st.rerun()

    question = st.chat_input("Nhập câu hỏi về pháp luật ma tuý hoặc tin tức liên quan")
    if question:
        with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
            submit_question(question, top_k)
        st.rerun()

    st.divider()
    st.subheader("Source Documents")
    render_sources(st.session_state.last_sources)


if __name__ == "__main__":
    main()
