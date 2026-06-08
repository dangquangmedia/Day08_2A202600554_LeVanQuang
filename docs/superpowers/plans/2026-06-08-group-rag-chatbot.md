# Group RAG Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable group RAG chatbot demo and a local evaluation pipeline that satisfies `group_project/README.md`.

**Architecture:** Streamlit provides the chat UI and calls Task 10 for retrieval-augmented generation with citations. Evaluation uses the existing Task 4-10 pipeline, compares hybrid+rereank against semantic-only retrieval, and writes a Markdown report.

**Tech Stack:** Python, Streamlit, existing `src.task9_retrieval_pipeline`, existing `src.task10_generation`, JSON golden dataset, pytest.

---

### Task 1: Group Tests

**Files:**
- Create: `tests/test_group_project.py`
- Modify: none

- [ ] Add tests that verify the golden dataset has at least 15 entries, each entry has required fields, evaluation metrics stay in `[0, 1]`, and config comparison returns two configs.

- [ ] Run: `pytest tests/test_group_project.py -q`

### Task 2: Streamlit Chatbot

**Files:**
- Create: `group_project/app.py`

- [ ] Add a Streamlit chatbot that keeps conversation history in `st.session_state`, calls `generate_with_citation`, displays answer text, retrieval/generation mode, and source chunks.

- [ ] Run: `python -m py_compile group_project/app.py`

### Task 3: Evaluation Dataset And Pipeline

**Files:**
- Modify: `group_project/evaluation/golden_dataset.json`
- Modify: `group_project/evaluation/eval_pipeline.py`
- Modify: `group_project/evaluation/results.md`

- [ ] Expand the golden dataset to at least 15 Q&A pairs covering legal documents and news documents.

- [ ] Implement deterministic local metrics: faithfulness, answer relevance, context recall, and context precision.

- [ ] Implement A/B comparison between `hybrid_rerank` and `semantic_only`.

- [ ] Export a complete Markdown report with scores, bottom cases, and recommendations.

- [ ] Run: `python group_project/evaluation/eval_pipeline.py`

### Task 4: Documentation

**Files:**
- Modify: `group_project/README.md`
- Modify: `requirements.txt` only if Streamlit is missing

- [ ] Fill architecture, deliverables, and run instructions.

- [ ] Run: `pytest tests/test_group_project.py tests/test_individual.py -q`

### Self-Review

- The plan covers chatbot UI, citations, conversation memory, source display, evaluation dataset, A/B comparison, report, and README.
- No placeholder implementation steps remain.
- Function names match the intended implementation files.
