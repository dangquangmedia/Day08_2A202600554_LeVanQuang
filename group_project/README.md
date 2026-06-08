# Bài Tập Nhóm - RAG Chatbot Pháp Luật Ma Tuý

## Mục Tiêu

Sản phẩm nhóm là một **RAG Chatbot** trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan. Chatbot sử dụng dữ liệu đã xử lý từ bài cá nhân:

- Văn bản pháp luật PDF trong `data/standardized/legal/`
- Tin tức đã crawl trong `data/standardized/news/`
- Chunk/index từ Task 4 trong `data/index/chunks.jsonl`
- Retrieval pipeline từ Task 9
- Generation có citation từ Task 10

---

## Deliverables

### RAG Chatbot

- [x] Giao diện chat bằng Streamlit: `group_project/app.py`
- [x] Trả lời có citation dựa trên Task 10
- [x] Hỗ trợ follow-up questions bằng conversation memory trong `st.session_state`
- [x] Hiển thị source documents đã dùng ở cuối giao diện

### RAG Evaluation Pipeline

- [x] Golden dataset 15 Q&A: `group_project/evaluation/golden_dataset.json`
- [x] Script evaluation: `group_project/evaluation/eval_pipeline.py`
- [x] Báo cáo kết quả: `group_project/evaluation/results.md`
- [x] So sánh A/B hai config:
  - Config A: hybrid search + rerank
  - Config B: semantic-only

---

## Kiến Trúc Hệ Thống

```text
User
  |
  v
Streamlit Chat UI (group_project/app.py)
  |
  v
Conversation Memory (st.session_state)
  |
  v
Task 10 - generate_with_citation()
  |
  v
Task 9 - Retrieval Pipeline
  |-- Task 5: Semantic Search
  |-- Task 6: Lexical Search / BM25
  |-- Task 7: Reranking
  |-- Task 8: PageIndex fallback
  |
  v
Task 4 Index (data/index/chunks.jsonl)
  |
  v
Legal Docs + News Markdown
```

---

## Luồng Chatbot

1. Người dùng nhập câu hỏi trong Streamlit.
2. App lấy các lượt hội thoại gần nhất để tạo contextual query cho follow-up questions.
3. `generate_with_citation()` gọi Task 9 để lấy top chunks.
4. Task 10 reorder chunks để giảm lỗi "lost in the middle".
5. Nếu `OPENAI_API_KEY` hợp lệ, hệ thống gọi OpenAI để tạo câu trả lời có citation.
6. Nếu API lỗi hoặc thiếu key, hệ thống fallback sang local extractive answer, vẫn có source citation.
7. UI hiển thị answer, retrieval mode, generation mode và source chunks.

---

## Evaluation

Framework sử dụng: **Local heuristic evaluator**.

Lý do chọn:

- Chạy được ngay trên local, không cần thêm quota DeepEval/RAGAS/TruLens.
- Không tốn thêm API trong lúc demo.
- Vẫn đo đủ 4 nhóm metric trong README: faithfulness, answer relevance, context recall, context precision.

Kết quả mới nhất:

| Metric | Hybrid + Rerank | Semantic-only | Delta |
|--------|-----------------|---------------|-------|
| Faithfulness | 0.999 | 0.999 | 0.000 |
| Answer relevance | 0.823 | 0.742 | 0.081 |
| Context recall | 0.952 | 0.850 | 0.102 |
| Context precision | 0.987 | 0.960 | 0.027 |
| Average | 0.940 | 0.888 | 0.052 |

Kết luận: config hybrid + rerank tốt hơn semantic-only, đặc biệt ở answer relevance và context recall.

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Thành viên 1 | Điền MSSV | Thu thập văn bản pháp luật, chuẩn hoá PDF/Markdown | Hoàn thành |
| Thành viên 2 | Điền MSSV | Crawl tin tức, làm sạch dữ liệu, tạo markdown | Hoàn thành |
| Thành viên 3 | Điền MSSV | Chunking, indexing, semantic/lexical retrieval | Hoàn thành |
| Thành viên 4 | Điền MSSV | Reranking, RAG chatbot, evaluation report | Hoàn thành |

---

## Hướng Dẫn Chạy

Chạy từ thư mục gốc repository:

```bash
pip install -r requirements.txt
```

Nếu chưa có index:

```bash
python src/task4_chunking_indexing.py
```

Chạy chatbot:

```bash
streamlit run group_project/app.py
```

Chạy evaluation:

```bash
python group_project/evaluation/eval_pipeline.py
```

Chạy test kiểm chứng:

```bash
pytest tests/test_group_project.py -q
pytest tests/test_individual.py -q
```

---

## Ghi Chú API

- `OPENAI_API_KEY` dùng cho Task 10 generation thật.
- Nếu OpenAI lỗi hoặc thiếu key, chatbot vẫn trả lời bằng local extractive fallback.
- `JINA_API_KEY` hiện cần balance nếu muốn dùng reranker API thật; bản hiện tại dùng reranking local để demo ổn định.
- PageIndex được giữ làm fallback theo Task 8, nhưng evaluation mặc định dùng pipeline local để tránh phụ thuộc quota.

---

## Hướng Phát Triển

Ở giai đoạn tiếp theo có thể phát triển thêm knowledge graph để xử lý câu hỏi khó, ví dụ câu hỏi nhiều bước, câu hỏi so sánh điều luật, hoặc truy vấn cần liên kết giữa người nổi tiếng, hành vi, văn bản pháp luật và mức xử phạt.
