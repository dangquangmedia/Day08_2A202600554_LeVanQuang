# RAG Evaluation Results

## Framework sử dụng

Framework sử dụng: **Local heuristic evaluator**.

Lý do chọn: chạy được offline, không tốn API quota, bám đúng 4 nhóm metric trong README và phù hợp demo lớp học.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (semantic-only) | Delta |
|--------|---------------------------|--------------------------|-------|
| faithfulness | 0.999 | 0.999 | 0.000 |
| answer_relevance | 0.823 | 0.742 | 0.081 |
| context_recall | 0.952 | 0.850 | 0.102 |
| context_precision | 0.987 | 0.960 | 0.027 |
| average | 0.940 | 0.888 | 0.052 |

---

## A/B Comparison Analysis

**Config A:** Hybrid retrieval từ Task 9, kết hợp semantic search, lexical search, RRF và reranking local.

**Config B:** Semantic-only retrieval từ Task 5, dùng cùng local extractive answer để so sánh công bằng phần retrieval.

**Kết luận:** Config A tốt hơn Config B với chênh lệch average 0.052. Hybrid thường ổn hơn khi câu hỏi có từ khoá pháp lý hoặc tên riêng vì lexical search bổ sung cho dense retrieval.

---

## Worst Performers (Bottom 3 - Config A)

| # | Question | Faithfulness | Relevance | Recall | Precision | Root Cause |
|---|----------|--------------|-----------|--------|-----------|------------|
| 1 | Khi không tìm thấy bằng chứng trong tài liệu, chatbot cần trả lời thế nào? | 1.000 | 0.833 | 0.696 | 0.800 | Retriever chưa lấy đủ đúng source hoặc câu hỏi cần evidence chi tiết hơn. |
| 2 | Nguồn tin về Chi Dân và An Tây liên quan đến hành vi gì? | 1.000 | 0.833 | 0.667 | 1.000 | Retriever chưa lấy đủ đúng source hoặc câu hỏi cần evidence chi tiết hơn. |
| 3 | Hiệp Gà từng bị xử lý vì hành vi gì theo nguồn tin đã thu thập? | 1.000 | 0.583 | 1.000 | 1.000 | Retriever chưa lấy đủ đúng source hoặc câu hỏi cần evidence chi tiết hơn. |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng chất lượng metadata khi chunking, đặc biệt `source`, `doc_type`, số điều/khoản nếu trích từ văn bản pháp luật.

**Expected impact:** Tăng context recall và giúp citation rõ hơn.

### Cải tiến 2
**Action:** Thêm query rewriting cho follow-up questions trước khi retrieval.

**Expected impact:** Giảm lỗi khi người dùng hỏi tiếp bằng đại từ như "nó", "trường hợp này", "mức phạt đó".

### Cải tiến 3
**Action:** Bổ sung reranker API khi Jina có balance hoặc dùng cross-encoder local.

**Expected impact:** Tăng context precision, giảm chunk nhiễu ở top results.
