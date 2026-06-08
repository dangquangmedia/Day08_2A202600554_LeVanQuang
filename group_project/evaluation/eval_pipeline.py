"""
Group project RAG evaluation pipeline.

This script is intentionally local-first: it evaluates the existing RAG pipeline
without requiring paid evaluation frameworks or extra API calls. The metrics map
to the README requirements:
    - faithfulness: answer tokens supported by retrieved contexts
    - answer_relevance: answer overlap with question + expected answer
    - context_recall: expected answer/context evidence found in retrieval
    - context_precision: share of retrieved contexts that are useful
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

MetricResult = dict[str, float]
PipelineResult = dict[str, object]
PipelineFn = Callable[[str, int], PipelineResult]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset from JSON."""
    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if len(data) < 15:
        raise ValueError("Golden dataset must contain at least 15 Q&A pairs.")
    return data


def normalize_text(text: str) -> str:
    """Lowercase, strip Vietnamese accents, and collapse punctuation."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    no_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", no_accents)).strip()


def tokenize(text: str) -> set[str]:
    """Tokenize normalized text and remove very short filler tokens."""
    stopwords = {
        "la", "va", "co", "cua", "cho", "the", "ve", "voi", "trong", "theo",
        "mot", "cac", "nhung", "duoc", "bi", "tu", "den", "thi", "khi", "noi",
    }
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1 and token not in stopwords
    }


def keyword_overlap_score(left: str, right: str) -> float:
    """Return bounded token recall from left into right."""
    left_tokens = tokenize(left)
    if not left_tokens:
        return 0.0
    right_tokens = tokenize(right)
    return round(len(left_tokens & right_tokens) / len(left_tokens), 4)


def source_text(source: dict) -> str:
    """Flatten source content and metadata for evidence scoring."""
    metadata = source.get("metadata", {}) if isinstance(source, dict) else {}
    return " ".join([
        str(source.get("content", "")) if isinstance(source, dict) else "",
        str(metadata.get("source", "")),
        str(metadata.get("path", "")),
        str(metadata.get("doc_type", "")),
        str(metadata.get("type", "")),
    ])


def calculate_metrics(item: dict, result: PipelineResult) -> MetricResult:
    """Calculate local approximations for the four required RAG metrics."""
    answer = str(result.get("answer", ""))
    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    context_texts = [source_text(source) for source in sources]
    combined_context = " ".join(context_texts)
    expected_answer = item["expected_answer"]
    expected_context = item["expected_context"]

    faithfulness = keyword_overlap_score(answer, combined_context)
    answer_relevance = max(
        keyword_overlap_score(item["question"], answer),
        keyword_overlap_score(expected_answer, answer),
    )
    context_recall = max(
        keyword_overlap_score(expected_answer, combined_context),
        keyword_overlap_score(expected_context, combined_context),
    )

    useful_contexts = 0
    target = f"{item['question']} {expected_answer} {expected_context}"
    for context in context_texts:
        if max(
            keyword_overlap_score(expected_answer, context),
            keyword_overlap_score(expected_context, context),
            keyword_overlap_score(target, context),
        ) >= 0.18:
            useful_contexts += 1
    context_precision = useful_contexts / len(context_texts) if context_texts else 0.0

    return {
        "faithfulness": round(min(faithfulness, 1.0), 4),
        "answer_relevance": round(min(answer_relevance, 1.0), 4),
        "context_recall": round(min(context_recall, 1.0), 4),
        "context_precision": round(min(context_precision, 1.0), 4),
    }


def evaluate_case(pipeline: PipelineFn, item: dict, top_k: int = 5) -> dict:
    """Run one question through a pipeline and score it."""
    result = pipeline(item["question"], top_k)
    metrics = calculate_metrics(item, result)
    return {
        "question": item["question"],
        "expected_context": item["expected_context"],
        "answer": str(result.get("answer", "")),
        "sources": result.get("sources", []),
        "retrieval_source": str(result.get("retrieval_source", "unknown")),
        "generation_mode": str(result.get("generation_mode", "unknown")),
        "metrics": metrics,
        "average": round(mean(metrics.values()), 4),
    }


def build_hybrid_pipeline() -> PipelineFn:
    """Config A: Task 9 hybrid retrieval with reranking, local answer synthesis."""
    from src.task10_generation import build_extractive_answer, reorder_for_llm
    from src.task9_retrieval_pipeline import retrieve

    def pipeline(question: str, top_k: int = 5) -> PipelineResult:
        chunks = retrieve(
            question,
            top_k=top_k,
            score_threshold=-1.0,
            use_reranking=True,
        )
        ordered = reorder_for_llm(chunks)
        return {
            "answer": build_extractive_answer(question, ordered),
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
            "generation_mode": "local_extractive_eval",
        }

    return pipeline


def build_semantic_pipeline() -> PipelineFn:
    """Config B: Task 5 semantic-only retrieval, local answer synthesis."""
    from src.task10_generation import build_extractive_answer, reorder_for_llm
    from src.task5_semantic_search import semantic_search

    def pipeline(question: str, top_k: int = 5) -> PipelineResult:
        chunks = semantic_search(question, top_k=top_k)
        marked = []
        for chunk in chunks:
            item = chunk.copy()
            item["source"] = "semantic"
            item["metadata"] = dict(item.get("metadata", {}))
            marked.append(item)
        ordered = reorder_for_llm(marked)
        return {
            "answer": build_extractive_answer(question, ordered),
            "sources": marked,
            "retrieval_source": "semantic",
            "generation_mode": "local_extractive_eval",
        }

    return pipeline


def summarize_cases(cases: list[dict]) -> dict:
    """Aggregate case-level scores."""
    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    scores = {
        metric: round(mean(case["metrics"][metric] for case in cases), 4)
        for metric in metric_names
    }
    scores["average"] = round(mean(scores.values()), 4)
    scores["cases"] = cases
    return scores


def compare_configs(
    golden_dataset: list[dict],
    pipelines: dict[str, PipelineFn] | None = None,
    top_k: int = 5,
) -> dict:
    """Compare at least two RAG configs."""
    if pipelines is None:
        pipelines = {
            "hybrid_rerank": build_hybrid_pipeline(),
            "semantic_only": build_semantic_pipeline(),
        }

    comparison = {}
    for name, pipeline in pipelines.items():
        cases = [evaluate_case(pipeline, item, top_k=top_k) for item in golden_dataset]
        comparison[name] = summarize_cases(cases)
    return comparison


def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """Compatibility wrapper: return local metric results in DeepEval-like shape."""
    cases = [evaluate_case(rag_pipeline, item) for item in golden_dataset]
    return summarize_cases(cases)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """Compatibility wrapper: return local metric results in RAGAS-like shape."""
    return evaluate_with_deepeval(rag_pipeline, golden_dataset)


def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """Compatibility wrapper: return local metric results in TruLens-like shape."""
    return evaluate_with_deepeval(rag_pipeline, golden_dataset)


def markdown_score(value: float) -> str:
    """Format score for Markdown tables."""
    return f"{value:.3f}"


def export_results(comparison: dict, path: Path = RESULTS_PATH) -> None:
    """Export evaluation results to Markdown."""
    config_a = comparison["hybrid_rerank"]
    config_b = comparison["semantic_only"]
    delta_avg = config_a["average"] - config_b["average"]

    worst_cases = sorted(config_a["cases"], key=lambda case: case["average"])[:3]

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "Framework sử dụng: **Local heuristic evaluator**.",
        "",
        "Lý do chọn: chạy được offline, không tốn API quota, bám đúng 4 nhóm metric trong README và phù hợp demo lớp học.",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (semantic-only) | Delta |",
        "|--------|---------------------------|--------------------------|-------|",
    ]

    for metric in ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average"]:
        lines.append(
            f"| {metric} | {markdown_score(config_a[metric])} | "
            f"{markdown_score(config_b[metric])} | {markdown_score(config_a[metric] - config_b[metric])} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## A/B Comparison Analysis",
        "",
        "**Config A:** Hybrid retrieval từ Task 9, kết hợp semantic search, lexical search, RRF và reranking local.",
        "",
        "**Config B:** Semantic-only retrieval từ Task 5, dùng cùng local extractive answer để so sánh công bằng phần retrieval.",
        "",
        f"**Kết luận:** Config A {'tốt hơn' if delta_avg >= 0 else 'thấp hơn'} Config B với chênh lệch average {markdown_score(delta_avg)}. "
        "Hybrid thường ổn hơn khi câu hỏi có từ khoá pháp lý hoặc tên riêng vì lexical search bổ sung cho dense retrieval.",
        "",
        "---",
        "",
        "## Worst Performers (Bottom 3 - Config A)",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Precision | Root Cause |",
        "|---|----------|--------------|-----------|--------|-----------|------------|",
    ])

    for index, case in enumerate(worst_cases, 1):
        metrics = case["metrics"]
        root_cause = "Retriever chưa lấy đủ đúng source hoặc câu hỏi cần evidence chi tiết hơn."
        question = case["question"].replace("|", " ")
        lines.append(
            f"| {index} | {question} | {markdown_score(metrics['faithfulness'])} | "
            f"{markdown_score(metrics['answer_relevance'])} | {markdown_score(metrics['context_recall'])} | "
            f"{markdown_score(metrics['context_precision'])} | {root_cause} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "### Cải tiến 1",
        "**Action:** Tăng chất lượng metadata khi chunking, đặc biệt `source`, `doc_type`, số điều/khoản nếu trích từ văn bản pháp luật.",
        "",
        "**Expected impact:** Tăng context recall và giúp citation rõ hơn.",
        "",
        "### Cải tiến 2",
        "**Action:** Thêm query rewriting cho follow-up questions trước khi retrieval.",
        "",
        "**Expected impact:** Giảm lỗi khi người dùng hỏi tiếp bằng đại từ như \"nó\", \"trường hợp này\", \"mức phạt đó\".",
        "",
        "### Cải tiến 3",
        "**Action:** Bổ sung reranker API khi Jina có balance hoặc dùng cross-encoder local.",
        "",
        "**Expected impact:** Tăng context precision, giảm chunk nhiễu ở top results.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run A/B evaluation and export report."""
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    comparison = compare_configs(golden_dataset)
    export_results(comparison)
    print(f"Wrote {RESULTS_PATH}")
    for name, scores in comparison.items():
        print(f"{name}: average={scores['average']:.3f}")


if __name__ == "__main__":
    main()
