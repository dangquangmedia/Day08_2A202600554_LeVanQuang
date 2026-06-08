import json
from pathlib import Path

from group_project.evaluation.eval_pipeline import (
    compare_configs,
    evaluate_case,
    keyword_overlap_score,
    load_golden_dataset,
)
from group_project.app import count_unique_documents


def test_golden_dataset_has_required_size_and_fields():
    dataset = load_golden_dataset()

    assert len(dataset) >= 15
    for item in dataset:
        assert item["question"].strip()
        assert item["expected_answer"].strip()
        assert item["expected_context"].strip()


def test_keyword_overlap_score_is_bounded():
    assert keyword_overlap_score("", "abc") == 0.0
    assert keyword_overlap_score("ma tuy", "ma tuy") == 1.0
    assert 0.0 <= keyword_overlap_score("ma tuy", "luat ma tuy") <= 1.0


def test_evaluate_case_returns_four_metrics():
    item = {
        "question": "Luật phòng chống ma tuý quy định gì?",
        "expected_answer": "Luật quy định phòng ngừa, cai nghiện và quản lý người sử dụng trái phép chất ma tuý.",
        "expected_context": "luat-phong-chong-ma-tuy-2021.md",
    }

    def fake_pipeline(question, top_k=5):
        return {
            "answer": "Luật quy định phòng ngừa và cai nghiện ma tuý. [luat-phong-chong-ma-tuy-2021.md]",
            "sources": [
                {
                    "content": "Luật quy định phòng ngừa, cai nghiện và quản lý người sử dụng trái phép chất ma tuý.",
                    "metadata": {"source": "luat-phong-chong-ma-tuy-2021.md"},
                    "score": 0.9,
                }
            ],
            "retrieval_source": "hybrid",
            "generation_mode": "local_extractive",
        }

    result = evaluate_case(fake_pipeline, item)

    assert set(result["metrics"]) == {
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "context_precision",
    }
    assert all(0.0 <= score <= 1.0 for score in result["metrics"].values())


def test_compare_configs_accepts_injected_pipelines():
    dataset = load_golden_dataset()[:2]

    def good_pipeline(question, top_k=5):
        return {
            "answer": "ma tuý pháp luật cai nghiện hình phạt",
            "sources": [{"content": "ma tuý pháp luật cai nghiện hình phạt", "metadata": {"source": "demo.md"}}],
            "retrieval_source": "test",
            "generation_mode": "test",
        }

    comparison = compare_configs(
        dataset,
        pipelines={
            "hybrid_rerank": good_pipeline,
            "semantic_only": good_pipeline,
        },
    )

    assert set(comparison) == {"hybrid_rerank", "semantic_only"}
    assert all("average" in config for config in comparison.values())


def test_count_unique_documents_uses_source_metadata():
    sources = [
        {"metadata": {"source": "a.md"}},
        {"metadata": {"source": "a.md"}},
        {"metadata": {"source": "b.md"}},
        {"metadata": {"path": "c.md"}},
    ]

    assert count_unique_documents(sources) == 3
