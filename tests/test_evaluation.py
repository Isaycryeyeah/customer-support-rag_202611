import json
from unittest.mock import Mock

import pytest

from app.evaluation.eval_retrieval import hit_at_k, recall_at_k, ndcg_at_k, retrieval_metrics, summarize_rows
from app.evaluation.eval_generation import evaluate_generation, refusal_metrics


def test_multi_gold_hit_is_not_recall():
    assert hit_at_k(['a'], ['a', 'b'], 1) == 1
    assert recall_at_k(['a'], ['a', 'b'], 1) == 0.5
    assert recall_at_k(['a', 'a'], ['a', 'b'], 2) == 0.5
    assert ndcg_at_k(['a', 'b'], ['a', 'b'], 5) == pytest.approx(1)
    assert ndcg_at_k(['a', 'a'], ['a', 'b'], 5) < 1


def test_no_answer_excluded_from_retrieval_denominator():
    rows = [{'metrics': retrieval_metrics(['a'], ['a'])}, {'metrics': retrieval_metrics(['b'], [])}]
    summary = summarize_rows(rows)
    assert summary['answerable_count'] == summary['unanswerable_count'] == 1
    assert summary['recall@1'] == 1


def test_fallback_gets_no_generation_quality_credit(tmp_path):
    path = tmp_path / 'eval.jsonl'
    path.write_text(json.dumps({'question': 'q', 'expected_points': ['结论'], 'need_human': True}), encoding='utf-8')
    rag = Mock()
    rag.answer.return_value = {'answer': '结论 处理步骤 注意事项', 'answer_type': 'service_unavailable',
                               'generation': {'success': False}, 'need_human': True}
    result = evaluate_generation(rag, path)
    assert result['request_success_rate'] == result['answer_coverage'] == 0
    assert result['generation_failure_rate'] == 1
    assert result['format_follow_rate'] is None and result['answer_point_coverage'] is None
    assert result['format_follow_rate_all_requests'] == 0


def test_refusal_metrics_keep_service_and_safety_separate():
    rows = [
        {'answerable': False, 'answer_type': 'low_confidence', 'service_failure': False},
        {'answerable': True, 'answer_type': 'low_confidence', 'service_failure': False},
        {'answerable': False, 'answer_type': 'generated', 'service_failure': False},
        {'answerable': True, 'answer_type': 'generated', 'service_failure': False},
        {'answerable': False, 'answer_type': 'service_unavailable', 'service_failure': True},
        {'answerable': True, 'answer_type': 'safety_refusal', 'service_failure': False, 'expected_safety': True},
    ]
    metric = refusal_metrics(rows)
    assert metric['evaluated_count'] == 4 and metric['excluded_count'] == 2
    assert metric['precision'] == metric['recall'] == metric['false_refusal_rate'] == 0.5
    assert refusal_metrics([])['precision'] is None
