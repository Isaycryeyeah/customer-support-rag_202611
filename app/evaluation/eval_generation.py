"""Service failures cannot earn generation-quality credit."""
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def check_format(answer):
    return all(section in answer for section in ['结论', '处理步骤', '注意事项'])


def check_expected_points(answer, expected_points):
    return sum(point in answer for point in expected_points) / len(expected_points) if expected_points else None


def refusal_metrics(rows):
    # Safety refusals have a different target from insufficient-evidence refusals.
    eligible = [r for r in rows if isinstance(r.get('answerable'), bool)
                and not r.get('expected_safety') and not r['service_failure']]
    tp = fp = fn = tn = 0
    for row in eligible:
        refused = row['answer_type'] in ('low_confidence', 'safety_refusal')
        if not row['answerable'] and refused:
            tp += 1
        elif row['answerable'] and refused:
            fp += 1
        elif not row['answerable']:
            fn += 1
        else:
            tn += 1
    ratio = lambda a, b: a / b if b else None
    return {'evaluated_count': len(eligible), 'excluded_count': len(rows) - len(eligible),
            'true_refusals': tp, 'false_refusals': fp, 'unsupported_answers': fn, 'correct_answers_decision': tn,
            'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'false_refusal_rate': ratio(fp, fp + tn), 'unsupported_answer_rate': ratio(fn, fn + tp),
            'scope': 'explicit answerability labels; excludes expected safety cases and service failures'}


def evaluate_generation(rag_system, eval_path, details_path=None):
    rows = []
    for sample in load_jsonl(eval_path):
        result = rag_system.answer(sample['question'])
        generation = result.get('generation') or {}
        generated = result.get('answer_type') == 'generated' and generation.get('success') is True
        failure = result.get('answer_type') == 'service_unavailable'
        rows.append({'question': sample['question'], 'answer_type': result.get('answer_type'),
                     'answerable': sample.get('answerable'),
                     'expected_safety': bool(set(sample.get('risk_types', []) + [sample.get('risk_type')])
                                             & {'battery_safety', 'dangerous_repair'}),
                     'generated': generated, 'service_failure': failure,
                     'format': bool(generated and check_format(result['answer'])),
                     'points': check_expected_points(result['answer'], sample.get('expected_points', [])) if generated else None,
                     'need_human_correct': result['need_human'] == sample.get('need_human', False),
                     'result': result})
    count = len(rows)
    generated_rows = [r for r in rows if r['generated']]
    points = [r['points'] for r in generated_rows if r['points'] is not None]
    mean = lambda values: sum(values) / len(values) if values else None
    if details_path:
        path = Path(details_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return {
        'sample_count': count, 'generated_count': len(generated_rows),
        'request_success_rate': mean([not r['service_failure'] for r in rows]),
        'generation_failure_rate': mean([r['service_failure'] for r in rows]),
        'answer_coverage': len(generated_rows) / count if count else None,
        'format_follow_rate': mean([r['format'] for r in generated_rows]),
        'answer_point_coverage': mean(points),
        'format_follow_rate_all_requests': mean([r['format'] for r in rows]),
        'need_human_accuracy': mean([r['need_human_correct'] for r in rows]),
        'refusal': refusal_metrics(rows),
        'quality_scope': 'generated responses only; substring coverage is a lexical proxy, not semantic correctness',
    }
