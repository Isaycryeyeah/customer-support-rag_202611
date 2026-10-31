from scripts.prepare_eval import assign_splits, validate


def sample(i, group=None):
    return {'query_id': f'q{i}', 'group_id': group or f'g{i}', 'question': f'number {i}',
            'category': 'warranty', 'source': 'manual', 'answerable': True, 'gold_chunk_ids': ['a']}


def test_group_split_is_deterministic_and_keeps_rewrites_together():
    rows = [sample(i) for i in range(10)] + [sample(11, 'g1')]
    first = assign_splits([dict(r) for r in rows])
    second = assign_splits([dict(r) for r in rows])
    assert first == second
    assert {r['split'] for r in first if r['group_id'] == 'g1'} in ({'train'}, {'dev'}, {'test'})
    assert {r['split'] for r in first} == {'train', 'dev', 'test'}


def test_leakage_invalid_gold_and_duplicate_question_rejected():
    a, b = sample(1), sample(2, 'g1')
    a['split'], b['split'] = 'train', 'test'
    b['gold_chunk_ids'] = ['nonexistent']
    b['question'] = 'NUMBER 1!'
    errors, _ = validate([a, b], {'a'})
    assert any('group crosses' in e for e in errors)
    assert any('unknown gold' in e for e in errors)
    assert any('duplicate normalized' in e for e in errors)
