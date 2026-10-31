"""Verify frozen data bytes and common candidate pools without model/API calls."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise SystemExit('Use a new verification report path')
    checks = []
    manifest = read('data/eval/v1/manifest.json')
    partitions = {}
    for name, expected in manifest['files_sha256'].items():
        path = Path('data/eval/v1') / name
        assert sha(path) == expected, f'Dataset hash mismatch: {path}'
        partitions[name] = {r['group_id'] for r in rows(path)}
        checks.append({'file': str(path), 'sha256': expected})
    for index, name in enumerate(partitions):
        for other in list(partitions)[index + 1:]:
            assert not partitions[name] & partitions[other], f'Group leakage: {name}, {other}'
    for version in ('v1', 'v2'):
        path = Path('data/training') / version
        assert sha(path / 'queries.jsonl') == read(path / 'manifest.json')['queries_sha256']
        checks.append({'file': str(path / 'queries.jsonl'), 'sha256': sha(path / 'queries.jsonl')})
    test3 = Path('data/eval/clean-study-v1/test_v3.jsonl')
    assert sha(test3) == read(test3.parent / 'manifest.json')['test_sha256']
    assert sha(test3) == read('configs/experiments/clean-study-decision.json')['test_sha256']
    checks.append({'file': str(test3), 'sha256': sha(test3)})
    comparisons = []
    for prefix, names, dataset in [
        ('m2-test-', ('pretrained', 'random', 'hard', 'hard-60'), Path('data/eval/v1/test_v2.jsonl')),
        ('clean-study-test-', ('pretrained', 'raw', 'clean'), test3),
    ]:
        baseline = None
        expected_ids = {r['query_id'] for r in rows(dataset)}
        for name in names:
            folder = Path('data/eval/runs') / (prefix + name)
            report = read(folder / 'report.json')
            assert report['input_sha256'] == sha(dataset), f'Wrong test input: {folder}'
            result_rows = rows(folder / 'rows.jsonl')
            assert {r['query_id'] for r in result_rows} == expected_ids
            pool = {r['query_id']: [c['chunk_id'] for c in r['candidates'][:20]] for r in result_rows}
            if baseline is None:
                baseline = pool
            assert pool == baseline, f'Candidate mismatch: {folder}'
        comparisons.append({'experiment': prefix, 'models': list(names),
                            'queries': len(expected_ids), 'same_ordered_top20_candidates': True})
    result = {'passed': True, 'checks': checks, 'group_partitions_disjoint': True,
              'candidate_comparisons': comparisons,
              'scope': 'File integrity and controlled inputs; not human label validation or an unseen rerun.'}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'dataset_files': len(checks), 'comparisons': len(comparisons)}))


if __name__ == '__main__':
    main()
