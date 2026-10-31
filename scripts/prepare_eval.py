"""Validate annotations and split independent question groups, never individual rewrites."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower() if c.isalnum())


def validate(samples, chunk_ids):
    errors, warnings = [], []
    ids, questions, groups = set(), {}, {}
    for row in samples:
        qid = row.get('query_id')
        if not qid or qid in ids:
            errors.append(f'Missing/duplicate query_id: {qid}')
        ids.add(qid)
        if not row.get('group_id') or not row.get('category') or not row.get('source'):
            errors.append(f'{qid}: group/category/source required')
        if not isinstance(row.get('answerable'), bool):
            errors.append(f'{qid}: explicit boolean answerable required')
        gold = row.get('gold_chunk_ids', [])
        if row.get('answerable') and not gold:
            errors.append(f'{qid}: answerable query needs gold')
        if not row.get('answerable') and gold:
            errors.append(f'{qid}: unanswerable query must not have a fabricated gold')
        if set(gold) - chunk_ids:
            errors.append(f'{qid}: unknown gold IDs {sorted(set(gold) - chunk_ids)}')
        question = normalize(row.get('question', ''))
        if not question or question in questions:
            errors.append(f'{qid}: empty/duplicate normalized question')
        questions[question] = qid
        split = row.get('split')
        if split is not None and split not in ('train', 'dev', 'test'):
            errors.append(f'{qid}: invalid split')
        group = row.get('group_id')
        if group in groups and split != groups[group]:
            errors.append(f'{qid}: group crosses splits')
        groups[group] = split
    for i, a in enumerate(samples):
        for b in samples[i + 1:]:
            if a.get('group_id') != b.get('group_id') and SequenceMatcher(
                None, normalize(a.get('question', '')), normalize(b.get('question', ''))).ratio() >= 0.85:
                warnings.append({'query_ids': [a['query_id'], b['query_id']],
                                 'reason': 'near duplicate across groups; review required'})
    return errors, warnings


def assign_splits(samples, seed=20261031):
    grouped = defaultdict(list)
    for row in samples:
        grouped[row['group_id']].append(row)
    strata = defaultdict(list)
    for group, rows in grouped.items():
        if len({r['category'] for r in rows}) != 1:
            raise ValueError(f'Group {group} spans categories; review annotation.')
        strata[rows[0]['category']].append(group)
    rng = random.Random(seed)
    for category in sorted(strata):
        groups = sorted(strata[category])
        rng.shuffle(groups)
        # Small strata cannot guarantee all splits; manifest exposes the actual counts.
        n_dev = max(1, round(len(groups) * .2)) if len(groups) >= 3 else 0
        n_test = n_dev
        for i, group in enumerate(groups):
            split = 'dev' if i < n_dev else 'test' if i < n_dev + n_test else 'train'
            for row in grouped[group]:
                row['split'] = split
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--reviewed', action='store_true', help='Assert annotations and near-duplicate warnings were manually reviewed.')
    args = parser.parse_args()
    source = Path(args.input)
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise SystemExit('Input is empty.')
    chunks = {json.loads(line)['chunk_id'] for line in (ROOT / 'data/processed/chunks.jsonl').read_text(encoding='utf-8').splitlines()}
    errors, warnings = validate(rows, chunks)
    if errors:
        raise SystemExit('\n'.join(errors))
    provided = [r.get('split') is not None for r in rows]
    if any(provided) and not all(provided):
        raise SystemExit('Provide all split assignments or none.')
    if not any(provided):
        rows = assign_splits(rows, args.seed)
    output = Path(args.output)
    if output.exists():
        raise SystemExit('Output already exists. Use a new dataset version.')
    output.mkdir(parents=True)
    files = {}
    for split in ('train', 'dev', 'test'):
        data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows if r['split'] == split)
        path = output / f'{split}.jsonl'
        path.write_text(data, encoding='utf-8')
        files[path.name] = hashlib.sha256(data.encode('utf-8')).hexdigest()
    manifest = {
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'seed': args.seed,
        'status': 'reviewed' if args.reviewed else 'draft_requires_annotation_review',
        'sample_counts': dict(Counter(r['split'] for r in rows)),
        'group_counts': dict(Counter(v[0]['split'] for v in _groups(rows).values())),
        'category_counts': dict(Counter(r['category'] for r in rows)),
        'near_duplicate_warnings': warnings, 'files_sha256': files,
        'note': 'Known-knowledge-base new-query evaluation; dataset size does not establish label quality.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    return groups


if __name__ == '__main__':
    main()
