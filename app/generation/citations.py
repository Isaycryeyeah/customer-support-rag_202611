"""Keep passage-level anchors, emit each source document only once."""
import re


def group_sources(contexts, cited_ids=None):
    groups = {}
    for index, context in enumerate(contexts, 1):
        if cited_ids is not None and index not in cited_ids:
            continue
        key = context.get('source_path') or context.get('doc_id') or context.get('title') or str(index)
        if key not in groups:
            groups[key] = {'title': context.get('title', '参考文档'),
                           'source_path': context.get('source_path'), 'doc_id': context.get('doc_id'),
                           'cited': cited_ids is not None, 'passages': []}
        groups[key]['passages'].append({'passage_id': index, 'chunk_id': context.get('chunk_id'),
                                        'section': context.get('section', '未标注章节')})
    return list(groups.values())


def format_citations(answer, contexts):
    # Remove only a standalone source-list section, never an inline discussion of a source.
    body = re.split(r'(?m)^\s*(?:#{1,6}\s*)?\*{0,2}引用来源\*{0,2}\s*[:：]?\s*\*{0,2}\s*$',
                    answer, maxsplit=1)[0].rstrip()
    mentioned = {int(i) for i in re.findall(r'资料\s*(\d+)', body)}
    valid = mentioned & set(range(1, len(contexts) + 1))
    invalid = sorted(mentioned - valid)
    sources = group_sources(contexts, valid if valid else None)
    if not sources:
        return body, sources, {'status': 'missing', 'invalid_passage_ids': invalid}
    title = '引用来源' if valid else '检索参考资料（正文未标明具体引用）'
    lines = []
    for source in sources:
        anchors = '、'.join('资料' + str(p['passage_id']) for p in source['passages'])
        sections = '；'.join(dict.fromkeys(p['section'] for p in source['passages']))
        lines.append(f"- {source['title']}（{anchors}；章节：{sections}）")
    status = 'invalid' if invalid else 'valid' if valid else 'missing'
    return body + '\n\n' + title + '：\n' + '\n'.join(lines), sources, {'status': status, 'invalid_passage_ids': invalid}
