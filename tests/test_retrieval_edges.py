import json

import numpy as np

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker


class FakeEncoder:
    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] for _ in texts], dtype='float32')


def build(tmp_path, chunks):
    source = tmp_path / 'chunks.jsonl'
    source.write_text(''.join(json.dumps(c) + '\n' for c in chunks), encoding='utf-8')
    return DenseRetriever(str(source), 'fake', str(tmp_path / 'faiss.index'), str(tmp_path / 'meta.pkl'), model=FakeEncoder())


def test_dense_caps_k_and_rebuilds_changed_knowledge(tmp_path):
    r = build(tmp_path, [{'chunk_id': 'a', 'content': 'one'}])
    assert [c['chunk_id'] for c in r.search('q', 50)] == ['a']
    r = build(tmp_path, [{'chunk_id': 'b', 'content': 'two'}])
    assert [c['chunk_id'] for c in r.search('q', 50)] == ['b']
    assert r.search('q', 0) == []


def test_empty_indexes_and_reranker(tmp_path):
    r = build(tmp_path, [])
    assert r.search('q', 5) == []
    assert BM25Retriever(str(tmp_path / 'chunks.jsonl')).search('q') == []
    reranker = Reranker.__new__(Reranker)
    assert reranker.rerank('q', []) == []
