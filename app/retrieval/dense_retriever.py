import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np


class DenseRetriever:
    def __init__(
        self,
        chunks_path: str,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_path: str = "data/processed/faiss.index",
        meta_path: str = "data/processed/faiss_meta.pkl",
        *, model=None
    ):
        if model is None:
            from sentence_transformers import SentenceTransformer
            revision = '7999e1d3359715c523056ef9478215996d62a620' if model_name == 'BAAI/bge-small-zh-v1.5' else None
            model = SentenceTransformer(model_name, revision=revision)
        self.model = model
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.chunks_path = chunks_path
        self.manifest_path = self.index_path.with_suffix('.manifest.json')
        self.chunks = self._load_chunks(chunks_path)
        self.manifest = {
            'chunks_sha256': hashlib.sha256(Path(chunks_path).read_bytes()).hexdigest(),
            'model_name': model_name,
            'dimension': int(self.model.get_sentence_embedding_dimension()),
            'chunk_count': len(self.chunks),
        }
        # Revision is available on HF model configs; include it when present.
        try:
            self.manifest['model_revision'] = self.model[0].auto_model.config._commit_hash
        except (AttributeError, KeyError, TypeError):
            self.manifest['model_revision'] = None

        saved_manifest = json.loads(self.manifest_path.read_text(encoding='utf-8')) if self.manifest_path.exists() else None
        if self.index_path.exists() and self.meta_path.exists() and saved_manifest == self.manifest:
            self.index = faiss.read_index(str(self.index_path))
            with self.meta_path.open("rb") as f:
                saved_chunks = pickle.load(f)
            if saved_chunks != self.chunks or self.index.ntotal != len(self.chunks) or self.index.d != self.manifest['dimension']:
                raise ValueError('Index files are inconsistent. Rebuild the index.')
        else:
            self.index = self._build_index(self.chunks)
            self._save_index()

    def _load_chunks(self, path: str) -> List[Dict]:
        with Path(path).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _build_index(self, chunks: List[Dict]):
        if not chunks:
            return faiss.IndexFlatIP(self.manifest['dimension'])
        texts = [item["content"] for item in chunks]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def _save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("wb") as f:
            pickle.dump(self.chunks, f)
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.chunks or top_k <= 0:
            return []
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        query_vec = np.asarray(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            item = dict(self.chunks[idx])
            item["dense_score"] = float(score)
            results.append(item)

        return results
