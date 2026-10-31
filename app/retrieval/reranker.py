from typing import List, Dict
from pathlib import Path
import json


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, dtype="float32"):
        from sentence_transformers import CrossEncoder
        path = Path(model_name)
        adapter = path / 'adapter.json'
        if adapter.is_file():
            import torch
            settings = json.loads(adapter.read_text(encoding='utf-8'))
            self.model = CrossEncoder(settings['base_model'], revision=settings.get('base_revision'),
                                      max_length=settings.get('max_length', 512))
            state = torch.load(path / 'adapter.pt', map_location='cpu', weights_only=True)
            if set(state) != set(settings['trainable_names']):
                raise ValueError('Adapter parameter set differs from its manifest.')
            unexpected = self.model.model.load_state_dict(state, strict=False).unexpected_keys
            if unexpected:
                raise ValueError(f'Adapter contains unexpected parameter names: {unexpected}')
        else:
            revision = '2cfc18c9415c912f9d8155881c133215df768a70' if model_name == 'BAAI/bge-reranker-base' else None
            self.model = CrossEncoder(model_name, revision=revision)
        if dtype == 'auto':
            dtype = 'float16' if self.model.device.type == 'cuda' else 'float32'
        if dtype not in ('float32', 'float16'):
            raise ValueError('Reranker dtype must be auto, float32 or float16.')
        if dtype == 'float16':
            if self.model.device.type != 'cuda':
                raise ValueError('float16 inference requires a CUDA device.')
            self.model.model.half()
        self.dtype = dtype

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        if not candidates or top_k <= 0:
            return []
        pairs = [(query, item["content"]) for item in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for item, score in zip(candidates, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked = sorted(
            reranked,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
