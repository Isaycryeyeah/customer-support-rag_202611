import time
from uuid import uuid4

from app.generation.llm_client import FALLBACK_ANSWER
from app.generation.prompt import build_rag_prompt
from app.safety.risk_detector import detect_risk
from app.safety.refusal import LOW_CONFIDENCE_ANSWER, build_high_risk_refusal
from app.ticket.classifier import classify_ticket
from app.ticket.ticket_builder import build_ticket


class CustomerSupportRAG:
    def __init__(self, hybrid_retriever, reranker, llm_client, *,
                 threshold=0.1, candidate_k=50, rerank_k=20, context_k=5):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = llm_client
        self.threshold = threshold
        if min(candidate_k, rerank_k, context_k) < 1:
            raise ValueError('Retrieval counts must be positive.')
        self.candidate_k, self.rerank_k, self.context_k = candidate_k, rerank_k, context_k

    def answer(self, query):
        query = query.strip()
        if not query:
            raise ValueError('question must not be blank')
        started = time.perf_counter()
        timings = {'risk_ms': 0.0, 'retrieval_ms': 0.0, 'rerank_ms': 0.0, 'llm_ms': 0.0}
        mark = time.perf_counter()
        risk_info = detect_risk(query)
        timings['risk_ms'] = (time.perf_counter() - mark) * 1000
        contexts, candidates, generation = [], [], None
        low_confidence = False
        handoff_reason = None
        if risk_info['risk_level'] == 'high':
            answer_type = 'safety_refusal'
            answer = build_high_risk_refusal(risk_info['risk_types'])
            handoff_reason = 'safety_risk'
        else:
            mark = time.perf_counter()
            candidates = self.hybrid_retriever.search(query, top_k=self.rerank_k, candidate_k=self.candidate_k)
            timings['retrieval_ms'] = (time.perf_counter() - mark) * 1000
            mark = time.perf_counter()
            contexts = self.reranker.rerank(query, candidates, top_k=self.context_k) if candidates else []
            timings['rerank_ms'] = (time.perf_counter() - mark) * 1000
            low_confidence = not contexts or contexts[0].get('rerank_score', 0.0) < self.threshold
            if low_confidence:
                answer_type, answer = 'low_confidence', LOW_CONFIDENCE_ANSWER
                handoff_reason = 'insufficient_evidence'
            else:
                generation = self.llm.generate_safe(build_rag_prompt(query, contexts, risk_info))
                timings['llm_ms'] = generation.elapsed_ms
                if generation.success:
                    answer_type, answer = 'generated', generation.text
                else:
                    answer_type, answer = 'service_unavailable', FALLBACK_ANSWER
                    handoff_reason = 'service_unavailable'
        if handoff_reason is None and risk_info['need_human']:
            handoff_reason = 'policy_review'
        need_human = handoff_reason is not None
        ticket = build_ticket(query, answer, risk_info, classify_ticket(query, risk_info),
                              need_human=need_human, handoff_reason=handoff_reason,
                              answer_type=answer_type) if need_human else None
        timings['total_ms'] = (time.perf_counter() - started) * 1000
        return {
            'request_id': str(uuid4()), 'question': query, 'answer': answer,
            'answer_type': answer_type, 'risk_info': risk_info,
            'contexts': contexts, 'low_confidence': low_confidence,
            'need_human': need_human, 'handoff_reason': handoff_reason, 'ticket': ticket,
            'generation': generation.to_dict() if generation else None,
            'timings': timings,
            'retrieval_trace': [{k: c[k] for k in ('chunk_id', 'bm25_score', 'dense_score', 'hybrid_score') if k in c}
                                for c in candidates],
            'config': {'threshold': self.threshold, 'candidate_k': self.candidate_k,
                       'rerank_k': self.rerank_k, 'context_k': self.context_k},
        }
