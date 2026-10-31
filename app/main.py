import time
import hashlib
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import AskRequest, AskResponse, FeedbackRequest, FeedbackResponse
from app.main_pipeline import CustomerSupportRAG
from app.storage.db import init_db, save_query_log, save_feedback, query_log_exists
from app.config import (
    LLM_PROVIDER,
    CHUNKS_PATH,
    DENSE_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    RERANKER_MODEL_NAME,
    BM25_WEIGHT,
    LOW_CONFIDENCE_THRESHOLD, CANDIDATE_K, RERANK_K, CONTEXT_K, RERANKER_DTYPE,
)

app = FastAPI(title="中文 3C 售后客服助手")


rag_system: Optional[CustomerSupportRAG] = None
knowledge_version: Optional[str] = None
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup():
    global rag_system, knowledge_version

    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import Reranker
    from app.generation.llm_client import LLMClient
    from pathlib import Path

    init_db()

    bm25 = BM25Retriever(CHUNKS_PATH)
    dense = DenseRetriever(CHUNKS_PATH, DENSE_MODEL_NAME, FAISS_INDEX_PATH, FAISS_META_PATH)
    hybrid = HybridRetriever(bm25, dense, bm25_weight=BM25_WEIGHT)
    reranker = Reranker(RERANKER_MODEL_NAME, dtype=RERANKER_DTYPE)
    llm = LLMClient(provider=LLM_PROVIDER)

    rag_system = CustomerSupportRAG(
        hybrid_retriever=hybrid,
        reranker=reranker,
        llm_client=llm, threshold=LOW_CONFIDENCE_THRESHOLD,
        candidate_k=CANDIDATE_K, rerank_k=RERANK_K, context_k=CONTEXT_K,
    )
    knowledge_version = hashlib.sha256(Path(CHUNKS_PATH).read_bytes()).hexdigest()


@app.post("/ask", response_model=AskResponse, responses={503: {'model': AskResponse}})
def ask(request: AskRequest):
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty or whitespace only.")

    start = time.perf_counter()
    try:
        result = rag_system.answer(question)
    except Exception as error:
        logger.error('Pipeline failed: %s', type(error).__name__)
        raise HTTPException(status_code=500, detail='Unable to process this request.') from None
    latency_ms = (time.perf_counter() - start) * 1000

    log_id = save_query_log(
        question=result["question"],
        answer=result["answer"],
        risk_level=result["risk_info"]["risk_level"],
        need_human=result["need_human"],
        low_confidence=result["low_confidence"],
        latency_ms=latency_ms,
        request_id=result['request_id'], answer_type=result['answer_type'],
        handoff_reason=result['handoff_reason'],
        trace={
            'knowledge_version': knowledge_version,
            'timings': result['timings'], 'generation': result['generation'],
            'candidates': result['retrieval_trace'], 'config': result['config'],
            'contexts': [{k: c[k] for k in ('chunk_id', 'rerank_score') if k in c}
                         for c in result['contexts']],
            'model_config': {'dense': DENSE_MODEL_NAME, 'reranker': RERANKER_MODEL_NAME,
                             'bm25_weight': BM25_WEIGHT},
        },
    )

    result["log_id"] = log_id
    if result['answer_type'] == 'service_unavailable':
        return JSONResponse(status_code=503, content=AskResponse(**result).model_dump())
    return result


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    if not query_log_exists(request.log_id):
        raise HTTPException(status_code=404, detail=f"log_id {request.log_id} not found.")

    feedback_id = save_feedback(
        log_id=request.log_id,
        rating=request.rating,
        comment=request.comment
    )

    return {"feedback_id": feedback_id}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get('/ready')
def ready():
    if rag_system is None:
        return JSONResponse(status_code=503, content={'status': 'loading'})
    return {'status': 'ready', 'knowledge_version': knowledge_version}
