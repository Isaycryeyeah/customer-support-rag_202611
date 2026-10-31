import os
import json
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')
_deployment_path = BASE_DIR / 'configs' / 'deployment.json'
_deployment = json.loads(_deployment_path.read_text(encoding='utf-8')) if _deployment_path.exists() else {}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

CHUNKS_PATH = str(BASE_DIR / "data" / "processed" / "chunks.jsonl")
FAISS_INDEX_PATH = str(BASE_DIR / "data" / "processed" / "faiss.index")
FAISS_META_PATH = str(BASE_DIR / "data" / "processed" / "faiss_meta.pkl")

DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", _deployment.get('reranker_model', "BAAI/bge-reranker-base"))
RERANKER_DTYPE = os.getenv('RERANKER_DTYPE', _deployment.get('reranker_dtype', 'float32'))

BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

DB_PATH = str(BASE_DIR / "data" / "app.db")

LOW_CONFIDENCE_THRESHOLD = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', str(_deployment.get('threshold', 0.1))))
CANDIDATE_K = int(os.getenv('CANDIDATE_K', '50'))
RERANK_K = int(os.getenv('RERANK_K', str(_deployment.get('rerank_k', 20))))
CONTEXT_K = int(os.getenv('CONTEXT_K', '5'))
