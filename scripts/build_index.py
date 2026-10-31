import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.dense_retriever import DenseRetriever
from app.config import CHUNKS_PATH, DENSE_MODEL_NAME, FAISS_INDEX_PATH, FAISS_META_PATH


def main():
    DenseRetriever(
        chunks_path=CHUNKS_PATH,
        model_name=DENSE_MODEL_NAME,
        index_path=FAISS_INDEX_PATH,
        meta_path=FAISS_META_PATH
    )
    print("Dense index built successfully.")


if __name__ == "__main__":
    main()
