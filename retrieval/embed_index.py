"""
RegLens — embedding + vector index.

Embeds all chunks with a HuggingFace sentence-transformer and builds
a FAISS index for retrieval. Model choice:

  - default: sentence-transformers/all-MiniLM-L6-v2 (fast, 384-dim, great baseline)
  - legal:   nlpaueb/legal-bert-base-uncased via mean pooling (domain-adapted)

Usage:
    python -m retrieval.embed_index                 # build index with default model
    python -m retrieval.embed_index --model legal   # use LegalBERT
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROC_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

MODELS = {
    "default": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "legal": "nlpaueb/legal-bert-base-uncased",
}


def load_chunks() -> list[dict]:
    chunks_path = PROC_DIR / "chunks.jsonl"
    with chunks_path.open() as fh:
        return [json.loads(line) for line in fh]


def embed_chunks(chunks: list[dict], model_key: str = "default", batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_name = MODELS[model_key]
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (batch={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors → inner product = cosine sim
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact inner-product search; fine at this scale
    index.add(embeddings)
    return index


def run(model_key: str) -> None:
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    embeddings = embed_chunks(chunks, model_key=model_key)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(INDEX_DIR / f"reglens_{model_key}.faiss"))
    (INDEX_DIR / f"chunks_{model_key}.json").write_text(json.dumps(chunks))
    meta = {"model": MODELS[model_key], "dim": int(embeddings.shape[1]), "n_chunks": len(chunks)}
    (INDEX_DIR / f"meta_{model_key}.json").write_text(json.dumps(meta, indent=2))
    print(f"Index built: {len(chunks)} chunks, dim={embeddings.shape[1]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="default", choices=list(MODELS))
    args = p.parse_args()
    run(args.model)
