"""
RegLens — FastAPI backend.

Serves the RAG engine over HTTP for the React frontend.

Usage:
    uvicorn api:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    from retrieval.query import RegLensEngine
    engine = RegLensEngine()
    yield


app = FastAPI(title="RegLens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    k: int = 6


@app.post("/query")
def query(req: QueryRequest):
    return engine.answer(req.question, k=req.k)


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    """Retrieval-only endpoint — useful for debugging and the source browser."""
    return {"results": engine.retrieve(req.question, k=req.k)}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(engine.chunks) if engine else 0}
