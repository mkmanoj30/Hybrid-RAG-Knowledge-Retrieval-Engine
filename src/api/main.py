# src/api/main.py

from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import shutil

from src.config import settings
from src.ingestion.chunker import MarkdownStructureChunker
from src.retrieval.hybrid_engine import HybridRetrievalEngine

# Global in-memory hybrid engine
engine: Optional[HybridRetrievalEngine] = None
chunker = MarkdownStructureChunker(
    target_chunk_size=settings.TARGET_CHUNK_SIZE,
    overlap=settings.CHUNK_OVERLAP
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = HybridRetrievalEngine(k_constant=settings.RRF_K)
    # Attempt to load warm indices from disk if available
    try:
        engine.load()
        print("Warm FAISS index and BM25 store loaded from disk.")
    except FileNotFoundError:
        print("No pre-existing indices found. Initialized blank hybrid engine.")
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)


# --- Request & Response Schemas ---

class IngestDocumentRequest(BaseModel):
    document_name: str = Field(..., example="cloud_storage_policy.md")
    content: str = Field(..., example="# Storage Tiering\n\nCold records archive after 90 days.")


class IngestResponse(BaseModel):
    status: str
    document_name: str
    chunks_indexed: int
    total_tokens_indexed: int


class RetrieveRequest(BaseModel):
    query: str = Field(..., example="What is the retention rule for inactive storage?")
    top_k: int = Field(default=settings.TOP_K_RETRIEVAL, ge=1, le=20)


class RankedChunk(BaseModel):
    chunk_id: str
    document_name: str
    section_path: str
    content: str
    token_count: int
    rrf_rank: int
    rrf_score: float
    retrieval_metrics: Dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    total_returned: int
    results: List[RankedChunk]


# --- Endpoints ---

@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "dense_model": settings.DENSE_MODEL_NAME
    }


@app.post("/api/v1/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(payload: IngestDocumentRequest):
    if not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document content cannot be empty."
        )

    # 1. Structure-aware chunking
    chunks = chunker.chunk_document(
        content=payload.content,
        document_name=payload.document_name
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text chunks could be extracted from payload."
        )

    # 2. Update hybrid stores and save to disk
    engine.index_chunks(chunks)
    engine.save()

    total_tokens = sum(c.token_count for c in chunks)
    return IngestResponse(
        status="success",
        document_name=payload.document_name,
        chunks_indexed=len(chunks),
        total_tokens_indexed=total_tokens
    )


@app.post("/api/v1/retrieve", response_model=RetrieveResponse, status_code=status.HTTP_200_OK)
async def retrieve_context(payload: RetrieveRequest):
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    fused_hits = engine.search(query=payload.query, top_k=payload.top_k)

    return RetrieveResponse(
        query=payload.query,
        total_returned=len(fused_hits),
        results=fused_hits
    )

@app.post("/api/v1/reset", status_code=status.HTTP_200_OK)
async def reset_indices():
    global engine
    # Re-initialize blank engine
    engine = HybridRetrievalEngine(k_constant=settings.RRF_K)
    
    # Wipe the disk files
    if settings.DATA_PROCESSED_DIR.exists():
        shutil.rmtree(settings.DATA_PROCESSED_DIR)
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
    return {"status": "cleared", "message": "Indices wiped successfully."}