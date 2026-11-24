# src/retrieval/hybrid_engine.py

from typing import List, Dict, Any
from src.config import settings
from src.ingestion.chunker import DocumentChunk
from src.retrieval.dense_store import DenseVectorStore
from src.retrieval.sparse_store import SparseBM25Store


class HybridRetrievalEngine:
    """
    Orchestrates dense semantic search and sparse BM25 keyword search,
    combining ranked outputs using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, k_constant: int = settings.RRF_K):
        self.k_constant = k_constant
        self.dense_store = DenseVectorStore()
        self.sparse_store = SparseBM25Store()

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Populates both dense FAISS and sparse BM25 indices simultaneously."""
        if not chunks:
            return
        self.dense_store.add_chunks(chunks)
        self.sparse_store.add_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
        candidate_pool_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel dense and sparse retrieval, calculates RRF scores,
        and returns deduplicated, fused rankings.
        """
        dense_hits = self.dense_store.search(query, top_k=candidate_pool_size)
        sparse_hits = self.sparse_store.search(query, top_k=candidate_pool_size)

        rrf_scores: Dict[str, float] = {}
        document_registry: Dict[str, Dict[str, Any]] = {}
        rank_tracking: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Rankings
        for rank, (doc, sim_score) in enumerate(dense_hits, start=1):
            chunk_id = doc["chunk_id"]
            document_registry[chunk_id] = doc
            rrf_contrib = 1.0 / (self.k_constant + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contrib
            rank_tracking[chunk_id] = {
                "dense_rank": rank,
                "dense_score": round(sim_score, 4),
                "sparse_rank": None,
                "sparse_score": None
            }

        # 2. Process Sparse BM25 Rankings
        for rank, (doc, bm25_score) in enumerate(sparse_hits, start=1):
            chunk_id = doc["chunk_id"]
            document_registry[chunk_id] = doc
            rrf_contrib = 1.0 / (self.k_constant + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contrib
            
            if chunk_id not in rank_tracking:
                rank_tracking[chunk_id] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank,
                    "sparse_score": round(bm25_score, 4)
                }
            else:
                rank_tracking[chunk_id]["sparse_rank"] = rank
                rank_tracking[chunk_id]["sparse_score"] = round(bm25_score, 4)

        # 3. Sort by aggregated RRF score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        # 4. Construct final payload
        fused_results = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            chunk_data = document_registry[cid].copy()
            chunk_data["rrf_rank"] = rank
            chunk_data["rrf_score"] = round(rrf_scores[cid], 6)
            chunk_data["retrieval_metrics"] = rank_tracking[cid]
            fused_results.append(chunk_data)

        return fused_results

    def save(self):
        """Persists both store states to the processed directory."""
        self.dense_store.save()
        self.sparse_store.save()

    def load(self):
        """Loads both store states from the processed directory."""
        self.dense_store.load()
        self.sparse_store.load()