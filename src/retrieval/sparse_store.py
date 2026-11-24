# src/retrieval/sparse_store.py

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.chunker import DocumentChunk


class SparseBM25Store:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.metadata_store: List[Dict[str, Any]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenizes text by lowercasing and extracting alphanumeric tokens,
        preserving identifiers like 'v1.4', 'emp-102', or 'err_502'.
        """
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Tokenizes chunk content and builds the BM25 inverted index."""
        if not chunks:
            return

        corpus = []
        for chunk in chunks:
            # Combine section path with content for keyword indexing
            full_text = f"{chunk.section_path} {chunk.content}"
            tokenized = self._tokenize(full_text)
            corpus.append(tokenized)

            self.metadata_store.append({
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "section_path": chunk.section_path,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata
            })

        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Searches BM25 corpus and returns metadata with BM25 scores."""
        if not self.bm25 or not self.metadata_store:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each chunk metadata with its BM25 score
        scored_results = [
            (self.metadata_store[idx], float(score))
            for idx, score in enumerate(scores)
            if score > 0.0  # Filter out documents with zero keyword overlap
        ]

        # Sort descending by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def save(self, directory: Path = settings.DATA_PROCESSED_DIR):
        """Persists the BM25 model and metadata store to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "sparse_bm25.pkl"
        metadata_path = directory / "sparse_metadata.json"

        with open(index_path, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2)

    def load(self, directory: Path = settings.DATA_PROCESSED_DIR):
        """Loads BM25 model and metadata store from disk."""
        index_path = directory / "sparse_bm25.pkl"
        metadata_path = directory / "sparse_metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Persisted BM25 index or metadata files not found.")

        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)