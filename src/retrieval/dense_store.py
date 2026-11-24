# src/retrieval/dense_store.py

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import DocumentChunk


class DenseVectorStore:
    def __init__(self, model_name: str = settings.DENSE_MODEL_NAME):
        self.model = SentenceTransformer(model_name, device=settings.DEVICE)
        self.embedding_dim = self.model.get_embedding_dimension()
        
        # IndexFlatIP computes exact inner products (cosine similarity when normalized)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata_store: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Embeds document chunks and adds them to the FAISS index."""
        if not chunks:
            return

        texts_to_embed = [
            f"{c.section_path}\n{c.content}" for c in chunks
        ]

        embeddings = self.model.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Crucial for unit sphere cosine distance via IndexFlatIP
        ).astype(np.float32)

        self.index.add(embeddings)
        
        for chunk in chunks:
            self.metadata_store.append({
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "section_path": chunk.section_path,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata
            })

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Queries the dense FAISS index and returns matching metadata + similarity scores."""
        if self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append((self.metadata_store[idx], float(score)))

        return results

    def save(self, directory: Path = settings.DATA_PROCESSED_DIR):
        """Persists the FAISS index and metadata store to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "dense_faiss.index"
        metadata_path = directory / "dense_metadata.json"

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2)

    def load(self, directory: Path = settings.DATA_PROCESSED_DIR):
        """Loads the FAISS index and metadata store from disk."""
        index_path = directory / "dense_faiss.index"
        metadata_path = directory / "dense_metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Persisted FAISS index or metadata files not found.")

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)