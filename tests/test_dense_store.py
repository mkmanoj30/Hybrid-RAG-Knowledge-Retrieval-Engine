# tests/test_dense_store.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import MarkdownStructureChunker
from src.retrieval.dense_store import DenseVectorStore

sample_doc = """# Security Policy

## Subsection A: VPC Configuration
All production databases must be placed in a private subnet with no public IP allocation.
Direct SSH ingress is strictly blocked by AWS Security Groups.

## Subsection B: Backup Lifecycle
Database snapshots are taken nightly at 02:00 UTC.
Cold archives migrate to S3 Glacier after 180 days.
"""

def test_dense():
    chunker = MarkdownStructureChunker(target_chunk_size=100)
    chunks = chunker.chunk_document(sample_doc, "security_policy.md")

    store = DenseVectorStore()
    print("Embedding chunks and building FAISS index...")
    store.add_chunks(chunks)

    query = "How are databases protected from outside internet access?"
    print(f"\nQuery: '{query}'")
    hits = store.search(query, top_k=2)

    for rank, (meta, score) in enumerate(hits, 1):
        print(f"\n[Rank {rank}] Cosine Similarity: {score:.4f}")
        print(f"Section: {meta['section_path']}")
        print(f"Content: {meta['content']}")

if __name__ == "__main__":
    test_dense()