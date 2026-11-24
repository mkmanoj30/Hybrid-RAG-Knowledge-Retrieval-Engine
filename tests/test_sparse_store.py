# tests/test_sparse_store.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import MarkdownStructureChunker
from src.retrieval.sparse_store import SparseBM25Store

sample_doc = """# Incident Protocols

## Section 1: Error Code ERR-8041
When receiving ERR-8041, immediately restart the ingress proxy container.
Do not purge persistent volume mounts during this phase.

## Section 2: Error Code ERR-9022
ERR-9022 indicates Redis cluster partition failure.
Check node reachability across port 6379 and notify DevOps on-call.
"""

def test_sparse():
    chunker = MarkdownStructureChunker(target_chunk_size=100)
    chunks = chunker.chunk_document(sample_doc, "incident_protocols.md")

    store = SparseBM25Store()
    print("Building BM25 keyword index...")
    store.add_chunks(chunks)

    query = "How to resolve ERR-9022 failure?"
    print(f"\nQuery: '{query}'")
    hits = store.search(query, top_k=2)

    for rank, (meta, score) in enumerate(hits, 1):
        print(f"\n[Rank {rank}] BM25 Score: {score:.4f}")
        print(f"Section: {meta['section_path']}")
        print(f"Content: {meta['content']}")

if __name__ == "__main__":
    test_sparse()