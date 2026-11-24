# tests/test_hybrid_engine.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import MarkdownStructureChunker
from src.retrieval.hybrid_engine import HybridRetrievalEngine

corpus_doc = """# Operational Runbook

## Section A: Ingress Health Checks
The health endpoint is exposed on internal port 8080 at /healthz.
If HTTP 503 is returned for 3 consecutive probes, trigger automatic pod recycling.

## Section B: Cluster Eviction Code ERR-9022
ERR-9022 indicates Redis cluster partition failure across subnets.
Check node reachability on port 6379 and notify DevOps on-call immediately.

## Section C: General Cache Partitioning
When distributed memory caches become desynchronized or drop heartbeat signals,
the service falls back to direct relational database reads.
"""

def test_hybrid():
    chunker = MarkdownStructureChunker(target_chunk_size=100)
    chunks = chunker.chunk_document(corpus_doc, "runbook.md")

    engine = HybridRetrievalEngine(k_constant=60)
    print("Indexing chunks into Hybrid Engine (Dense + BM25)...")
    engine.index_chunks(chunks)

    # Test query combining exact keyword 'ERR-9022' with semantic phrase 'cache heartbeat problem'
    query = "What happens during cache heartbeat failure and ERR-9022?"
    print(f"\nEvaluating Hybrid Query: '{query}'\n")

    results = engine.search(query, top_k=3)

    for doc in results:
        metrics = doc["retrieval_metrics"]
        print(f"Rank {doc['rrf_rank']} | RRF Score: {doc['rrf_score']}")
        print(f"Section: {doc['section_path']}")
        print(f"Dense Rank: {metrics['dense_rank']} (Score: {metrics['dense_score']}) | "
              f"Sparse Rank: {metrics['sparse_rank']} (Score: {metrics['sparse_score']})")
        print(f"Content: {doc['content']}\n" + "-"*60)

if __name__ == "__main__":
    test_hybrid()