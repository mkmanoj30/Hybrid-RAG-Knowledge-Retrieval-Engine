# scripts/query_pipeline.py

import json
import httpx

# =============================================================================
# QUICK TEST INPUTS (Paste your test data here, or leave empty to be prompted)
# =============================================================================
CONTEXT_TEXT = """
# Developer Workstation Provisioning Standards

## Section 1: Memory & Compute Allocations
All new engineering hires receive standardized laptop hardware based on their assigned track.
Platform and DevOps engineers are allocated machines with 32 GB of unified memory.
Quality Assurance and Automation engineers are provisioned with 16 GB of unified memory.

## Section 2: Storage Retention
Local temporary build artifacts stored in `/tmp/builds` are automatically purged every 24 hours.
Project repositories cloned in the primary workspace directory are exempted from automatic cleanup routines.
"""

QUERY_TEXT = "How much unified memory is provisioned for Quality Assurance and Automation engineers?"
# Service Endpoints
RAG_INGEST_URL = "http://127.0.0.1:8001/api/v1/ingest"
RAG_RETRIEVE_URL = "http://127.0.0.1:8001/api/v1/retrieve"
GATEWAY_URL = "http://127.0.0.1:8000/api/v1/generate"


def run_test():
    global CONTEXT_TEXT, QUERY_TEXT
    
    # 1. Fallback to interactive terminal input if variables are empty
    if not CONTEXT_TEXT.strip():
        print("Enter context text (type END on a new line when done):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        CONTEXT_TEXT = "\n".join(lines)

    if not QUERY_TEXT.strip():
        QUERY_TEXT = input("\nEnter query: ")

    print("\n" + "=" * 60)
    print(f"QUERY SUBMITTED: {QUERY_TEXT}")
    print("=" * 60)

    with httpx.Client(timeout=None) as client:
        client.post("http://127.0.0.1:8001/api/v1/reset")
        # 2. Ingest context on-the-fly into Project 2
        print("\n[1/3] Ingesting context into Project 2 (:8001)...")
        ingest_res = client.post(
            RAG_INGEST_URL,
            json={"document_name": "ad_hoc_test.md", "content": CONTEXT_TEXT}
        )
        ingest_res.raise_for_status()
        print(f"      Ingested {ingest_res.json()['chunks_indexed']} chunk(s).")

        # 3. Retrieve relevant chunks via RRF
        print(f"[2/3] Retrieving matching context via Hybrid RRF for: '{QUERY_TEXT}'...")
        ret_res = client.post(
            RAG_RETRIEVE_URL,
            json={"query": QUERY_TEXT, "top_k": 4}
        )
        ret_res.raise_for_status()
        retrieved_chunks = [c["content"] for c in ret_res.json().get("results", [])]
        print(f"      Retrieved {len(retrieved_chunks)} relevant chunk(s).")

        # 4. Generate & evaluate output via Project 1 Gateway
        print("[3/3] Sending query + retrieved chunks to Gateway (:8000)...")
        gw_res = client.post(
            GATEWAY_URL,
            json={"query": QUERY_TEXT, "retrieval_context": retrieved_chunks}
        )
        gw_res.raise_for_status()
        ret_res = client.post(
            RAG_RETRIEVE_URL,
            json={"query": QUERY_TEXT, "top_k": 4}
        )
        ret_res.raise_for_status()
        raw_results = ret_res.json().get("results", [])
        retrieved_chunks = [c["content"] for c in raw_results]

        print(f"\n--- RETRIEVED CHUNKS ({len(retrieved_chunks)}) ---")
        for idx, chunk in enumerate(retrieved_chunks, 1):
            print(f"[Chunk {idx}]\n{chunk}\n")
        print("-----------------------------------")
    
    # Output formatted response
    print("\n" + "=" * 60)
    print(f"INPUT QUERY: {QUERY_TEXT}")
    print("=" * 60)
    print("GATEWAY RESPONSE:")
    print("=" * 60)
    print(json.dumps(gw_res.json(), indent=2))
    

if __name__ == "__main__":
    run_test()