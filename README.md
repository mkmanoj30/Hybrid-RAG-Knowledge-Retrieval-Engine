# Production Hybrid RAG & Knowledge Retrieval Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A low-latency, structure-aware hybrid retrieval microservice combining dense semantic vector search (**FAISS** + `all-MiniLM-L6-v2`) with sparse lexical keyword matching (**BM25**), unified through **Reciprocal Rank Fusion (RRF)** to deliver grounded context chunks for enterprise LLM workflows.

> **Gateway Integration:** Designed to operate standalone on port `8001` or serve as the retrieval upstream for the [**LLM Guardrail & Routing Gateway**](https://github.com/mkmanoj30/LLM-Guardrail-Automated-Evaluation-Gateway) (`:8000`) to enforce PII redaction, dynamic model tiering, and egress faithfulness verification.

---

## Architecture Overview

```text
Incoming Markdown / Documents
             │
             ▼
┌─────────────────────────────────────────┐
│  Structure-Aware Markdown Chunker       │
│  (Preserves Tables & Section Hierarchy) │
└──────┬──────────────────────────┬───────┘
       │                          │
       ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐
│  Dense Vector Store │  │  Sparse BM25 Store  │
│  FAISS CPU + MiniLM │  │  Exact Token Match  │
└──────┬──────────────┘  └────────┬────────────┘
       │                          │
       └────────────┬─────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Reciprocal Rank Fusion  │
       │     (RRF, k = 60)       │
       └────────────┬────────────┘
                    ▼
        Top-K Grounded Citations
                    │
                    ▼  (Optional Downstream Integration)
┌─────────────────────────────────────────┐
│  LLM Guardrail Gateway (:8000)          │
│  PII Guard -> Tier Routing -> Egress CB │
└─────────────────────────────────────────┘
```

---

## Key Features

* **Structure-Aware Markdown Ingestion:** Splits documents along heading boundaries while keeping Markdown tables, YAML blocks, and multi-step lists intact.
* **Dual-Index Retrieval:** Pairs dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384-d) for semantic intent with `rank-bm25` for exact error codes (`SCHEMA-MISMATCH-422`, `AUTH-TOKEN-401`), IDs, and numerical thresholds.
* **Reciprocal Rank Fusion (RRF):** Merges dense and sparse rankings using $RRF(d) = \sum \frac{1}{k + \text{rank}(d)}$ ($k=60$) to eliminate score-scale bias.
* **Disk Persistence & Hot Reset:** Automatically serializes FAISS indexes, BM25 models, and chunk metadata to `data/processed/`, with a `/reset` endpoint for clean test cycles.
* **Plug-and-Play Gateway Orchestration:** Includes `scripts/query_pipeline.py` to ingest documents, retrieve top-K chunks, and forward context payloads directly to the LLM Guardrail Gateway.

---

## Project Structure

```text
hybrid-rag-engine/
├── data/
│   ├── raw/                  # Raw source documents
│   └── processed/            # Persisted FAISS index, BM25 cache, and chunk store
├── scripts/
│   └── query_pipeline.py     # End-to-end ingestion, retrieval & gateway orchestrator
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI endpoints (/ingest, /retrieve, /reset, /health)
│   ├── config.py             # Pydantic settings & directory management
│   ├── ingestion/
│   │   └── chunker.py        # Heading- and table-preserving Markdown chunker
│   └── retrieval/
│       ├── dense_store.py    # FAISS CPU index & MiniLM embedding wrapper
│       ├── sparse_store.py   # BM25 tokenizer & persistent keyword index
│       └── hybrid_engine.py  # RRF fusion & top-K citation builder
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu rank-bm25 pydantic pydantic-settings httpx
```

### 2. Start the Retrieval Service (`:8001`)
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

### 3. Verify Health
```bash
curl [http://127.0.0.1:8001/api/v1/health](http://127.0.0.1:8001/api/v1/health)
```

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service readiness and index status check |
| `POST` | `/api/v1/ingest` | Chunks, embeds, indexes, and persists Markdown documents |
| `POST` | `/api/v1/retrieve` | Executes hybrid Dense + Sparse RRF search and returns top-K chunks |
| `POST` | `/api/v1/reset` | Clears persisted FAISS/BM25 indexes and resets memory state |

### Example Retrieval Request
```bash
curl -X POST "[http://127.0.0.1:8001/api/v1/retrieve](http://127.0.0.1:8001/api/v1/retrieve)" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What immediate action occurs when SCHEMA-MISMATCH-422 is triggered?",
    "top_k": 3
  }'
```

---

## Connecting with LLM Guardrail Gateway

To run the full end-to-end RAG + Guardrail pipeline:

1. Start the **LLM Guardrail Gateway** on port `8000` (`uvicorn app.main:app --port 8000`).
2. Start this **Hybrid RAG Engine** on port `8001` (`uvicorn src.api.main:app --port 8001`).
3. Execute the orchestrator script:
   ```bash
   python scripts/query_pipeline.py
   ```
