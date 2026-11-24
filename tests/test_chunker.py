# tests/test_chunker.py
import sys
from pathlib import Path

# Inject repository root into module search path before importing src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingestion.chunker import MarkdownStructureChunker

sample_markdown = """# Architecture Standards

## Section 1: Security Boundaries
All production traffic must pass through the edge gateway.
Unauthenticated requests are dropped at the ingress layer.

| Port | Service | Status |
|---|---|---|
| 443 | HTTPS Ingress | Open |
| 8000 | FastAPI Gateway | Internal |

### Subsection 1.1: Database Isolation
Production databases must reside within an isolated VPC.
Direct external SSH or TCP connections are strictly prohibited.

## Section 2: Data Retention
All cold backups are archived to S3 Glacier after 90 days of storage.
Audit logs are preserved for a statutory period of 7 years.
"""

def test_chunker():
    chunker = MarkdownStructureChunker(target_chunk_size=100)
    chunks = chunker.chunk_document(sample_markdown, "architecture_spec.md")

    print(f"Total Chunks Generated: {len(chunks)}\n")
    for idx, c in enumerate(chunks, 1):
        print(f"--- Chunk {idx} ---")
        print(f"Path:   {c.section_path}")
        print(f"Tokens: ~{c.token_count}")
        print(f"Body:\n{c.content}\n")

if __name__ == "__main__":
    test_chunker()