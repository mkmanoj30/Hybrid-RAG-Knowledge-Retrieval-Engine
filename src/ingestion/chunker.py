# src/ingestion/chunker.py

import re
import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    document_name: str
    section_path: str
    content: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarkdownStructureChunker:
    """
    Structure-aware chunker designed for enterprise policy, standard, 
    and technical markdown documentation.
    """

    def __init__(self, target_chunk_size: int = 400, overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap
        # Regex matching markdown headers (# through ####)
        self.header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Heuristic token estimate based on word length."""
        return max(1, len(text.split()))

    def chunk_document(self, content: str, document_name: str) -> List[DocumentChunk]:
        """
        Parses Markdown into hierarchical sections based on headers,
        then bundles content to respect target token sizes.
        """
        lines = content.splitlines()
        chunks: List[DocumentChunk] = []

        current_hierarchy: Dict[int, str] = {}
        current_section_lines: List[str] = []

        def get_current_path() -> str:
            levels = sorted(current_hierarchy.keys())
            if not levels:
                return "Root"
            return " > ".join(current_hierarchy[lvl] for lvl in levels)

        def flush_current_section():
            if not current_section_lines:
                return

            section_text = "\n".join(current_section_lines).strip()
            if not section_text:
                return

            tokens = self._estimate_tokens(section_text)
            path = get_current_path()

            # If section is within budget, emit as a single cohesive chunk
            if tokens <= self.target_chunk_size:
                chunks.append(
                    DocumentChunk(
                        document_name=document_name,
                        section_path=path,
                        content=section_text,
                        token_count=tokens,
                        metadata={"type": "standard_section"}
                    )
                )
            else:
                # Sub-split oversized sections by paragraphs or list blocks
                sub_chunks = self._subdivide_large_section(
                    section_text, path, document_name
                )
                chunks.extend(sub_chunks)

            current_section_lines.clear()

        for line in lines:
            header_match = self.header_pattern.match(line)
            if header_match:
                # Flush the prior section buffer
                flush_current_section()

                hashes, title = header_match.groups()
                level = len(hashes)
                
                # Prune any deeper headers from hierarchy
                current_hierarchy = {
                    lvl: name for lvl, name in current_hierarchy.items() if lvl < level
                }
                current_hierarchy[level] = title.strip()
            else:
                current_section_lines.append(line)

        # Flush any trailing section
        flush_current_section()
        return chunks

    def _subdivide_large_section(
        self, text: str, section_path: str, document_name: str
    ) -> List[DocumentChunk]:
        """
        Splits oversized sections along double-newlines (paragraphs),
        avoiding splits across table rows or list items.
        """
        paragraphs = text.split("\n\n")
        sub_chunks: List[DocumentChunk] = []
        current_buffer: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._estimate_tokens(para)

            if current_tokens + para_tokens > self.target_chunk_size and current_buffer:
                chunk_text = "\n\n".join(current_buffer)
                sub_chunks.append(
                    DocumentChunk(
                        document_name=document_name,
                        section_path=section_path,
                        content=chunk_text,
                        token_count=current_tokens,
                        metadata={"type": "subdivided_paragraph"}
                    )
                )
                current_buffer = []
                current_tokens = 0

            current_buffer.append(para)
            current_tokens += para_tokens

        if current_buffer:
            chunk_text = "\n\n".join(current_buffer)
            sub_chunks.append(
                DocumentChunk(
                    document_name=document_name,
                    section_path=section_path,
                    content=chunk_text,
                    token_count=current_tokens,
                    metadata={"type": "subdivided_paragraph"}
                )
            )

        return sub_chunks