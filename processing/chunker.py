"""
RegLens — document processing: parse, clean, and chunk regulatory documents.

Takes raw HTML/PDF from data/raw/, produces clean text chunks with metadata
in data/processed/chunks.jsonl ready for embedding.

Chunking strategy: regulatory documents have strong internal structure
(articles, sections, recitals). We chunk on structural boundaries first,
then fall back to token-window chunking with overlap for long sections.

Usage:
    python -m processing.chunker
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

CHUNK_SIZE = 512       # target words per chunk
CHUNK_OVERLAP = 64     # word overlap between adjacent chunks

# Regulatory structure markers (EU + US styles)
ARTICLE_RE = re.compile(r"^(Article\s+\d+|ARTICLE\s+\d+|SEC\.\s*\d+|Section\s+\d+|§\s*\d+)", re.MULTILINE)


def html_to_text(html: str) -> str:
    """Strip HTML to clean text, preserving paragraph structure."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Normalize whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text(path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def structural_split(text: str) -> list[str]:
    """Split on regulatory structure (Articles/Sections) if present."""
    matches = list(ARTICLE_RE.finditer(text))
    if len(matches) < 3:  # not structured enough; return whole text
        return [text]
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())
    # Include preamble before first article
    if matches[0].start() > 200:
        sections.insert(0, text[: matches[0].start()].strip())
    return [s for s in sections if s]


def window_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunking on word boundaries."""
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + size]
        if len(chunk_words) < 50:  # skip trailing fragments
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def chunk_document(text: str, doc_meta: dict) -> list[dict]:
    """Structural split, then window-chunk oversized sections."""
    records = []
    for si, section in enumerate(structural_split(text)):
        # Capture section header for citation metadata
        header_match = ARTICLE_RE.match(section)
        section_label = header_match.group(0) if header_match else f"part-{si}"
        section_label = " ".join(section_label.split())  # normalize whitespace
        for ci, chunk in enumerate(window_chunk(section)):
            records.append({
                "doc_id": doc_meta.get("id") or doc_meta.get("accession", "unknown"),
                "title": doc_meta.get("title") or doc_meta.get("company", "unknown"),
                "source": doc_meta.get("source", "unknown"),
                "section": section_label,
                "chunk_index": ci,
                "text": chunk,
            })
    return records


def run() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for manifest_path in RAW_DIR.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        for doc_meta in manifest:
            path = Path(doc_meta["path"])
            if not path.exists():
                print(f"MISSING {path}")
                continue
            print(f"Processing {doc_meta.get('title') or doc_meta.get('company')}...")
            if path.suffix == ".pdf":
                text = pdf_to_text(path)
            else:
                text = html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
            chunks = chunk_document(text, doc_meta)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks")

    out = PROC_DIR / "chunks.jsonl"
    with out.open("w") as fh:
        for rec in all_chunks:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks → {out}")


if __name__ == "__main__":
    run()
