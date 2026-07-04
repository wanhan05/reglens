"""Run with: python -m pytest tests/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from processing.chunker import structural_split, window_chunk, chunk_document, html_to_text


def test_structural_split_on_articles():
    doc = "Article 1\nFirst " + "w " * 50 + "\nArticle 2\nSecond " + "w " * 50 + "\nArticle 3\nThird"
    sections = structural_split(doc)
    assert len(sections) == 3


def test_unstructured_returns_whole():
    assert len(structural_split("no structure at all here")) == 1


def test_window_overlap():
    text = " ".join(f"w{i}" for i in range(1200))
    chunks = window_chunk(text)
    assert chunks[0].split()[-64:] == chunks[1].split()[:64]


def test_html_strips_scripts():
    out = html_to_text("<script>x()</script><p>keep</p>")
    assert "x()" not in out and "keep" in out


def test_chunk_metadata():
    doc = "Article 7\n" + "word " * 600
    chunks = chunk_document(doc, {"id": "d1", "title": "T", "source": "s"})
    assert all(c["doc_id"] == "d1" for c in chunks)
    assert chunks[0]["section"] == "Article 7"
