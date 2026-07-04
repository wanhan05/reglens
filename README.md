# RegLens — Cross-Jurisdictional AI Regulatory RAG System

A retrieval-augmented generation system for AI regulatory research, indexing the EU AI Act, GDPR, DSA, DMA, NIST AI RMF, and SEC AI-related filings. Ask compliance questions in natural language; get grounded answers with inline citations to specific articles and sections.

## Architecture

```
Data Ingestion              Processing                 Retrieval & Generation
─────────────               ──────────                 ──────────────────────
SEC EDGAR API      ──┐      HTML/PDF parsing           Query → embed →
EUR-Lex (EU AI     ──┼──►   Structural chunking   ──►  FAISS top-k retrieval →
 Act, GDPR, DSA)     │      (Article/Section-aware)    LLM generation with
NIST AI RMF        ──┘      512-word windows,          inline [n] citations
                            64-word overlap
```

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2 default; LegalBERT option)
- **Vector store:** FAISS (exact inner-product search)
- **Generation:** local HuggingFace model (Qwen2.5-1.5B-Instruct) or Anthropic API
- **Chunking:** structure-aware — splits on Article/Section boundaries first, then sliding window
- **Evaluation:** recall@k on hand-labeled gold questions + citation coverage for faithfulness

## Setup

```bash
pip install -r requirements.txt

# 1. Ingest documents
python -m ingestion.eurlex                          # EU AI Act, GDPR, DSA, DMA, NIST RMF
python -m ingestion.sec_edgar --query "artificial intelligence" --forms 10-K --limit 50

# 2. Process and chunk
python -m processing.chunker

# 3. Build vector index
python -m retrieval.embed_index

# 4. Query from CLI
python -m retrieval.query "What does the EU AI Act require for high-risk systems?"

# 5. Run evaluation
python -m evaluation.evaluate

# 6. Serve API + frontend
uvicorn api:app --port 8000
# then in frontend/: npm install && npm run dev
```

## Design decisions

**Structure-aware chunking.** Regulatory documents have strong internal structure (Articles, Sections, Recitals). Chunking on structural boundaries preserves legal meaning — a chunk that spans two Articles produces incoherent retrievals. Falls back to sliding-window chunking (512 words, 64 overlap) for long sections.

**Citation-first generation.** The system prompt requires every claim to carry an inline citation, and the evaluation harness measures citation coverage. An uncited regulatory answer is worse than no answer.

**Jurisdiction awareness.** Sources span EU and US frameworks; the generation prompt explicitly instructs the model to flag jurisdictional differences rather than blending them.

## Evaluation

Retrieval: recall@6 measured against a hand-labeled gold set mapping questions to expected documents and section patterns (e.g., "automated decision-making" → GDPR Article 22).

Faithfulness proxy: citation coverage (fraction of answer sentences carrying at least one [n] citation).

## Roadmap

- [ ] Expand gold evaluation set to 50+ questions
- [ ] Add regulations.gov (CFPB, FTC proposed rules) ingestion
- [ ] Cross-encoder reranking after FAISS retrieval
- [ ] Compliance gap analysis mode: given a system description, surface applicable requirements across all indexed frameworks
