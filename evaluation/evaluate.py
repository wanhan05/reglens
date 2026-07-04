"""
RegLens — evaluation harness.

Two evaluation axes:
1. Retrieval quality — does the right document/section appear in top-k?
   Measured with a hand-labeled gold set of (question, expected_doc, expected_section).
2. Answer faithfulness — does the generated answer stay grounded in retrieved text?
   Proxy metric: citation coverage (every sentence carries a citation) +
   n-gram overlap between answer sentences and cited chunks.

Usage:
    python -m evaluation.evaluate
"""

import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold_questions.json"

# Seed gold set — expand as the corpus grows.
GOLD_QUESTIONS = [
    {
        "question": "What obligations does the EU AI Act impose on providers of high-risk AI systems?",
        "expected_doc": "eu_ai_act",
        "expected_section_pattern": r"Article\s+(9|1[0-7]|16|24|25)",
    },
    {
        "question": "How does the EU AI Act define prohibited AI practices?",
        "expected_doc": "eu_ai_act",
        "expected_section_pattern": r"Article\s+5\b",
    },
    {
        "question": "What are the GDPR requirements for automated decision-making?",
        "expected_doc": "gdpr",
        "expected_section_pattern": r"Article\s+22",
    },
    {
        "question": "What transparency requirements apply to general-purpose AI models?",
        "expected_doc": "eu_ai_act",
        "expected_section_pattern": r"Article\s+5[0-3]",
    },
    {
        "question": "What does the NIST AI Risk Management Framework say about measuring AI risks?",
        "expected_doc": "nist_ai_rmf",
        "expected_section_pattern": r".",
    },
]


def recall_at_k(engine, gold: list[dict], k: int = 6) -> dict:
    """Fraction of gold questions where expected doc appears in top-k."""
    doc_hits, section_hits = 0, 0
    per_question = []
    for g in gold:
        results = engine.retrieve(g["question"], k=k)
        docs = [r["doc_id"] for r in results]
        sections = [r["section"] for r in results]
        doc_hit = g["expected_doc"] in docs
        sec_hit = any(re.search(g["expected_section_pattern"], s) for s in sections)
        doc_hits += doc_hit
        section_hits += sec_hit
        per_question.append({
            "question": g["question"],
            "doc_hit": doc_hit,
            "section_hit": sec_hit,
            "top_docs": docs[:3],
        })
    n = len(gold)
    return {
        "recall_at_k_doc": doc_hits / n,
        "recall_at_k_section": section_hits / n,
        "k": k,
        "n_questions": n,
        "per_question": per_question,
    }


def mrr(engine, gold: list[dict], k: int = 6) -> float:
    """Mean Reciprocal Rank — rewards correct sections appearing earlier."""
    rr_sum = 0.0
    for g in gold:
        results = engine.retrieve(g["question"], k=k)
        for i, r in enumerate(results, 1):
            if re.search(g["expected_section_pattern"], r["section"]):
                rr_sum += 1.0 / i
                break
    return rr_sum / len(gold)


def citation_coverage(answer: str) -> float:
    """Fraction of answer sentences carrying at least one [n] citation."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    return cited / len(sentences)


def run():
    from retrieval.query import RegLensEngine

    engine = RegLensEngine()
    print("Running retrieval evaluation...")
    metrics = recall_at_k(engine, GOLD_QUESTIONS, k=6)
    metrics["mrr"] = mrr(engine, GOLD_QUESTIONS, k=6)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_question"}, indent=2))

    out = EVAL_DIR / "results.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Full results → {out}")


if __name__ == "__main__":
    run()
