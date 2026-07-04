"""
RegLens — RAG query engine.

Retrieval: FAISS top-k over embedded regulatory chunks.
Generation: grounded answer with inline citations via either
  (a) a local HuggingFace model (default: Qwen2.5-1.5B-Instruct, runs on laptop), or
  (b) the Anthropic API if ANTHROPIC_API_KEY is set (better quality).

Every answer cites its sources as [1], [2], ... mapped to document + section.

Usage:
    python -m retrieval.query "What does the EU AI Act require for high-risk systems?"
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_DIR = DATA_DIR / "index"

SYSTEM_PROMPT = """You are RegLens, a regulatory research assistant. Answer the user's question using ONLY the provided source excerpts. Rules:
- Cite sources inline as [1], [2], etc. after each claim.
- If the sources do not contain the answer, say so explicitly — do not speculate.
- Quote regulation text sparingly and precisely.
- Note jurisdictional differences when sources span EU and US frameworks."""


class RegLensEngine:
    def __init__(self, model_key: str = "default"):
        import faiss
        from sentence_transformers import SentenceTransformer

        meta = json.loads((INDEX_DIR / f"meta_{model_key}.json").read_text())
        self.embedder = SentenceTransformer(meta["model"])
        self.index = faiss.read_index(str(INDEX_DIR / f"reglens_{model_key}.faiss"))
        self.chunks = json.loads((INDEX_DIR / f"chunks_{model_key}.json").read_text())

    def retrieve(self, query: str, k: int = 6) -> list[dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    @staticmethod
    def _format_context(results: list[dict]) -> str:
        blocks = []
        for i, r in enumerate(results, 1):
            blocks.append(
                f"[{i}] {r['title']} — {r['section']} (source: {r['source']})\n{r['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, query: str, k: int = 6) -> dict:
        results = self.retrieve(query, k=k)
        context = self._format_context(results)
        prompt = f"Source excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"

        if os.environ.get("ANTHROPIC_API_KEY"):
            answer_text = self._generate_anthropic(prompt)
        else:
            answer_text = self._generate_local(prompt)

        return {
            "query": query,
            "answer": answer_text,
            "sources": [
                {"n": i + 1, "title": r["title"], "section": r["section"],
                 "source": r["source"], "score": round(r["score"], 3)}
                for i, r in enumerate(results)
            ],
        }

    def _generate_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _generate_local(self, prompt: str) -> str:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument("--model", default="default")
    p.add_argument("-k", type=int, default=6)
    args = p.parse_args()

    engine = RegLensEngine(model_key=args.model)
    result = engine.answer(" ".join(args.question), k=args.k)

    print("\n" + "=" * 70)
    print(textwrap.fill(result["answer"], width=70))
    print("=" * 70 + "\nSources:")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['title']} — {s['section']} (sim {s['score']})")


if __name__ == "__main__":
    main()
