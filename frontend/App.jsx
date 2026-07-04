import { useState } from "react";

// RegLens — regulatory RAG interface
// Run backend first: uvicorn api:app --port 8000
// Then: npm run dev

const API = "http://localhost:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const ask = async () => {
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: 6 }),
      });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      setResult(await resp.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const exampleQuestions = [
    "What obligations does the EU AI Act impose on high-risk AI systems?",
    "How does GDPR Article 22 treat automated decision-making?",
    "Compare EU and US approaches to AI transparency requirements",
  ];

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>RegLens</h1>
        <p style={styles.subtitle}>
          Cross-jurisdictional AI regulatory research — EU AI Act, GDPR, DSA, NIST AI RMF, SEC filings
        </p>
      </header>

      <div style={styles.searchRow}>
        <input
          style={styles.input}
          value={question}
          placeholder="Ask a regulatory question..."
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button style={styles.button} onClick={ask} disabled={loading}>
          {loading ? "Searching..." : "Ask"}
        </button>
      </div>

      {!result && (
        <div style={styles.examples}>
          {exampleQuestions.map((q) => (
            <button key={q} style={styles.exampleChip} onClick={() => setQuestion(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      {error && <div style={styles.error}>{error}</div>}

      {result && (
        <div style={styles.resultCard}>
          <div style={styles.answer}>{result.answer}</div>
          <div style={styles.sourcesHeader}>Sources</div>
          {result.sources.map((s) => (
            <div key={s.n} style={styles.source}>
              <span style={styles.sourceNum}>[{s.n}]</span>
              <span style={styles.sourceTitle}>{s.title}</span>
              <span style={styles.sourceSection}>{s.section}</span>
              <span style={styles.sourceScore}>sim {s.score}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  page: { maxWidth: 780, margin: "0 auto", padding: "48px 24px", fontFamily: "Georgia, serif", color: "#2d2a24" },
  header: { marginBottom: 32 },
  title: { fontSize: 42, margin: 0, letterSpacing: "-1px" },
  subtitle: { color: "#6b6558", fontSize: 15, marginTop: 8 },
  searchRow: { display: "flex", gap: 12 },
  input: { flex: 1, padding: "14px 18px", fontSize: 16, border: "1.5px solid #c8c2b4", borderRadius: 8, fontFamily: "inherit", background: "#faf8f3" },
  button: { padding: "14px 28px", fontSize: 15, background: "#4a5940", color: "#faf8f3", border: "none", borderRadius: 8, cursor: "pointer", fontFamily: "inherit" },
  examples: { marginTop: 20, display: "flex", flexDirection: "column", gap: 8 },
  exampleChip: { textAlign: "left", padding: "10px 16px", background: "#f0ece2", border: "1px solid #ddd6c4", borderRadius: 8, cursor: "pointer", fontSize: 14, fontFamily: "inherit", color: "#5a5648" },
  error: { marginTop: 20, padding: 14, background: "#f8e8e4", borderRadius: 8, color: "#8b3a2f" },
  resultCard: { marginTop: 28, padding: 24, background: "#faf8f3", border: "1.5px solid #ddd6c4", borderRadius: 12 },
  answer: { fontSize: 16, lineHeight: 1.7, whiteSpace: "pre-wrap" },
  sourcesHeader: { marginTop: 24, fontSize: 13, textTransform: "uppercase", letterSpacing: 1, color: "#8a8474", borderTop: "1px solid #ddd6c4", paddingTop: 16 },
  source: { display: "flex", gap: 10, alignItems: "baseline", padding: "8px 0", fontSize: 14 },
  sourceNum: { color: "#4a5940", fontWeight: "bold" },
  sourceTitle: { fontWeight: 600 },
  sourceSection: { color: "#6b6558" },
  sourceScore: { marginLeft: "auto", color: "#a09a88", fontSize: 12 },
};
