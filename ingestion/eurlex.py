"""
RegLens — EU regulatory document ingestion.

Downloads the EU AI Act and related EU regulatory texts from EUR-Lex.
EUR-Lex serves official documents at stable CELEX-numbered URLs.

Usage:
    python -m ingestion.eurlex
"""

import json
from pathlib import Path

import requests

HEADERS = {"User-Agent": "RegLens Research wanhansun22@gmail.com"}
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "eu"

# CELEX identifiers for key EU digital/AI regulation
DOCUMENTS = {
    "eu_ai_act": {
        "celex": "32024R1689",
        "title": "EU AI Act (Regulation 2024/1689)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
    },
    "gdpr": {
        "celex": "32016R0679",
        "title": "GDPR (Regulation 2016/679)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
    },
    "dsa": {
        "celex": "32022R2065",
        "title": "Digital Services Act (Regulation 2022/2065)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2065",
    },
    "dma": {
        "celex": "32022R1925",
        "title": "Digital Markets Act (Regulation 2022/1925)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R1925",
    },
}

# US framework documents (public, stable URLs)
US_FRAMEWORKS = {
    "nist_ai_rmf": {
        "title": "NIST AI Risk Management Framework 1.0",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "format": "pdf",
    },
}


def download_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for key, doc in DOCUMENTS.items():
        print(f"Downloading {doc['title']}...")
        resp = requests.get(doc["url"], headers=HEADERS, timeout=120)
        resp.raise_for_status()
        out = RAW_DIR / f"{key}.html"
        out.write_text(resp.text, encoding="utf-8")
        manifest.append({
            "id": key,
            "title": doc["title"],
            "celex": doc.get("celex"),
            "path": str(out),
            "source": "EUR-Lex",
        })
        print(f"  saved → {out}")

    for key, doc in US_FRAMEWORKS.items():
        print(f"Downloading {doc['title']}...")
        resp = requests.get(doc["url"], headers=HEADERS, timeout=120)
        resp.raise_for_status()
        out = RAW_DIR / f"{key}.pdf"
        out.write_bytes(resp.content)
        manifest.append({
            "id": key,
            "title": doc["title"],
            "path": str(out),
            "source": "NIST",
        })
        print(f"  saved → {out}")

    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written with {len(manifest)} documents.")


if __name__ == "__main__":
    download_all()
