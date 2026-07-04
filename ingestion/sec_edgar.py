"""
RegLens — SEC EDGAR ingestion module.

Pulls AI-related regulatory and disclosure documents from SEC EDGAR
full-text search API. No authentication required, but SEC requests
a descriptive User-Agent header with contact info.

Usage:
    python -m ingestion.sec_edgar --query "artificial intelligence" --forms 10-K --limit 50
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import requests

BASE_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SEARCH_API = "https://efts.sec.gov/LATEST/search-index"

# SEC requires a User-Agent identifying you. Replace with your info.
HEADERS = {
    "User-Agent": "RegLens Research wanhansun22@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "sec"


def search_filings(query: str, forms: str = "10-K", start_date: str = "2020-01-01",
                   end_date: str | None = None, limit: int = 50) -> list[dict]:
    """Search SEC EDGAR full-text search for filings matching a query."""
    end_date = end_date or date.today().isoformat()
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": f'"{query}"',
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "forms": forms,
    }
    # The public JSON endpoint used by the EDGAR full-text search UI:
    search_url = "https://efts.sec.gov/LATEST/search-index?" 
    resp = requests.get(
        "https://efts.sec.gov/LATEST/search-index",
        params=params,
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])[:limit]
    return hits


def fetch_filing_text(accession_no: str, cik: str) -> str:
    """Fetch the primary document text for a filing."""
    acc_clean = accession_no.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/index.json"
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    items = resp.json()["directory"]["item"]
    # Grab the largest .htm document (usually the primary filing)
    docs = [i for i in items if i["name"].endswith((".htm", ".txt"))]
    docs.sort(key=lambda d: int(d.get("size", 0) or 0), reverse=True)
    if not docs:
        return ""
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{docs[0]['name']}"
    resp = requests.get(doc_url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text


def run(query: str, forms: str, limit: int, delay: float = 0.15) -> None:
    """Search and download filings; save raw text + metadata to data/raw/sec."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    hits = search_filings(query, forms=forms, limit=limit)
    print(f"Found {len(hits)} filings for query '{query}' (forms={forms})")

    manifest = []
    for i, hit in enumerate(hits):
        src = hit["_source"]
        accession = src["adsh"]
        cik = src["ciks"][0]
        display_name = src.get("display_names", ["unknown"])[0]
        try:
            text = fetch_filing_text(accession, cik)
        except requests.HTTPError as e:
            print(f"  [{i+1}/{len(hits)}] SKIP {accession}: {e}")
            continue

        out_path = RAW_DIR / f"{accession.replace('-', '')}.html"
        out_path.write_text(text, encoding="utf-8")
        manifest.append({
            "accession": accession,
            "cik": cik,
            "company": display_name,
            "form": src.get("file_type", forms),
            "filed": src.get("file_date"),
            "path": str(out_path),
            "source": "SEC EDGAR",
            "query": query,
        })
        print(f"  [{i+1}/{len(hits)}] saved {display_name} ({accession})")
        time.sleep(delay)  # SEC rate limit: max 10 req/s; be polite

    manifest_path = RAW_DIR / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest_path.write_text(json.dumps(existing + manifest, indent=2))
    print(f"Wrote manifest with {len(manifest)} new entries → {manifest_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="artificial intelligence")
    p.add_argument("--forms", default="10-K")
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()
    run(args.query, args.forms, args.limit)
