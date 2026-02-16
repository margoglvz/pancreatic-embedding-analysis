"""
Collect PubMed abstracts + metadata into a local dataset.

Usage (PowerShell):
  python pubmed_collect.py --term "pancreatic cancer" --retmax 5000 --out data/pubmed_pancreatic_cancer.csv

Notes:
  - NCBI requires an email. Optionally set an API key for higher rate limits.
  - This script uses esearch (IDs) + efetch (records) in batches.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from Bio import Entrez # Bio is Biopython
from tqdm import tqdm


def _get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _extract_year(article: Dict[str, Any]) -> Optional[int]:
    # Try multiple locations; PubMed XML is not uniform.
    year = _get(article, ["MedlineCitation", "Article", "Journal", "JournalIssue", "PubDate", "Year"])
    if year is None:
        year = _get(article, ["MedlineCitation", "Article", "ArticleDate", 0, "Year"])
    try:
        return int(year) if year is not None else None
    except Exception:
        return None


def _extract_abstract_text(article: Dict[str, Any]) -> Optional[str]:
    abs_parts = _get(article, ["MedlineCitation", "Article", "Abstract", "AbstractText"])
    if abs_parts is None:
        return None
    # abs_parts can be list[str] or list[dict] depending on labels
    out: List[str] = []
    for p in abs_parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            # Biopython sometimes returns {attributes... : text} style
            # but typically the text is directly in the dict values
            out.extend([str(v) for v in p.values() if isinstance(v, (str, int, float))])
        else:
            out.append(str(p))
    text = " ".join(s.strip() for s in out if str(s).strip())
    return text or None


def _extract_title(article: Dict[str, Any]) -> Optional[str]:
    title = _get(article, ["MedlineCitation", "Article", "ArticleTitle"])
    if title is None:
        return None
    return str(title).strip() or None


def _extract_journal(article: Dict[str, Any]) -> Optional[str]:
    j = _get(article, ["MedlineCitation", "Article", "Journal", "Title"])
    return str(j).strip() if j else None


def _extract_doi(article: Dict[str, Any]) -> Optional[str]:
    ids = _get(article, ["PubmedData", "ArticleIdList"])
    if not ids:
        return None
    for item in ids:
        try:
            if item.attributes.get("IdType") == "doi":
                return str(item).strip() or None
        except Exception:
            continue
    return None


def search_pmids(term: str, retmax: int) -> List[str]:
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, usehistory="n")
    record = Entrez.read(handle)
    return list(record.get("IdList", []))


def fetch_records(pmids: List[str], batch_size: int = 200, sleep_s: float = 0.34) -> Iterable[Dict[str, Any]]:
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
        print("Handle: ", handle)
        records = Entrez.read(handle)
        print("Records: ", records)
        for article in records.get("PubmedArticle", []):
            yield article
        time.sleep(sleep_s)  # don't go above NCBI limit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--term", default="pancreatic cancer")
    p.add_argument("--retmax", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--out", default="data/pubmed_pancreatic_cancer.csv")
    p.add_argument("--email", default=os.environ.get("NCBI_EMAIL", "your_email@example.com"))
    p.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    args = p.parse_args()

    Entrez.email = "msgalvez@uci.edu"
    if args.api_key:
        Entrez.api_key = args.api_key

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pmids = search_pmids(args.term, args.retmax)
    rows: List[Dict[str, Any]] = []

    for article in tqdm(fetch_records(pmids, batch_size=args.batch_size), total=len(pmids)):
        pmid = _get(article, ["MedlineCitation", "PMID"])
        rows.append(
            {
                "pmid": str(pmid) if pmid is not None else None,
                "doi": _extract_doi(article),
                "title": _extract_title(article),
                "abstract": _extract_abstract_text(article),
                "journal": _extract_journal(article),
                "year": _extract_year(article),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["abstract"]).drop_duplicates(subset=["pmid"])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()

