"""
Collect PubMed abstracts + metadata into a local dataset.

Usage (PowerShell):
  python pubmed_collect.py --term "pancreatic cancer" --retmax 5000 --out data/pubmed_pancreatic_cancer.csv

Notes:
  - NCBI requires an email. Optionally set an API key for higher rate limits.
  - This script uses esearch (IDs) + efetch (records) in batches.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
from bs4 import BeautifulSoup
from Bio import Entrez
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

import os
import time
import math
import argparse
import requests
import pandas as pd

load_dotenv()
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")


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
        records = Entrez.read(handle)
        for article in records.get("PubmedArticle", []):
            yield article
        time.sleep(sleep_s)  # don't go above NCBI limit

# Source - https://stackoverflow.com/a/79391683
# Posted by jimnoneill
# Retrieved 2026-06-15, License - CC BY-SA 4.0

def gather_pmids_time_slice(term, start_date, end_date):
    """
    Recursively (or iteratively) gather all PMIDs from 'start_date' to 'end_date'
    even if count > 9999, by subdividing the date range into multiple slices
    proportional to the result count.

    start_date, end_date: "YYYY/MM/DD" strings
    returns: list of PMID strings
    """

    # 1) minimal eSearch to get count
    count_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={term}&rettype=count"
        f"&mindate={start_date}&maxdate={end_date}"
        "&usehistory=n"
    )
    if NCBI_API_KEY:
        count_url += f"&api_key={NCBI_API_KEY}"

    resp = None
    for attempt in range(1, 6):
        try:
            resp = requests.get(count_url, timeout=30)
            if resp.status_code == 200:
                break
            elif resp.status_code == 429:
                wait_s = 3 * attempt
                print(f"[time-slice count] Rate-limit, waiting {wait_s}s ...")
                time.sleep(wait_s)
        except Exception as e:
            print(f"Error counting range {start_date}-{end_date}: {str(e)}")
            if attempt == 5:
                break
            time.sleep(3 * attempt)

    if not resp or resp.status_code != 200:
        print(f"[time-slice] Could not get count for {start_date}-{end_date}")
        return []

    soup = BeautifulSoup(resp.content, 'lxml-xml')
    c_tag = soup.find('Count')
    if not c_tag:
        print(f"[time-slice] No count found for {start_date}-{end_date}")
        return []

    c = int(c_tag.text)
    print(f"Range {start_date}-{end_date} => count = {c}")

    if c == 0:
        return []

    # 2) if c <= 9999 => normal chunk retrieval
    if c <= 9999:
        pmid_list = []
        retmax = 10000  # up to 10000
        start_offset = 0
        # retrieve in standard lumps
        while start_offset < c:
            chunk_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
                f"db=pubmed&retmode=xml&term={term}"
                f"&mindate={start_date}&maxdate={end_date}"
                f"&retstart={start_offset}&retmax={retmax}"
                "&usehistory=n"
            )
            if NCBI_API_KEY:
                chunk_url += f"&api_key={NCBI_API_KEY}"

            pmids_this_chunk = []
            for attempt in range(1, 6):
                try:
                    ch_resp = requests.get(chunk_url, timeout=30)
                    if ch_resp.status_code == 200:
                        s2 = BeautifulSoup(ch_resp.content, 'lxml-xml')
                        pmids_this_chunk = [x.text for x in s2.find_all('Id')]
                        break
                    elif ch_resp.status_code == 429:
                        w_s = 3 * attempt
                        print(f"[time-slice normal chunk] Rate-limit offset {start_offset}, wait {w_s}s.")
                        time.sleep(w_s)
                except Exception as e:
                    print(f"[time-slice chunk] Error offset={start_offset}: {e}")
                    if attempt == 5:
                        break
                    time.sleep(3*attempt)

            if not pmids_this_chunk:
                print(f"[time-slice chunk] No PMIDs found offset={start_offset}, breaking.")
                break

            pmid_list.extend(pmids_this_chunk)
            got_count = len(pmids_this_chunk)
            print(f"[time-slice normal chunk] {start_date}-{end_date}: got {got_count} pmids at offset={start_offset}, total={len(pmid_list)}")
            if got_count < retmax:
                break
            start_offset += got_count

        return pmid_list

    # 3) if c > 9999 => we must subdivide into intervals_needed
    # Let's do date slicing in equal intervals by day
    # so intervals_needed = ceil(c / 9999)
    intervals_needed = math.ceil(c / 9999)
    print(f"[time-slice] c={c} >9999 => subdividing into {intervals_needed} slices")

    sd = datetime.strptime(start_date, "%Y/%m/%d").date()
    ed = datetime.strptime(end_date, "%Y/%m/%d").date()
    total_days = (ed - sd).days + 1
    # each slice covers ~ total_days / intervals_needed days
    # watch rounding
    slice_days = math.ceil(total_days / intervals_needed)

    all_pmids = []
    slice_start = sd
    while slice_start <= ed:
        slice_end = slice_start + pd.Timedelta(days=slice_days - 1)
        if slice_end > ed:
            slice_end = ed
        slice_start_str = slice_start.strftime("%Y/%m/%d")
        slice_end_str = slice_end.strftime("%Y/%m/%d")

        print(f"[time-slice] Sub-slice => {slice_start_str} to {slice_end_str}")
        sub_pmids = gather_pmids_time_slice(term, slice_start_str, slice_end_str)
        all_pmids.extend(sub_pmids)

        # move to next slice
        slice_start = slice_end + pd.Timedelta(days=1)

    return all_pmids



def main():
    p = argparse.ArgumentParser()
    query = "pancreatic cancer"
    # query = """
    # (
    #     "Pancreatic Neoplasms"[MeSH Terms]
    #     OR "pancreatic cancer"[Title/Abstract]
    #     OR "pancreatic neoplasm"[Title/Abstract]
    #     OR "pancreatic neoplasms"[Title/Abstract]
    #     OR "pancreatic tumor"[Title/Abstract]
    #     OR "pancreatic tumors"[Title/Abstract]

    #     OR "pancreatic adenocarcinoma"[Title/Abstract]
    #     OR "ductal adenocarcinoma"[Title/Abstract]
    #     OR "pancreatic ductal adenocarcinoma"[Title/Abstract]
    #     OR "ductal carcinoma"[Title/Abstract]
    #     OR PDAC[Title/Abstract]
    #     OR "pancreatic acinar cell carcinoma"[Title/Abstract]

    # OR "pancreatic adenosquamous carcinoma"[Title/Abstract]
    # OR "PASC"[Title/Abstract]

    # OR "squamous cell carcinoma"[Title/Abstract]
    # OR "epidermoid carcinoma of the pancreas"[Title/Abstract]

    # OR "colloid carcinoma"[Title/Abstract]
    # OR "mucinous carcinoma"[Title/Abstract]

    # OR "pancreatic neuroendocrine tumor"[Title/Abstract]
    # OR "pancreatic neuroendocrine tumors"[Title/Abstract]
    # OR "neuroendocrine tumor"[Title/Abstract]
    # OR "neuroendocrine tumors"[Title/Abstract]
    # OR "islet cell tumor"[Title/Abstract]
    # OR "islet cell tumors"[Title/Abstract]

    # OR IPMN[Title/Abstract]
    # OR "intraductal papillary mucinous neoplasm"[Title/Abstract]
    # OR "intraductal papillary mucinous neoplasms"[Title/Abstract]
    # OR "intraductal papillary-mucinous neoplasm"[Title/Abstract]

    # OR "pancreatic mucinous neoplasm"[Title/Abstract]
    # OR "pancreatic mucinous neoplasms"[Title/Abstract]

    # OR "pancreatic cyst"[Title/Abstract]
    # OR "pancreatic cysts"[Title/Abstract]
    p.add_argument("--term", default=query)
    p.add_argument("--retmax", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--out", default="data/pubmed_only_pancreaticcancer.csv")
    p.add_argument("--email", default=NCBI_EMAIL)
    p.add_argument("--api-key", default=NCBI_API_KEY)
    args = p.parse_args()

    Entrez.email = "msgalvez@uci.edu"
    if args.api_key:
        Entrez.api_key = args.api_key

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # pmids = search_pmids(args.term, args.retmax) # Old Method
    pmids = gather_pmids_time_slice(query, "2020/01/01", "2026/01/01")
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

