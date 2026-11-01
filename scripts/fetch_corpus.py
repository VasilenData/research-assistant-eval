"""Download a corpus of arXiv abstracts through the official arXiv API (polite: one request per 3 s).

    python scripts/fetch_corpus.py --per-category 1000 --out data/arxiv_corpus.jsonl
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"a": "http://www.w3.org/2005/Atom"}
CATS = ["cs.CL", "cs.IR", "cs.LG", "cs.DB"]


def fetch(cat: str, start: int, n: int) -> list[dict]:
    q = urllib.parse.urlencode({"search_query": f"cat:{cat}", "start": start, "max_results": n,
                                "sortBy": "submittedDate", "sortOrder": "descending"})
    req = urllib.request.Request(f"https://export.arxiv.org/api/query?{q}", headers={"User-Agent": "research-assistant-eval/1.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=60).read())  # noqa: S310 fixed https host
    out = []
    for e in root.findall("a:entry", NS):
        arx = e.find("a:id", NS).text.rsplit("/abs/", 1)[-1]
        abstract = re.sub(r"\s+", " ", e.find("a:summary", NS).text).strip()
        out.append({"id": re.sub(r"v\d+$", "", arx), "title": re.sub(r"\s+", " ", e.find("a:title", NS).text).strip(),
                    "abstract": abstract, "published": e.find("a:published", NS).text[:10], "category": cat})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=1000)
    ap.add_argument("--out", default="data/arxiv_corpus.jsonl")
    a = ap.parse_args()
    seen, rows = set(), []
    for cat in CATS:
        got = 0
        while got < a.per_category:
            batch = fetch(cat, got, min(500, a.per_category - got))
            if not batch:
                break
            for r in batch:
                if r["id"] not in seen and len(r["abstract"]) > 400:
                    seen.add(r["id"])
                    rows.append(r)
            got += len(batch)
            time.sleep(3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(len(rows), "papers", min(r["published"] for r in rows), "to", max(r["published"] for r in rows))


if __name__ == "__main__":
    main()
