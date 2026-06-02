# download_pdfs.py

import json
import time
from pathlib import Path
from scraper.collect import _download_pdf
pdf_dir = Path("data/raw/pdfs")
pdf_dir.mkdir(parents=True, exist_ok=True)
with open("data/raw/metadata.jsonl") as f:
    papers = [json.loads(line) for line in f]

for i, p in enumerate(papers, 1):
    path = _download_pdf(p, pdf_dir)
    if path:
        print(f"[{i}/{len(papers)}] downloaded")
    else:
        print(f"[{i}/{len(papers)}] failed")
    time.sleep(3)