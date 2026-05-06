"""Senate Periodic Transaction Report (PTR) scraper.

Senate EFD (efdsearch.senate.gov) hosts STOCK Act disclosures filed by
senators. The site requires a CSRF + cookie session and a terms-of-use
agreement before serving any data.

Pipeline:
  1. GET /search/ to obtain a CSRF token.
  2. POST /search/home/ with the agreement and CSRF token to set a session cookie.
  3. POST /search/report/data/ paginated, filtering for report_types=[11] (PTR).
  4. For each report row, fetch the per-PTR HTML page and extract the trade table.
  5. Normalize the rows into the schema downstream modules expect.

Outputs:
  - data/senate_trades.csv    raw scrape, one row per disclosed trade
  - data/ptr_trades.csv       normalized to canonical schema
"""

from __future__ import annotations
import json
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import DATA_DIR, PTR_AMOUNT_BUCKETS

from config import SENATE_EFD_BASE_URL as BASE  # noqa: E402
SEARCH_URL = f"{BASE}/search/"
HOME_URL = f"{BASE}/search/home/"
DATA_URL = f"{BASE}/search/report/data/"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def _new_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    r = s.get(SEARCH_URL, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    csrf = soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]
    s.headers.update({"Origin": BASE, "Referer": HOME_URL})
    s.post(HOME_URL, data={"prohibition_agreement": 1,
                           "csrfmiddlewaretoken": csrf}, timeout=30)
    s.headers.update({
        "Referer": SEARCH_URL,
        "X-CSRFToken": s.cookies.get_dict().get("csrftoken", csrf),
        "X-Requested-With": "XMLHttpRequest",
    })
    return s


def _list_reports(s, start_date="01/01/2023 00:00:00", page_size=100):
    """Return [{senator, filer_type, report_title, date_filed, report_link}, ...]."""
    out, offset = [], 0
    while True:
        form = {
            "start": str(offset), "length": str(page_size),
            "report_types": "[11]", "filter_types": "[1]",
            "submitted_start_date": start_date, "submitted_end_date": "",
            "candidate_state": "", "senator_state": "",
            "first_name": "", "last_name": "",
        }
        r = s.post(DATA_URL, data=form, timeout=30)
        if r.status_code != 200:
            print(f"[load_senate] HTTP {r.status_code} at offset {offset}")
            break
        try:
            payload = r.json()
        except json.JSONDecodeError:
            print(f"[load_senate] non-JSON at offset {offset}")
            break
        rows = payload.get("data", [])
        total = payload.get("recordsTotal", 0)
        for row in rows:
            link_soup = BeautifulSoup(row[3], "html.parser") if len(row) > 3 else None
            tag = link_soup.find("a") if link_soup else None
            out.append({
                "senator": f"{row[0]} {row[1]}".strip(),
                "filer_type": row[2] if len(row) > 2 else "",
                "report_title": tag.text.strip() if tag else "",
                "date_filed": row[4] if len(row) > 4 else "",
                "report_link": tag["href"] if tag else "",
            })
        if len(out) >= total or not rows:
            break
        offset += len(rows)
        time.sleep(1)
    return out


def _parse_one_report(s, link):
    """Extract trade rows from a single PTR page."""
    url = BASE + link if link.startswith("/") else link
    r = s.get(url, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    trades = []
    for table in soup.find_all("table"):
        thead = table.find("thead")
        if not thead:
            continue
        cols = [th.text.strip() for th in thead.find_all("th")]
        text = " ".join(cols).lower()
        if not any(k in text for k in ["transaction", "ticker", "asset", "type", "amount"]):
            continue
        tbody = table.find("tbody")
        if not tbody:
            continue
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            row = {cols[i] if i < len(cols) else f"col_{i}": td.text.strip()
                   for i, td in enumerate(cells)}
            if row:
                trades.append(row)
    return trades


def fetch_senate_trades(start_date="01/01/2023 00:00:00") -> pd.DataFrame:
    """End-to-end scrape. Writes data/senate_trades.csv."""
    s = _new_session()
    reports = _list_reports(s, start_date=start_date)
    print(f"[load_senate] {len(reports)} PTRs found")
    all_trades = []
    for i, rep in enumerate(reports):
        if not rep["report_link"]:
            continue
        try:
            for txn in _parse_one_report(s, rep["report_link"]):
                txn["senator"] = rep["senator"]
                txn["date_filed"] = rep["date_filed"]
                txn["report_url"] = BASE + rep["report_link"]
                all_trades.append(txn)
        except Exception as e:
            print(f"[load_senate] parse failed for {rep['senator']}: {e}")
        time.sleep(0.5)
        if (i + 1) % 25 == 0:
            print(f"[load_senate]   {i+1}/{len(reports)} done, {len(all_trades)} trades")
    df = pd.DataFrame(all_trades)
    if "Transaction Date" in df.columns:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].replace("--", pd.NA)
    out = DATA_DIR / "senate_trades.csv"
    df.to_csv(out, index=False)
    print(f"[load_senate] wrote {out} rows={len(df)}")
    return df


def _amount_midpoint(bucket):
    if not isinstance(bucket, str):
        return None
    for k, v in PTR_AMOUNT_BUCKETS.items():
        if k.lower() in bucket.lower():
            return float(v)
    return None


def normalize_senate_to_canonical():
    """Map Senate EFD column names to the canonical PTR schema used downstream.

    Canonical columns:
      filer_name, filer_state_district, asset_name, ticker, transaction_type,
      transaction_date, amount_bucket, amount_midpoint, notification_date, year
    """
    src = DATA_DIR / "senate_trades.csv"
    if not src.exists():
        print(f"[load_senate] {src} missing; run fetch_senate_trades first")
        return None
    raw = pd.read_csv(src)

    def col(*names):
        for n in names:
            if n in raw.columns:
                return raw[n]
        return pd.Series([None] * len(raw))

    out = pd.DataFrame({
        "filer_name": raw["senator"] if "senator" in raw.columns else None,
        "filer_state_district": None,
        "asset_name": col("Asset Name", "Asset", "asset"),
        "ticker": col("Ticker", "ticker"),
        "transaction_type": col("Type", "Transaction Type"),
        "transaction_date": col("Transaction Date", "transaction_date", "Date"),
        "amount_bucket": col("Amount", "Amount Range", "amount"),
        "notification_date": None,
    })
    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")
    out = out.dropna(subset=["transaction_date"])
    out["amount_midpoint"] = out["amount_bucket"].apply(_amount_midpoint)
    out["year"] = out["transaction_date"].dt.year
    out["transaction_date"] = out["transaction_date"].dt.strftime("%Y-%m-%d")
    path = DATA_DIR / "ptr_trades.csv"
    out.to_csv(path, index=False)
    print(f"[load_senate] wrote {path} rows={len(out)}")
    return out


def main():
    fetch_senate_trades()
    normalize_senate_to_canonical()


if __name__ == "__main__":
    main()
