"""
Political Alpha – Data Loading
Fetches data from Yahoo Finance API, House Clerk, and Congress.gov API.
"""

import os
import json
import time
import requests
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


def fetch_sector_etf_data(tickers, start, end=None, output_dir="../data"):
    """Download daily OHLCV data for sector ETFs and VIX from Yahoo Finance."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    os.makedirs(output_dir, exist_ok=True)
    frames = []

    for label, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            print(f"[WARN] No data returned for {ticker}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["ticker"] = ticker
        df["sector"] = label
        df.index.name = "date"
        frames.append(df)

        out_path = os.path.join(output_dir, f"{label}_{ticker.replace('^', '')}.csv")
        df.to_csv(out_path)
        print(f"[OK] {ticker} ({label}): {len(df)} rows -> {out_path}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined.to_csv(os.path.join(output_dir, "all_sector_etfs.csv"))
    return combined


def fetch_house_disclosures(year, output_dir="../data"):
    """Download House financial disclosure listings for a given year."""
    os.makedirs(output_dir, exist_ok=True)

    url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.txt"
    resp = requests.get(url, timeout=15)

    if resp.status_code != 200:
        print(f"[WARN] House disclosure file not found for {year} (status {resp.status_code})")
        return []

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = [h.strip() for h in lines[0].split("\t")]
    filings = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) >= len(headers):
            filing = {headers[i]: fields[i].strip() for i in range(len(headers))}
            filings.append(filing)

    out_path = os.path.join(output_dir, f"house_disclosures_{year}.json")
    with open(out_path, "w") as f:
        json.dump(filings, f, indent=2)
    print(f"[OK] House disclosures ({year}): {len(filings)} filings -> {out_path}")
    return filings


def fetch_bills(congress, api_key=None, limit=250, output_dir="../data"):
    """Fetch bills from the Congress.gov API."""
    if api_key is None:
        api_key = os.environ.get("CONGRESS_API_KEY", "")
    if not api_key:
        print("[WARN] No Congress.gov API key. Set CONGRESS_API_KEY in src/.env")
        return []

    os.makedirs(output_dir, exist_ok=True)
    bills = []
    offset = 0

    while offset < limit:
        fetch_size = min(250, limit - offset)
        params = {
            "api_key": api_key,
            "limit": fetch_size,
            "offset": offset,
            "format": "json",
        }
        resp = requests.get(
            f"https://api.congress.gov/v3/bill/{congress}",
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[WARN] Congress API returned {resp.status_code}")
            break

        data = resp.json()
        batch = data.get("bills", [])
        if not batch:
            break
        bills.extend(batch)
        offset += len(batch)
        time.sleep(0.5)

    out_path = os.path.join(output_dir, f"bills_{congress}.json")
    with open(out_path, "w") as f:
        json.dump(bills, f, indent=2)
    print(f"[OK] Bills (Congress {congress}): {len(bills)} bills -> {out_path}")
    return bills
