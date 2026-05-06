"""Congress.gov legislative-activity ingestion.

Pulls bill metadata from the Congress.gov v3 REST API and tags each bill to
one of the five sectors using a keyword match on the title.

Output:
  - data/bills_all.csv     One row per bill with sector, sector_pretty,
                           introduced_date, latest_action_date, etc.

Note (per challenges slide in deck): the Congress.gov defaults return mostly
older housekeeping bills, so the legislative half of the heat score has
poor coverage in the 2023-2026 trade window. This module pulls the data
faithfully; the consequences are documented downstream.
"""

from __future__ import annotations
import time
import pandas as pd
import requests

from config import DATA_DIR, SECTORS, CONGRESS_API_KEY


SECTOR_KEYWORDS = {
    "defense":    ["defense", "armed services", "military", "aerospace", "homeland"],
    "finance":    ["bank", "financial", "credit", "securities", "insurance", "budget"],
    "energy":     ["energy", "oil", "gas", "pipeline", "renewable", "climate"],
    "healthcare": ["health", "medicare", "medicaid", "pharmaceutical", "drug", "hospital"],
    "tech":       ["technology", "semiconductor", "artificial intelligence", "cyber", "broadband"],
}


def _tag_sector(title: str):
    if not isinstance(title, str):
        return None
    text = title.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(k in text for k in keywords):
            return sector
    return None


def fetch_bills(start="2020-01-01T00:00:00Z", end="2025-12-31T00:00:00Z",
                limit=250, max_records=5000):
    """Paginated pull. Writes data/bills_all.csv."""
    if not CONGRESS_API_KEY:
        raise RuntimeError(
            "CONGRESS_API_KEY missing. Add it to src/.env "
            "(see src/.env.example). Get a free key at api.congress.gov/sign-up."
        )

    base = f"{CONGRESS_API_BASE_URL}/bill"
    records, offset = [], 0
    while offset < max_records:
        params = {
            "api_key": CONGRESS_API_KEY,
            "fromDateTime": start, "toDateTime": end,
            "limit": limit, "offset": offset,
            "sort": "updateDate+desc",
        }
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        bills = r.json().get("bills", [])
        if not bills:
            break
        records.extend(bills)
        offset += limit
        time.sleep(0.5)

    df = pd.json_normalize(records)
    keep = ["number", "title", "type", "updateDate",
            "latestAction.actionDate", "latestAction.text"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(columns={
        "number": "BillNumber",
        "title": "Title",
        "type": "Type",
        "updateDate": "UpdateDate",
        "latestAction.actionDate": "LatestActionDate",
        "latestAction.text": "LatestActionText",
    })
    df["UpdateDate"] = pd.to_datetime(df["UpdateDate"], errors="coerce")
    df["LatestActionDate"] = pd.to_datetime(df["LatestActionDate"], errors="coerce")

    sector_pretty = df["Title"].apply(_tag_sector)
    out = pd.DataFrame({
        "congress": None,
        "bill_type": df["Type"],
        "number": df["BillNumber"],
        "title": df["Title"],
        "introduced_date": df["UpdateDate"],
        "latest_action_date": df["LatestActionDate"],
        "origin_chamber": None,
        "sector_pretty": sector_pretty,
        "sector": sector_pretty,  # already lowercase keys
    })
    path = DATA_DIR / "bills_all.csv"
    out.to_csv(path, index=False)
    tagged = out["sector"].notna().sum()
    print(f"[load_bills] wrote {path} rows={len(out)} tagged={tagged} "
          f"({tagged/max(len(out),1)*100:.1f}%)")
    return out


def main():
    fetch_bills()


if __name__ == "__main__":
    main()
