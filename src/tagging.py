"""Sector tagging.

Two-step rule:
  1. If the trade has a recognized ticker, look it up in the per-sector ticker
     whitelist (8 tickers per sector defined in config.SECTORS).
  2. Otherwise, score the asset_name string against the keyword dictionary;
     the sector with the most keyword matches wins. Ties broken by the order
     in config.SECTOR_NAMES.

Trades that match neither are dropped. In the 2023-2026 sample, this yields
roughly 18% tagged (419 of 2,284), driven mostly by ticker matches.

Inputs:
  - data/ptr_trades.csv
Output:
  - data/ptr_trades_tagged.csv  with an added `sector` column
"""

from __future__ import annotations
from typing import Optional
import pandas as pd

from config import DATA_DIR, SECTORS, SECTOR_NAMES


def infer_sector_from_ticker(ticker) -> Optional[str]:
    if ticker is None or not isinstance(ticker, str) or not ticker.strip():
        return None
    t = ticker.upper().strip()
    for sector, cfg in SECTORS.items():
        if t in cfg["tickers"]:
            return sector
    return None


def infer_sector_from_text(text) -> Optional[str]:
    if text is None or not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    scores = {}
    for sector, cfg in SECTORS.items():
        hits = sum(1 for kw in cfg["keywords"] if kw.lower() in t)
        if hits:
            scores[sector] = hits
    if not scores:
        return None
    # Tie break by config ordering
    best = max(scores.values())
    for s in SECTOR_NAMES:
        if scores.get(s) == best:
            return s
    return None


def tag_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Add `sector` to a trades DataFrame; drop rows that don't match."""
    if trades.empty:
        return trades.assign(sector=[])
    out = trades.copy()
    sectors = []
    for _, row in out.iterrows():
        s = infer_sector_from_ticker(row.get("ticker"))
        if not s:
            s = infer_sector_from_text(row.get("asset_name", ""))
        sectors.append(s)
    out["sector"] = sectors
    return out.dropna(subset=["sector"])


def main():
    src = DATA_DIR / "ptr_trades.csv"
    if not src.exists():
        print(f"[tagging] {src} missing; run load_senate.py first")
        return
    trades = pd.read_csv(src)
    tagged = tag_trades(trades)
    out = DATA_DIR / "ptr_trades_tagged.csv"
    tagged.to_csv(out, index=False)
    print(f"[tagging] wrote {out} rows={len(tagged)} "
          f"({len(tagged)/max(len(trades),1)*100:.1f}% of {len(trades)})")
    if not tagged.empty:
        print("[tagging] by sector:")
        print(tagged["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
