"""
fetch_market_data.py
--------------------
Downloads daily OHLCV data for the five sector ETFs and VIX from Yahoo Finance
and writes one CSV per ticker into data/market/.

Tickers
-------
  ITA  – iShares U.S. Aerospace & Defense ETF  (defense)
  XLF  – Financial Select Sector SPDR           (finance)
  XLE  – Energy Select Sector SPDR              (energy)
  XLV  – Health Care Select Sector SPDR         (healthcare)
  XLK  – Technology Select Sector SPDR          (tech)
  ^VIX – CBOE Volatility Index

Usage
-----
  python src/fetch_market_data.py
  python src/fetch_market_data.py --start 2020-01-01 --end 2024-12-31
"""

import argparse
import os
import sys
import logging
from datetime import datetime, date

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKERS: dict[str, str] = {
    "ITA":  "defense",
    "XLF":  "finance",
    "XLE":  "energy",
    "XLV":  "healthcare",
    "XLK":  "tech",
    "^VIX": "vix",
}

DEFAULT_START = "2019-01-01"
DEFAULT_END   = date.today().isoformat()
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    output_dir: str,
) -> pd.DataFrame | None:
    """
    Download daily OHLCV for *ticker* between *start* and *end* (inclusive).

    Returns the cleaned DataFrame, or None if the download fails or returns
    empty data.  Saves to <output_dir>/<clean_ticker>.csv.
    """
    log.info("Fetching %s (%s to %s)", ticker, start, end)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,   # adjusts for splits/dividends
            progress=False,
        )
    except Exception as exc:
        log.error("yfinance download failed for %s: %s", ticker, exc)
        return None

    if df.empty:
        log.warning("No data returned for %s", ticker)
        return None

    # yfinance returns a MultiIndex when multiple tickers are passed; single
    # ticker returns a flat DataFrame, so this is always safe.
    df.index.name = "date"
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Drop rows where close is NaN (market holidays that sneak through)
    df = df.dropna(subset=["close"])

    log.info("  %s rows for %s", len(df), ticker)

    os.makedirs(output_dir, exist_ok=True)
    clean_name = ticker.replace("^", "")          # ^VIX -> VIX
    out_path   = os.path.join(output_dir, f"{clean_name}.csv")
    df.to_csv(out_path)
    log.info("  Saved -> %s", out_path)

    return df


def fetch_all(
    start: str = DEFAULT_START,
    end: str   = DEFAULT_END,
    output_dir: str = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all tickers in TICKERS and return a dict of {ticker: DataFrame}.
    Tickers that fail are omitted from the returned dict.
    """
    results: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        df = fetch_ticker(ticker, start=start, end=end, output_dir=output_dir)
        if df is not None:
            results[ticker] = df
    return results


def load_combined(output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Read all CSVs from *output_dir*, keep only the 'close' column from each,
    and return a single wide DataFrame indexed by date.

    Useful downstream for building the Political Heat Score feature matrix.
    """
    frames: dict[str, pd.Series] = {}
    for ticker, sector in TICKERS.items():
        clean_name = ticker.replace("^", "")
        path = os.path.join(output_dir, f"{clean_name}.csv")
        if not os.path.exists(path):
            log.warning("Missing file for %s – run fetch_all() first", ticker)
            continue
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        frames[clean_name] = df["close"].rename(clean_name)

    if not frames:
        raise FileNotFoundError(
            f"No market CSVs found in {output_dir}. Run fetch_all() first."
        )

    combined = pd.concat(frames.values(), axis=1).sort_index()
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download sector ETF + VIX data from Yahoo Finance."
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory for CSVs (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Validate dates
    for label, val in [("--start", args.start), ("--end", args.end)]:
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            log.error("%s must be in YYYY-MM-DD format, got: %s", label, val)
            sys.exit(1)

    results = fetch_all(
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )

    if not results:
        log.error("All downloads failed. Check your internet connection.")
        sys.exit(1)

    log.info(
        "Done. Fetched %d/%d tickers.", len(results), len(TICKERS)
    )
