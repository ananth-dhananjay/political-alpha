"""Yahoo Finance data ingestion.

Pulls daily closing prices for:
  - The five sector ETFs (ITA, XLF, XLE, XLV, XLK)
  - VIX (^VIX) for volatility context
  - SPY for the market-factor benchmark
  - ^IRX (13-week T-bill yield) for the risk-free rate

Outputs:
  - data/market.csv      Date index, columns = sector ETFs + ^VIX
  - data/benchmarks.csv  Date index, columns = SPY, IRX
"""

from __future__ import annotations
import warnings
import pandas as pd

from config import DATA_DIR, START_DATE, END_DATE, SECTOR_ETFS

warnings.filterwarnings("ignore", category=FutureWarning)


def fetch_market() -> pd.DataFrame:
    """Sector ETFs + VIX. Returns a wide DataFrame; also writes data/market.csv."""
    import yfinance as yf
    tickers = SECTOR_ETFS + ["^VIX"]
    print(f"[load_market] fetching {tickers}")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.concat(
            {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)},
            axis=1,
        )
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all")
    close.index.name = "Date"
    out = DATA_DIR / "market.csv"
    close.to_csv(out)
    print(f"[load_market] wrote {out} shape={close.shape}")
    return close


def fetch_benchmarks() -> pd.DataFrame:
    """SPY + ^IRX for CAPM. Writes data/benchmarks.csv."""
    import yfinance as yf
    tickers = ["SPY", "^IRX"]
    print(f"[load_market] fetching benchmarks {tickers}")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        out = pd.DataFrame({
            "SPY": raw["SPY"]["Close"],
            "IRX": raw["^IRX"]["Close"],
        })
    else:
        out = raw[["Close"]]
    out = out.dropna(how="all")
    out.index.name = "Date"
    path = DATA_DIR / "benchmarks.csv"
    out.to_csv(path)
    print(f"[load_market] wrote {path} shape={out.shape}")
    return out


def main():
    fetch_market()
    fetch_benchmarks()


if __name__ == "__main__":
    main()
