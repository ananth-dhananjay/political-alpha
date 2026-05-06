"""Weekly Political Heat Score construction.

For each sector s and week t:
    x(s, t)    = number of tagged trades in sector s during week t
    heat(s, t) = (x(s, t) - mu_w) / sigma_w

where mu_w and sigma_w are the mean and standard deviation over the
preceding 26-week window. This rolling z-score is what gets used in the
RQ1 regressions in rq1_predictive.py, after a one-week lag is applied
in that module to prevent look-ahead.

Output:
  - data/heat_score.csv   long-form: date, sector, trade_count, bill_count,
                          trade_z, bill_z, heat
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, SECTOR_NAMES, SIGNAL_FREQUENCY, ZSCORE_WINDOW_WEEKS,
)


def _weekly_counts(events: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Pivot to weekly count per sector. Returns wide DF (date x sector)."""
    if events.empty:
        return pd.DataFrame()
    df = events.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    grouped = df.groupby([pd.Grouper(key=date_col, freq=SIGNAL_FREQUENCY),
                          "sector"]).size()
    wide = grouped.unstack(fill_value=0)
    for s in SECTOR_NAMES:
        if s not in wide.columns:
            wide[s] = 0
    return wide[SECTOR_NAMES]


# AI generated: rolling z-score normalization with configurable window
def _rolling_z(df: pd.DataFrame, window: int) -> pd.DataFrame:
    rolling = df.rolling(window, min_periods=max(window // 2, 4))
    mu = rolling.mean()
    sigma = rolling.std().replace(0, np.nan)
    return (df - mu) / sigma


# AI generated: long-form heat score construction from trade and bill counts
def build_heat_score(trade_counts: pd.DataFrame,
                     bill_counts: pd.DataFrame) -> pd.DataFrame:
    """Combine the trade and bill weekly counts into the long-form heat table."""
    idx = trade_counts.index.union(bill_counts.index) if not bill_counts.empty else trade_counts.index
    t = trade_counts.reindex(idx).fillna(0)
    b = bill_counts.reindex(idx).fillna(0) if not bill_counts.empty else \
        pd.DataFrame(0, index=idx, columns=SECTOR_NAMES)

    tz = _rolling_z(t, ZSCORE_WINDOW_WEEKS)
    bz = _rolling_z(b, ZSCORE_WINDOW_WEEKS)
    heat = 0.5 * tz + 0.5 * bz

    rows = []
    for d in idx:
        for s in SECTOR_NAMES:
            rows.append({
                "date": d,
                "sector": s,
                "trade_count": int(t.loc[d, s]) if s in t.columns else 0,
                "bill_count": int(b.loc[d, s]) if s in b.columns else 0,
                "trade_z": tz.loc[d, s] if s in tz.columns else np.nan,
                "bill_z": bz.loc[d, s] if s in bz.columns else np.nan,
                "heat": heat.loc[d, s] if s in heat.columns else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    tagged_path = DATA_DIR / "ptr_trades_tagged.csv"
    bills_path = DATA_DIR / "bills_all.csv"

    if not tagged_path.exists():
        print(f"[signal] {tagged_path} missing; run tagging.py first")
        return

    trades = pd.read_csv(tagged_path)
    bills = pd.read_csv(bills_path) if bills_path.exists() else pd.DataFrame()

    # Ensure bills has a sector column. The loader already tags but enforce
    # consistency.
    if not bills.empty and "sector" in bills.columns:
        bills = bills.dropna(subset=["sector"])

    # find whichever date column the tagged file uses
    date_col = next((c for c in trades.columns if c in ("date","transaction_date","Date")), trades.columns[0])
    trade_counts = _weekly_counts(trades, date_col)
    bill_counts = _weekly_counts(bills, "introduced_date") if not bills.empty else pd.DataFrame()

    heat = build_heat_score(trade_counts, bill_counts)
    out = DATA_DIR / "heat_score.csv"
    heat.to_csv(out, index=False)
    valid = heat["heat"].notna().sum()
    print(f"[signal] wrote {out} rows={len(heat)} non-null heat={valid}")


if __name__ == "__main__":
    main()
