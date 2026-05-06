"""RQ2 / sensitivity: alternative signal specifications.

The deck reports 10 sensitivity specifications using simpler signals that
don't z-score. The most defensible variant is the one that produced the
healthcare 20-day hit (p = 0.03) and then failed an out-of-sample replication
(2nd-half p = 0.46). This module reproduces both.

Three signal variants:
    1. Net-buy count: weekly (#buys - #sells) per sector.
    2. Dollar-weighted net flow: same idea, weighted by amount_midpoint.
    3. Raw trade count: total tagged trades per sector (no direction).

For each variant and each sector, regress 20-day forward return on the lagged
signal. Then split the sample at the midpoint and re-run on each half to
check whether the relationship replicates.

Inputs:
  - data/market.csv
  - data/ptr_trades_tagged.csv
Output:
  - results/rq2_sensitivity.csv
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, SECTORS, SECTOR_NAMES, SIGNAL_FREQUENCY


DIRECTION = {"Purchase": 1, "Sale (Full)": -1, "Sale (Partial)": -1,
             "Exchange": 0, "P": 1, "S": -1, "S (partial)": -1, "E": 0}


def _build_weekly_signals(tagged: pd.DataFrame) -> dict:
    df = tagged.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date"])
    df["dir_sign"] = df["transaction_type"].map(DIRECTION).fillna(0)
    df["signed_dollars"] = df["dir_sign"] * df["amount_midpoint"].fillna(0)

    grouper = pd.Grouper(key="transaction_date", freq=SIGNAL_FREQUENCY)
    return {
        "net_buy_count": df.groupby([grouper, "sector"])["dir_sign"].sum().unstack(fill_value=0),
        "dollar_net": df.groupby([grouper, "sector"])["signed_dollars"].sum().unstack(fill_value=0),
        "trade_count": df.groupby([grouper, "sector"]).size().unstack(fill_value=0),
    }


def _regress(signal: pd.Series, prices: pd.Series, h: int):
    import statsmodels.api as sm
    aligned = signal.reindex(prices.index, method="ffill").shift(1)  # 1-day lag
    fr = prices.shift(-h) / prices - 1
    df = pd.concat([aligned.rename("sig"), fr.rename("y")], axis=1).dropna()
    if len(df) < 50:
        return None
    X = sm.add_constant(df[["sig"]])
    return sm.OLS(df["y"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(int(h * 1.5), 5)},
    )


def run_rq2() -> pd.DataFrame:
    market = pd.read_csv(DATA_DIR / "market.csv", parse_dates=["Date"], index_col="Date")
    tagged_path = DATA_DIR / "ptr_trades_tagged.csv"
    if not tagged_path.exists():
        print("[rq2] no tagged trades; skipping")
        return pd.DataFrame()

    tagged = pd.read_csv(tagged_path)
    signals = _build_weekly_signals(tagged)

    rows = []
    for spec_name, wide in signals.items():
        for sector in SECTOR_NAMES:
            if sector not in wide.columns:
                continue
            etf = SECTORS[sector]["etf"]
            if etf not in market.columns:
                continue
            prices = market[etf].dropna()
            sig = wide[sector]

            # Full sample, 20-day forward return
            full = _regress(sig, prices, 20)
            if full is None:
                continue
            row = {
                "spec": spec_name,
                "sector": sector,
                "outcome": "fwd_ret_20d",
                "scope": "full",
                "n": int(full.nobs),
                "coef": float(full.params["sig"]),
                "t": float(full.tvalues["sig"]),
                "p": float(full.pvalues["sig"]),
                "r2": float(full.rsquared),
            }
            rows.append(row)

            # Out-of-sample replication: split daily-aligned series at midpoint
            aligned = sig.reindex(prices.index, method="ffill").shift(1)
            fr = prices.shift(-20) / prices - 1
            df = pd.concat([aligned.rename("sig"), fr.rename("y")], axis=1).dropna()
            if len(df) < 200:
                continue
            mid = len(df) // 2
            for label, sub in [("first_half", df.iloc[:mid]), ("second_half", df.iloc[mid:])]:
                import statsmodels.api as sm
                X = sm.add_constant(sub[["sig"]])
                m = sm.OLS(sub["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
                rows.append({
                    "spec": spec_name,
                    "sector": sector,
                    "outcome": "fwd_ret_20d",
                    "scope": label,
                    "n": int(m.nobs),
                    "coef": float(m.params["sig"]),
                    "t": float(m.tvalues["sig"]),
                    "p": float(m.pvalues["sig"]),
                    "r2": float(m.rsquared),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(RESULTS_DIR / "rq2_sensitivity.csv", index=False)
        full = out[out["scope"] == "full"]
        n_sig = int((full["p"] < 0.05).sum())
        print(f"[rq2] wrote rq2_sensitivity.csv rows={len(out)} "
              f"full-sample sig@p<.05: {n_sig}/{len(full)}")
    return out


# AI generated: sensitivity specification loop across alternative signal constructions
def main():
    run_rq2()


if __name__ == "__main__":
    main()
