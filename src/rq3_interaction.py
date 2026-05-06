"""
rq3_interaction.py
Does committee alignment add predictive power?

Split the buy-sell signal into aligned vs unaligned based on whether
the senator sits on a committee with jurisdiction over that sector.
Then run OLS with both signals and F-test whether the coefficients differ.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import (
    SECTORS, SECTOR_NAMES, SECTOR_COMMITTEES,
    DATA_DIR, RESULTS_DIR, HAC_BASE_LAGS,
    RQ3_RESULTS_FILE,
)

TAGGED_FILE = DATA_DIR / "ptr_trades_tagged.csv"
MARKET_FILE = DATA_DIR / "market.csv"


def _is_aligned(committees_str, sector):
    if not committees_str or sector not in SECTOR_COMMITTEES:
        return False
    low = (committees_str or "").lower()
    return any(kw.lower() in low for kw in SECTOR_COMMITTEES[sector])


# AI generated: committee alignment interaction test with F-test on coefficient equality
def run_rq3(market, trades, save=True):
    weekly = market.resample("W-FRI").last()
    rets   = weekly.pct_change()

    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "sector"])
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    df["dir"]  = (df["transaction_type"].str.lower()
                    .str.contains("purchase|buy", na=False).astype(int)
                  - df["transaction_type"].str.lower()
                    .str.contains("sale|sell", na=False).astype(int))
    df["aligned"] = df.apply(
        lambda r: _is_aligned(r.get("senator_committees", ""), r["sector"]), axis=1
    )

    results = []
    for sector, cfg in SECTORS.items():
        etf = cfg["etf"]
        if etf not in rets.columns:
            continue

        aligned_sig = (df[(df["sector"]==sector) & df["aligned"]]
                       .groupby("week")["dir"].sum().rename("aligned"))
        unaligned_sig = (df[(df["sector"]==sector) & ~df["aligned"]]
                         .groupby("week")["dir"].sum().rename("unaligned"))

        merged = pd.concat([aligned_sig, unaligned_sig], axis=1).fillna(0)
        merged.index = pd.to_datetime(merged.index)
        fwd = rets[etf].shift(-1).rename("fwd_ret")
        merged = merged.join(fwd, how="inner").dropna()

        if len(merged) < 15:
            continue

        X = sm.add_constant(merged[["aligned","unaligned"]])
        # AI generated: HAC regression and F-test boilerplate
        m = sm.OLS(merged["fwd_ret"], X).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_BASE_LAGS}
        )
        try:
            fp = float(m.f_test("aligned = unaligned").pvalue)
        except Exception:
            fp = np.nan

        results.append({
            "sector":         sector,
            "n":              int(m.nobs),
            "aligned_coef":   round(m.params.get("aligned", np.nan), 6),
            "aligned_p":      round(m.pvalues.get("aligned", np.nan), 4),
            "unaligned_coef": round(m.params.get("unaligned", np.nan), 6),
            "unaligned_p":    round(m.pvalues.get("unaligned", np.nan), 4),
            "f_test_p":       round(fp, 4) if not np.isnan(fp) else np.nan,
        })

    out = pd.DataFrame(results)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(RQ3_RESULTS_FILE, index=False)
    return out


def main():
    if not TAGGED_FILE.exists() or not MARKET_FILE.exists():
        print("[rq3] missing input files")
        return None
    trades = pd.read_csv(TAGGED_FILE)
    date_col = next((c for c in trades.columns if "date" in c.lower()), None)
    if date_col:
        trades["date"] = pd.to_datetime(trades[date_col], errors="coerce")
    market = pd.read_csv(MARKET_FILE, index_col=0, parse_dates=True)
    df = run_rq3(market, trades)
    print(f"[rq3] {len(df)} sectors tested")
    return df


if __name__ == "__main__":
    main()
