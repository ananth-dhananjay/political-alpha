"""RQ1: primary statistical inference.

Specification:
    For each sector s and forward-return horizon h in {1, 4, 12} weeks:
        r_w(t, t+h) = alpha + beta * heat(s, t-1) + epsilon(t)
    Plus a 4-week forward realized-vol regression:
        rv_4w(t+1, t+4) = alpha + beta * heat(s, t-1) + epsilon(t)

Weekly frequency throughout. Each observation is one week, so the sample
size in each regression equals the number of weeks of data available, not
the number of trading days. This is the honest sample size.

Standard errors are Newey-West (HAC) with max(h, 4) lags to handle serial
correlation from overlapping multi-week forward-return windows.

Total: 5 sectors x 4 outcomes = 20 specifications. Some specifications drop
out for sectors with sparse heat coverage (defense in particular has only 9
sector-tagged trades in the bundled sample).

Inputs:
  - data/market.csv
  - data/heat_score.csv
Output:
  - results/rq1_inference.csv
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, RESULTS_DIR, SECTORS, SECTOR_NAMES, SIGNAL_LAG_WEEKS,
)


# Weekly horizons. The deck quoted daily horizons (1/5/20 trading days);
# the bundled signal is weekly so we adopt weekly horizons for the canonical
# pipeline (same lookforward in calendar terms: ~1 week, ~1 month, ~3 months).
HORIZONS_WEEKS = [1, 4, 12]
VOL_HORIZON_WEEKS = 4


def _load_market_and_heat():
    m = pd.read_csv(DATA_DIR / "market.csv", parse_dates=["Date"], index_col="Date")
    h = pd.read_csv(DATA_DIR / "heat_score.csv", parse_dates=["date"])
    return m, h


def _heat_signal_for(heat_long: pd.DataFrame, sector: str, prefer="trade_z"):
    """Pick the column to use as the heat signal.

    The composite `heat` column requires both trade and bill components to be
    non-null. In the bundled data, bill coverage is poor and the composite is
    sparse (~20 weeks per sector). The `trade_z` column has ~98 weeks per
    sector and is the more usable signal. Default to it; fall back to `heat`
    if `trade_z` is missing entirely.
    """
    sdf = heat_long[heat_long["sector"] == sector].sort_values("date").set_index("date")
    if prefer in sdf.columns and sdf[prefer].notna().sum() >= 30:
        return sdf[prefer]
    if "heat" in sdf.columns:
        return sdf["heat"]
    return None


def run_rq1() -> pd.DataFrame:
    import statsmodels.api as sm

    market, heat = _load_market_and_heat()
    weekly_market = market.resample("W-FRI").last()

    rows = []
    for sector in SECTOR_NAMES:
        etf = SECTORS[sector]["etf"]
        if etf not in weekly_market.columns:
            continue
        sig = _heat_signal_for(heat, sector)
        if sig is None or sig.notna().sum() < 30:
            continue
        weekly_prices = weekly_market[etf]

        for h in HORIZONS_WEEKS:
            fr = weekly_prices.shift(-h) / weekly_prices - 1
            df = pd.concat([sig.shift(SIGNAL_LAG_WEEKS).rename("heat"),
                            fr.rename("y")], axis=1).dropna()
            if len(df) < 30:
                continue
            X = sm.add_constant(df[["heat"]])
            model = sm.OLS(df["y"], X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(h, 4)},
            )
            rows.append({
                "sector": sector,
                "outcome": f"fwd_ret_{h}w",
                "n": int(model.nobs),
                "coef_heat": float(model.params["heat"]),
                "t_heat": float(model.tvalues["heat"]),
                "p_heat": float(model.pvalues["heat"]),
                "r2": float(model.rsquared),
            })

        # 4-week forward realized vol
        weekly_returns = weekly_prices.pct_change()
        rv = weekly_returns.rolling(VOL_HORIZON_WEEKS).std() * np.sqrt(52)
        fv = rv.shift(-VOL_HORIZON_WEEKS)
        df = pd.concat([sig.shift(SIGNAL_LAG_WEEKS).rename("heat"),
                        fv.rename("y")], axis=1).dropna()
        if len(df) >= 30:
            X = sm.add_constant(df[["heat"]])
            model = sm.OLS(df["y"], X).fit(
                cov_type="HAC", cov_kwds={"maxlags": VOL_HORIZON_WEEKS},
            )
            rows.append({
                "sector": sector,
                "outcome": f"fwd_vol_{VOL_HORIZON_WEEKS}w",
                "n": int(model.nobs),
                "coef_heat": float(model.params["heat"]),
                "t_heat": float(model.tvalues["heat"]),
                "p_heat": float(model.pvalues["heat"]),
                "r2": float(model.rsquared),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(RESULTS_DIR / "rq1_inference.csv", index=False)
        n_sig_05 = int((out["p_heat"] < 0.05).sum())
        n_sig_10 = int((out["p_heat"] < 0.10).sum())
        print(f"[rq1] wrote rq1_inference.csv rows={len(out)} "
              f"sig@p<.05: {n_sig_05} | sig@p<.10: {n_sig_10}")
    return out


# AI generated: OLS regression loop with Newey-West HAC standard errors
def main():
    run_rq1()


if __name__ == "__main__":
    main()
