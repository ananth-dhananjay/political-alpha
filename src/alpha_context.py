"""Alpha context module.

Constructs a weekly long-short portfolio from the heat score (long top-2 by
heat, short bottom-2), then regresses excess returns on the market factor
(CAPM) and the Fama-French 3 factors. Produces:
  - portfolio summary statistics (Sharpe, drawdown, cumulative return)
  - CAPM alpha + beta
  - FF3 alpha + factor loadings

The deck does not include an alpha slide, but the underlying numbers are
useful context for the descriptive results and the methodology reflection.

Inputs:
  - data/market.csv
  - data/heat_score.csv
  - data/benchmarks.csv
  - data/ff3_daily.csv
Output:
  - results/alpha_summary.csv
  - results/portfolio_returns.csv
  - results/portfolio_weights.csv
  - results/alpha_report.txt
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, RESULTS_DIR, SECTORS, SECTOR_NAMES,
    PORTFOLIO_N_LONG, PORTFOLIO_N_SHORT, PERIODS_PER_YEAR, HAC_BASE_LAGS,
    SIGNAL_LAG_WEEKS,
)


SECTOR_TO_ETF = {s: SECTORS[s]["etf"] for s in SECTOR_NAMES}
ETF_TO_SECTOR = {v: k for k, v in SECTOR_TO_ETF.items()}


def build_weights(heat_long: pd.DataFrame) -> pd.DataFrame:
    """Long top-N heat sectors, short bottom-N. Equal-weight legs, dollar-neutral."""
    if heat_long.empty:
        return pd.DataFrame()
    wide = (heat_long.pivot(index="date", columns="sector", values="heat")
            .sort_index().reindex(columns=SECTOR_NAMES))
    wide = wide.shift(SIGNAL_LAG_WEEKS)

    weights = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    for dt, row in wide.iterrows():
        vals = row.dropna()
        if len(vals) < (PORTFOLIO_N_LONG + PORTFOLIO_N_SHORT):
            continue
        ranked = vals.sort_values()
        weights.loc[dt, ranked.head(PORTFOLIO_N_SHORT).index] = -1.0 / PORTFOLIO_N_SHORT
        weights.loc[dt, ranked.tail(PORTFOLIO_N_LONG).index] = 1.0 / PORTFOLIO_N_LONG
    return weights[(weights != 0).any(axis=1)]


# AI generated: weekly long-short portfolio return construction
def portfolio_returns(weights: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    etf_cols = [SECTOR_TO_ETF[s] for s in SECTOR_NAMES if SECTOR_TO_ETF[s] in market.columns]
    weekly_prices = market[etf_cols].resample("W-FRI").last()
    weekly_returns = weekly_prices.pct_change().rename(columns=ETF_TO_SECTOR)
    w = weights.reindex(weekly_returns.index).ffill()
    pr = (weekly_returns * w).sum(axis=1, skipna=True)
    return pr.where(w.notna().any(axis=1)).dropna()


def _capm_regression(port: pd.Series, bench: pd.DataFrame):
    import statsmodels.api as sm
    df = pd.concat([port.rename("r"), bench[["mkt_excess", "rf_weekly"]]], axis=1).dropna()
    if len(df) < 20:
        return None
    df["excess"] = df["r"] - df["rf_weekly"]
    X = sm.add_constant(df[["mkt_excess"]])
    return sm.OLS(df["excess"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_BASE_LAGS})


def _ff3_regression(port: pd.Series, ff3w: pd.DataFrame):
    import statsmodels.api as sm
    need = ["Mkt-RF", "SMB", "HML", "RF"]
    if any(c not in ff3w.columns for c in need):
        return None
    df = pd.concat([port.rename("r"), ff3w[need]], axis=1).dropna()
    if len(df) < 20:
        return None
    df["excess"] = df["r"] - df["RF"]
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    return sm.OLS(df["excess"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_BASE_LAGS})


def _summarize(port: pd.Series) -> dict:
    if port.empty:
        return {}
    mu, sigma = port.mean(), port.std()
    cum = (1 + port).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {
        "n_weeks": int(len(port)),
        "mean_weekly": float(mu),
        "std_weekly": float(sigma),
        "ann_return": float((1 + mu) ** PERIODS_PER_YEAR - 1),
        "ann_vol": float(sigma * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe": float(mu / sigma * np.sqrt(PERIODS_PER_YEAR)) if sigma > 0 else float("nan"),
        "max_drawdown": float(dd),
        "cumulative_return": float(cum.iloc[-1] - 1),
    }


def run_alpha():
    market_path = DATA_DIR / "market.csv"
    heat_path = DATA_DIR / "heat_score.csv"
    bench_path = DATA_DIR / "benchmarks.csv"
    ff3_path = DATA_DIR / "ff3_daily.csv"

    if not market_path.exists() or not heat_path.exists():
        print("[alpha] missing market.csv or heat_score.csv; skip")
        return

    market = pd.read_csv(market_path, parse_dates=["Date"], index_col="Date")
    heat = pd.read_csv(heat_path, parse_dates=["date"])
    weights = build_weights(heat)
    port = portfolio_returns(weights, market)
    stats = _summarize(port)

    summaries = []

    if bench_path.exists():
        b = pd.read_csv(bench_path, parse_dates=["Date"], index_col="Date")
        bw = b.resample("W-FRI").last()
        annual_yield = bw["IRX"] / 100.0
        rf = (1 + annual_yield) ** (1 / PERIODS_PER_YEAR) - 1
        mkt = bw["SPY"].pct_change()
        bench = pd.DataFrame({"mkt_ret": mkt, "rf_weekly": rf,
                              "mkt_excess": mkt - rf})
        m = _capm_regression(port, bench)
        if m is not None:
            aw = float(m.params["const"])
            summaries.append({
                "model": "CAPM",
                "n": int(m.nobs),
                "alpha_weekly": aw,
                "alpha_annual": (1 + aw) ** PERIODS_PER_YEAR - 1,
                "alpha_t": float(m.tvalues["const"]),
                "alpha_p": float(m.pvalues["const"]),
                "beta_mkt": float(m.params["mkt_excess"]),
                "r2": float(m.rsquared),
            })

    if ff3_path.exists():
        ff3 = pd.read_csv(ff3_path, parse_dates=["date"], index_col="date")
        ff3w = ff3.resample("W-FRI").agg(lambda x: (1 + x).prod() - 1)
        m = _ff3_regression(port, ff3w)
        if m is not None:
            aw = float(m.params["const"])
            summaries.append({
                "model": "FF3",
                "n": int(m.nobs),
                "alpha_weekly": aw,
                "alpha_annual": (1 + aw) ** PERIODS_PER_YEAR - 1,
                "alpha_t": float(m.tvalues["const"]),
                "alpha_p": float(m.pvalues["const"]),
                "beta_mkt": float(m.params["Mkt-RF"]),
                "beta_smb": float(m.params["SMB"]),
                "beta_hml": float(m.params["HML"]),
                "r2": float(m.rsquared),
            })

    if summaries:
        pd.DataFrame(summaries).to_csv(RESULTS_DIR / "alpha_summary.csv", index=False)
    if not port.empty:
        port.to_csv(RESULTS_DIR / "portfolio_returns.csv", header=["portfolio_return"])
    if not weights.empty:
        weights.to_csv(RESULTS_DIR / "portfolio_weights.csv")

    # Human-readable report
    lines = [
        "POLITICAL ALPHA - PORTFOLIO REPORT",
        "=" * 60,
        "",
        f"Strategy: long top-{PORTFOLIO_N_LONG} heat, short bottom-{PORTFOLIO_N_SHORT} heat.",
        f"Rebalance weekly. {SIGNAL_LAG_WEEKS}-week heat lag prevents look-ahead.",
        "",
        "Portfolio statistics:",
    ]
    for k, v in stats.items():
        lines.append(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
    lines += ["", "Factor regressions:"]
    for s in summaries:
        lines.append(f"  {s['model']}:")
        for k, v in s.items():
            if k == "model":
                continue
            lines.append(f"    {k:18s} {v:.6f}" if isinstance(v, float) else f"    {k:18s} {v}")
    (RESULTS_DIR / "alpha_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    run_alpha()


if __name__ == "__main__":
    main()
