"""
structural_break.py
Test whether the heat score's predictive coefficient is stable over time.

Methods:
  - Andrews (1993) SupF test: find break date that maximizes Chow F-stat
  - CUSUM test (Brown et al. 1975): recursive residuals
  - Rolling OLS coefficient for visual inspection
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from config import (
    SECTORS, SECTOR_NAMES, DATA_DIR, RESULTS_DIR,
    HAC_BASE_LAGS, HEAT_SCORE_FILE,
)

MARKET_FILE = DATA_DIR / "market.csv"

TRIM = 0.15   # exclude outer 15% from break date search (Andrews recommendation)
ROLLING_WINDOW = 60

# Andrews (1993) Table 1 critical values for k=2 regressors
_SUPF_CRIT = {0.10: 7.17, 0.05: 9.10, 0.01: 13.43}

# Events to annotate on plots
_EVENTS = {
    "COVID trading\nscandal": "2020-03-01",
    "ETHICS Act\nproposed":   "2022-01-15",
}


# AI generated: Chow F-statistic for structural break detection
def _chow_f_stat(y, X, k):
    """Chow F-statistic for a break at row k."""
    p = X.shape[1]
    n = len(y)
    if k < p + 5 or (n - k) < p + 5:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rss_full = sm.OLS(y, X).fit().ssr
        rss_sub  = sm.OLS(y.iloc[:k], X.iloc[:k]).fit().ssr + \
                   sm.OLS(y.iloc[k:], X.iloc[k:]).fit().ssr
    denom = rss_sub / (n - 2 * p)
    return np.nan if denom <= 0 else ((rss_full - rss_sub) / p) / denom


# AI generated: Andrews (1993) SupF test grid-search over candidate break dates
def detect_break_supf(y, X, trim=TRIM):
    n  = len(y)
    lo = int(np.ceil(trim * n))
    hi = int(np.floor((1 - trim) * n))
    if hi - lo < 5:
        return {"break_date": None, "break_idx": None, "supf_stat": np.nan,
                "significant": False, "f_by_date": pd.Series(dtype=float)}

    f_stats = {y.index[k]: _chow_f_stat(y, X, k)
               for k in range(lo, hi + 1)}
    f_series = pd.Series({d: v for d, v in f_stats.items() if not np.isnan(v)})
    if f_series.empty:
        return {"break_date": None, "break_idx": None, "supf_stat": np.nan,
                "significant": False, "f_by_date": f_series}

    supf       = f_series.max()
    break_date = f_series.idxmax()
    break_idx  = list(y.index).index(break_date)
    return {
        "break_date":  break_date,
        "break_idx":   break_idx,
        "supf_stat":   supf,
        "significant": supf > _SUPF_CRIT[0.05],
        "f_by_date":   f_series,
    }


# AI generated: Brown-Durbin-Evans CUSUM recursive residual test
def cusum_test(y, X):
    # AI generated: CUSUM recursive residual loop
    p, n = X.shape[1], len(y)
    init = p + 2
    cusum_vals, var_ests = [], []
    for k in range(init, n + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.OLS(y.iloc[:k-1], X.iloc[:k-1]).fit()
        xv  = X.iloc[k-1].values
        XtX = X.iloc[:k-1].values.T @ X.iloc[:k-1].values
        h   = float(xv @ np.linalg.pinv(XtX) @ xv)
        res = (y.iloc[k-1] - m.predict(X.iloc[k-1:k]).iloc[0]) / np.sqrt(max(1+h, 1e-10))
        cusum_vals.append(res)
        var_ests.append(res**2)

    sigma  = np.sqrt(np.mean(var_ests))
    cumsum = np.cumsum(cusum_vals) / (sigma + 1e-10)
    idx    = y.index[init-1:]
    m_n    = n - p
    t_arr  = np.arange(1, len(cumsum)+1)
    upper  = 1.358 * np.sqrt(m_n) + 2 * 1.358 * t_arr / np.sqrt(m_n)

    return {
        "cusum":       pd.Series(cumsum, index=idx),
        "upper_band":  pd.Series(upper, index=idx),
        "lower_band":  pd.Series(-upper, index=idx),
        "breaks_band": bool(np.any(np.abs(cumsum) > upper)),
    }


def rolling_coefficient(y, X, window=ROLLING_WINDOW, col="heat"):
    coefs, idx = [], []
    for end in range(window, len(y)+1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.OLS(y.iloc[end-window:end], X.iloc[end-window:end]).fit()
        coefs.append(m.params.get(col, np.nan))
        idx.append(y.index[end-1])
    return pd.Series(coefs, index=idx)


def run_structural_break(prices, heat, horizon_days=5, save=True):
    weekly = prices.resample("W-FRI").last()
    rets   = weekly.pct_change()
    wks    = max(1, round(horizon_days / 5))

    results = []
    for sector, cfg in SECTORS.items():
        etf = cfg["etf"]
        if etf not in rets.columns or sector not in heat.columns:
            continue
        df = pd.DataFrame({
            "heat":    heat[sector],
            "fwd_ret": rets[etf].shift(-wks).reindex(heat.index),
            "lag_ret": rets[etf].shift(1).reindex(heat.index),
        }).dropna()
        if len(df) < 40:
            continue

        y = df["fwd_ret"]
        X = sm.add_constant(df[["heat","lag_ret"]])

        supf  = detect_break_supf(y, X)
        cusum = cusum_test(y, X)
        roll  = rolling_coefficient(y, X, window=min(ROLLING_WINDOW, len(df)//2))

        pre, post = np.nan, np.nan
        if supf["break_idx"] is not None:
            k = supf["break_idx"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if k > X.shape[1]+2:
                    pre  = sm.OLS(y.iloc[:k], X.iloc[:k]).fit().params.get("heat", np.nan)
                if (len(y)-k) > X.shape[1]+2:
                    post = sm.OLS(y.iloc[k:], X.iloc[k:]).fit().params.get("heat", np.nan)

        if save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            roll.to_csv(RESULTS_DIR / f"rolling_coef_{sector}.csv", header=["coef"])
            supf["f_by_date"].to_csv(RESULTS_DIR / f"supf_by_date_{sector}.csv", header=["f_stat"])
            pd.DataFrame({
                "cusum": cusum["cusum"],
                "upper": cusum["upper_band"],
                "lower": cusum["lower_band"],
            }).to_csv(RESULTS_DIR / f"cusum_{sector}.csv")

        results.append({
            "sector":                sector,
            "n_obs":                 len(df),
            "break_date":            supf["break_date"],
            "supf_stat":             round(supf["supf_stat"], 3) if not np.isnan(supf["supf_stat"]) else np.nan,
            "supf_significant_5pct": supf["significant"],
            "pre_break_heat_coef":   round(pre, 6)  if not np.isnan(pre)  else np.nan,
            "post_break_heat_coef":  round(post, 6) if not np.isnan(post) else np.nan,
            "cusum_breaks_band":     cusum["breaks_band"],
        })

    return pd.DataFrame(results)


def main():
    if not HEAT_SCORE_FILE.exists() or not MARKET_FILE.exists():
        print("[break] missing input files")
        return None
    heat   = pd.read_csv(HEAT_SCORE_FILE, index_col=0, parse_dates=True)
    market = pd.read_csv(MARKET_FILE, index_col=0, parse_dates=True)
    df = run_structural_break(market, heat)
    for _, r in df.iterrows():
        flag = "*** BREAK ***" if r["supf_significant_5pct"] else ""
        bd = str(r["break_date"])[:10] if r["break_date"] else "none"
        print(f"[break] {r['sector']}: date={bd}, SupF={r['supf_stat']} {flag}")
    return df


if __name__ == "__main__":
    main()
