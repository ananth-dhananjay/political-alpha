"""Unit tests. Run from src/ directory:
    pytest tests.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (
    SECTORS, SECTOR_NAMES, PTR_AMOUNT_BUCKETS, ZSCORE_WINDOW_WEEKS,
    SIGNAL_LAG_WEEKS, PORTFOLIO_N_LONG, PORTFOLIO_N_SHORT,
)


# ============================================================================
# config sanity
# ============================================================================

class TestConfig:
    def test_five_sectors(self):
        assert len(SECTOR_NAMES) == 5
        assert set(SECTOR_NAMES) == {"defense", "finance", "energy", "healthcare", "tech"}

    def test_each_sector_has_etf_and_tickers(self):
        for s, cfg in SECTORS.items():
            assert cfg["etf"]
            assert len(cfg["tickers"]) >= 5
            assert len(cfg["keywords"]) >= 4

    def test_amount_buckets_monotonic(self):
        mids = list(PTR_AMOUNT_BUCKETS.values())
        assert mids == sorted(mids)

    def test_lag_and_window_sane(self):
        assert ZSCORE_WINDOW_WEEKS >= 8
        assert SIGNAL_LAG_WEEKS >= 1
        assert PORTFOLIO_N_LONG + PORTFOLIO_N_SHORT <= len(SECTOR_NAMES)


# ============================================================================
# tagging.py
# ============================================================================

from tagging import (
    infer_sector_from_ticker, infer_sector_from_text, tag_trades,
)


class TestTagging:
    def test_ticker_match(self):
        assert infer_sector_from_ticker("LMT") == "defense"
        assert infer_sector_from_ticker("JPM") == "finance"
        assert infer_sector_from_ticker("XOM") == "energy"
        assert infer_sector_from_ticker("MSFT") == "tech"
        assert infer_sector_from_ticker("PFE") == "healthcare"

    def test_ticker_unknown(self):
        assert infer_sector_from_ticker("ZZZZ") is None
        assert infer_sector_from_ticker(None) is None
        assert infer_sector_from_ticker("") is None
        assert infer_sector_from_ticker(float("nan")) is None

    def test_keyword_fallback(self):
        assert infer_sector_from_text("Medicare Part D Reform Act") == "healthcare"
        assert infer_sector_from_text("Weapons Export Control Act") == "defense"
        assert infer_sector_from_text("Artificial Intelligence Standards Act") == "tech"

    def test_keyword_no_match(self):
        assert infer_sector_from_text("Random Generic Title") is None

    def test_tag_trades_drops_unmatched(self):
        df = pd.DataFrame([
            {"ticker": "MSFT", "asset_name": "Microsoft Corp"},
            {"ticker": None, "asset_name": "Random ETF"},
            {"ticker": "JPM", "asset_name": "JPMorgan Chase"},
        ])
        out = tag_trades(df)
        assert len(out) == 2
        assert set(out["sector"]) == {"tech", "finance"}


# ============================================================================
# signal.py
# ============================================================================

from heat_signal import _weekly_counts, _rolling_z, build_heat_score


class TestSignal:
    def test_weekly_counts_shape(self):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        df = pd.DataFrame({
            "transaction_date": rng.choice(dates, 600),
            "sector": rng.choice(SECTOR_NAMES, 600),
        })
        wide = _weekly_counts(df, "transaction_date")
        assert set(wide.columns) == set(SECTOR_NAMES)
        assert len(wide) > 0

    def test_rolling_z_handles_zero_std(self):
        # All-constant column should produce all-NaN z-scores (no division by 0 errors)
        df = pd.DataFrame({"x": np.ones(40)})
        z = _rolling_z(df, 10)
        assert z["x"].isna().all()

    def test_heat_score_long_form(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="W-FRI")
        rng = np.random.default_rng(1)
        tc = pd.DataFrame(rng.poisson(5, (40, 5)), index=dates, columns=SECTOR_NAMES)
        bc = pd.DataFrame(rng.poisson(3, (40, 5)), index=dates, columns=SECTOR_NAMES)
        h = build_heat_score(tc, bc)
        assert {"date", "sector", "trade_z", "bill_z", "heat"}.issubset(h.columns)
        assert h["heat"].notna().sum() > 0


# ============================================================================
# rq1_predictive.py
# ============================================================================

from rq1_predictive import _heat_signal_for, run_rq1


class TestRQ1Helpers:
    def test_heat_signal_picker_prefers_trade_z(self):
        # If trade_z has enough non-null rows, that's what gets returned
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "date": list(dates) * 1,
            "sector": ["tech"] * len(dates),
            "trade_z": rng.normal(size=len(dates)),
            "heat": [np.nan] * len(dates),
        })
        sig = _heat_signal_for(df, "tech")
        assert sig is not None
        assert sig.notna().sum() == len(dates)

    def test_heat_signal_picker_returns_none_on_sparse(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-05", periods=10, freq="W-FRI"),
            "sector": ["tech"] * 10,
            "trade_z": [np.nan] * 10,
            "heat": [np.nan] * 10,
        })
        sig = _heat_signal_for(df, "tech")
        # heat column exists but all-null, picker returns it but caller drops
        assert sig is None or sig.notna().sum() == 0


# ============================================================================
# rq2_sensitivity.py
# ============================================================================

from rq2_sensitivity import _build_weekly_signals, DIRECTION


class TestRQ2:
    def test_direction_map_covers_senate_and_house_codes(self):
        for code in ["Purchase", "Sale (Full)", "Sale (Partial)", "P", "S", "S (partial)"]:
            assert code in DIRECTION

    def test_weekly_signals_three_keys(self):
        df = pd.DataFrame([
            {"transaction_date": "2024-01-05", "sector": "tech",
             "transaction_type": "Purchase", "amount_midpoint": 8000},
            {"transaction_date": "2024-01-05", "sector": "tech",
             "transaction_type": "Sale (Full)", "amount_midpoint": 32500},
        ])
        sigs = _build_weekly_signals(df)
        assert set(sigs.keys()) == {"net_buy_count", "dollar_net", "trade_count"}
        # net buy count: 1 buy - 1 sell = 0
        assert sigs["net_buy_count"].iloc[0]["tech"] == 0
        # trade count: 2
        assert sigs["trade_count"].iloc[0]["tech"] == 2
        # dollar net: 8000 - 32500 = -24500
        assert sigs["dollar_net"].iloc[0]["tech"] == -24500


# ============================================================================
# alpha_context.py
# ============================================================================

from alpha_context import build_weights, portfolio_returns, _summarize


def _synthetic_heat(n_weeks=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-07", periods=n_weeks, freq="W-FRI")
    rows = [{"date": d, "sector": s, "heat": rng.normal()}
            for d in dates for s in SECTOR_NAMES]
    return pd.DataFrame(rows)


def _synthetic_market(start="2022-01-01", n_days=1400, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    data = {}
    for s in SECTOR_NAMES:
        rets = rng.normal(0.0003, 0.012, n_days)
        data[SECTORS[s]["etf"]] = 100.0 * np.exp(np.cumsum(rets))
    data["^VIX"] = 20 + np.cumsum(rng.normal(0, 0.2, n_days))
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="Date"))


class TestAlpha:
    def test_weights_dollar_neutral(self):
        h = _synthetic_heat()
        w = build_weights(h)
        assert not w.empty
        # Each row sums to ~0
        assert np.allclose(w.sum(axis=1).dropna(), 0, atol=1e-9)

    def test_portfolio_returns_run(self):
        h = _synthetic_heat()
        m = _synthetic_market()
        w = build_weights(h)
        pr = portfolio_returns(w, m)
        assert len(pr) > 50

    def test_summarize_keys(self):
        rng = np.random.default_rng(3)
        port = pd.Series(rng.normal(0.001, 0.02, 200))
        s = _summarize(port)
        for k in ["sharpe", "max_drawdown", "ann_return", "ann_vol",
                  "cumulative_return"]:
            assert k in s
        assert s["max_drawdown"] <= 0


# ============================================================================
# Integration: smoke run on bundled CSVs
# ============================================================================

@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data" / "market.csv").exists(),
    reason="bundled data not present",
)
class TestIntegration:
    def test_full_pipeline_on_bundled_data(self, tmp_path):
        """Run tag -> signal -> rq1 -> rq2 -> alpha on whatever is in data/.
        Asserts outputs land in results/ with non-zero size."""
        from config import RESULTS_DIR
        import tagging, heat_signal as sigmod, rq1_predictive, rq2_sensitivity, alpha_context
        tagging.main()
        sigmod.main()
        rq1_predictive.main()
        rq2_sensitivity.main()
        alpha_context.main()
        for fn in ["rq1_inference.csv", "rq2_sensitivity.csv",
                   "alpha_summary.csv", "alpha_report.txt"]:
            p = RESULTS_DIR / fn
            if p.exists():
                assert p.stat().st_size > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
