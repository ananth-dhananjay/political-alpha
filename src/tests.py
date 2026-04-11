"""
tests.py
--------
Unit + integration tests for fetch_market_data.py.

Run with:
    pytest src/tests.py -v

Integration tests (marked with @pytest.mark.integration) hit the real
Yahoo Finance API and require an internet connection.  Run them with:
    pytest src/tests.py -v -m integration
"""

import os
import sys
import tempfile
import pandas as pd
import pytest

# Make sure src/ is importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from fetch_market_data import (
    TICKERS,
    fetch_ticker,
    fetch_all,
    load_combined,
    DEFAULT_START,
    DEFAULT_END,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_csv(tmp_dir):
    """
    Write a minimal market CSV into tmp_dir and return its path.
    Simulates what fetch_ticker() produces so load_combined() can be tested
    without hitting the network.
    """
    dates = pd.date_range("2023-01-03", periods=5, freq="B")
    for ticker_raw in TICKERS:
        clean = ticker_raw.replace("^", "")
        df = pd.DataFrame(
            {
                "open":   [100.0] * 5,
                "high":   [105.0] * 5,
                "low":    [98.0]  * 5,
                "close":  [102.0, 103.0, 101.0, 104.0, 100.0],
                "volume": [1_000_000] * 5,
            },
            index=dates,
        )
        df.index.name = "date"
        df.to_csv(os.path.join(tmp_dir, f"{clean}.csv"))
    return tmp_dir


# ---------------------------------------------------------------------------
# Unit tests — TICKERS config
# ---------------------------------------------------------------------------

class TestTickersConfig:
    def test_all_expected_tickers_present(self):
        for t in ("ITA", "XLF", "XLE", "XLV", "XLK", "^VIX"):
            assert t in TICKERS, f"{t} missing from TICKERS"

    def test_sector_labels_are_strings(self):
        for ticker, sector in TICKERS.items():
            assert isinstance(sector, str) and len(sector) > 0, (
                f"Empty sector label for {ticker}"
            )

    def test_ticker_count(self):
        assert len(TICKERS) == 6


# ---------------------------------------------------------------------------
# Unit tests — load_combined (no network needed)
# ---------------------------------------------------------------------------

class TestLoadCombined:
    def test_returns_dataframe(self, sample_csv):
        df = load_combined(output_dir=sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_columns(self, sample_csv):
        df = load_combined(output_dir=sample_csv)
        expected = {t.replace("^", "") for t in TICKERS}
        assert set(df.columns) == expected

    def test_index_is_datetime(self, sample_csv):
        df = load_combined(output_dir=sample_csv)
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_sorted_ascending(self, sample_csv):
        df = load_combined(output_dir=sample_csv)
        assert df.index.is_monotonic_increasing

    def test_raises_when_no_files(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_combined(output_dir=tmp_dir)

    def test_row_count(self, sample_csv):
        df = load_combined(output_dir=sample_csv)
        assert len(df) == 5

    def test_no_all_nan_rows(self, sample_csv):
        df = load_combined(output_dir=sample_csv)
        assert not df.isna().all(axis=1).any()


# ---------------------------------------------------------------------------
# Integration tests — real Yahoo Finance API
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFetchTicker:
    def test_returns_dataframe(self, tmp_dir):
        df = fetch_ticker("XLK", start="2023-01-01", end="2023-03-01", output_dir=tmp_dir)
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    def test_csv_written_to_disk(self, tmp_dir):
        fetch_ticker("XLK", start="2023-01-01", end="2023-03-01", output_dir=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "XLK.csv"))

    def test_required_columns_present(self, tmp_dir):
        df = fetch_ticker("XLK", start="2023-01-01", end="2023-03-01", output_dir=tmp_dir)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_close(self, tmp_dir):
        df = fetch_ticker("XLK", start="2023-01-01", end="2023-03-01", output_dir=tmp_dir)
        assert df["close"].isna().sum() == 0

    def test_index_name_is_date(self, tmp_dir):
        df = fetch_ticker("XLK", start="2023-01-01", end="2023-03-01", output_dir=tmp_dir)
        assert df.index.name == "date"

    def test_vix_caret_stripped_from_filename(self, tmp_dir):
        fetch_ticker("^VIX", start="2023-01-01", end="2023-03-01", output_dir=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "VIX.csv"))

    def test_bad_ticker_returns_none(self, tmp_dir):
        result = fetch_ticker("XXXXNOTREAL", start="2023-01-01", end="2023-03-01", output_dir=tmp_dir)
        assert result is None


@pytest.mark.integration
class TestFetchAll:
    def test_returns_dict(self, tmp_dir):
        result = fetch_all(start="2023-01-01", end="2023-02-01", output_dir=tmp_dir)
        assert isinstance(result, dict)

    def test_all_tickers_fetched(self, tmp_dir):
        result = fetch_all(start="2023-01-01", end="2023-02-01", output_dir=tmp_dir)
        assert len(result) == len(TICKERS)

    def test_all_csvs_written(self, tmp_dir):
        fetch_all(start="2023-01-01", end="2023-02-01", output_dir=tmp_dir)
        for ticker in TICKERS:
            clean = ticker.replace("^", "")
            assert os.path.exists(os.path.join(tmp_dir, f"{clean}.csv")), (
                f"Missing CSV for {ticker}"
            )
