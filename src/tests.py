"""
Political Alpha – Tests
Run with: pytest tests.py -v (from src/ directory)
"""

import os
import pytest
import pandas as pd
from load import fetch_sector_etf_data, fetch_house_disclosures
from config import DATA_DIR


class TestYahooFinanceAPI:
    """Tests for the Yahoo Finance API data source."""

    def test_fetch_single_ticker(self, tmp_path):
        df = fetch_sector_etf_data(
            tickers={"tech": "XLK"},
            start="2024-01-01",
            end="2024-01-31",
            output_dir=str(tmp_path),
        )
        assert not df.empty, "DataFrame should not be empty for XLK"
        assert len(df) > 10, "Should have multiple trading days in Jan 2024"

    def test_expected_columns(self, tmp_path):
        df = fetch_sector_etf_data(
            tickers={"finance": "XLF"},
            start="2024-06-01",
            end="2024-06-15",
            output_dir=str(tmp_path),
        )
        for col in ["Open", "High", "Low", "Close", "Volume", "ticker", "sector"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_output_created(self, tmp_path):
        fetch_sector_etf_data(
            tickers={"defense": "ITA"},
            start="2024-01-01",
            end="2024-01-10",
            output_dir=str(tmp_path),
        )
        csv_files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]
        assert len(csv_files) > 0, "Should create at least one CSV file"

    def test_invalid_ticker_handled(self, tmp_path):
        df = fetch_sector_etf_data(
            tickers={"fake": "ZZZZZZZZZ"},
            start="2024-01-01",
            end="2024-01-10",
            output_dir=str(tmp_path),
        )
        assert df.empty, "Invalid ticker should return empty DataFrame"

    def test_date_range_respected(self, tmp_path):
        df = fetch_sector_etf_data(
            tickers={"energy": "XLE"},
            start="2024-03-01",
            end="2024-03-31",
            output_dir=str(tmp_path),
        )
        dates = pd.to_datetime(df.index)
        assert dates.min() >= pd.Timestamp("2024-03-01")
        assert dates.max() <= pd.Timestamp("2024-03-31")


class TestHouseDisclosures:
    """Tests for House financial disclosure scraping."""

    def test_returns_list(self):
        result = fetch_house_disclosures(year=2023, output_dir="/tmp/test_house")
        assert isinstance(result, list)

    def test_has_filings(self):
        result = fetch_house_disclosures(year=2024, output_dir="/tmp/test_house")
        assert len(result) > 0, "2024 should have filings"

    def test_nonexistent_year(self):
        result = fetch_house_disclosures(year=1900, output_dir="/tmp/test_house")
        assert result == []
