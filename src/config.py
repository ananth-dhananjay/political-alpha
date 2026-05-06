"""Central configuration. All paths, parameters, and tunable choices live here.

Pipeline modules import from this single file so that re-running with different
windows, lag choices, or sector definitions only requires editing one place.
"""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DOC_DIR = PROJECT_ROOT / "doc"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Study window
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# Five sectors. Each has an ETF, a list of tickers commonly traded by senators,
# and a keyword set used as a fallback when a trade has no clean ticker.
SECTORS = {
    "defense": {
        "etf": "ITA",
        "tickers": ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII", "TDG"],
        "keywords": ["defense", "military", "armed forces", "pentagon", "weapons"],
    },
    "finance": {
        "etf": "XLF",
        "tickers": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW"],
        "keywords": ["banking", "financial", "securities", "credit", "federal reserve"],
    },
    "energy": {
        "etf": "XLE",
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
        "keywords": ["energy", "oil", "gas", "renewable", "climate", "petroleum"],
    },
    "healthcare": {
        "etf": "XLV",
        "tickers": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT"],
        "keywords": ["health", "medicare", "medicaid", "drug", "pharmaceutical"],
    },
    "tech": {
        "etf": "XLK",
        "tickers": ["MSFT", "AAPL", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "INTC"],
        "keywords": ["technology", "cybersecurity", "artificial intelligence", "semiconductor"],
    },
}
SECTOR_NAMES = list(SECTORS.keys())
SECTOR_ETFS = [SECTORS[s]["etf"] for s in SECTOR_NAMES]

# Senate disclosure amount buckets, mapped to midpoints (USD).
PTR_AMOUNT_BUCKETS = {
    "$1,001 - $15,000": 8000,
    "$15,001 - $50,000": 32500,
    "$50,001 - $100,000": 75000,
    "$100,001 - $250,000": 175000,
    "$250,001 - $500,000": 375000,
    "$500,001 - $1,000,000": 750000,
    "$1,000,001 - $5,000,000": 3000000,
    "$5,000,001 - $25,000,000": 15000000,
    "$25,000,001 - $50,000,000": 37500000,
    "Over $50,000,000": 75000000,
}

# Heat score parameters
SIGNAL_FREQUENCY = "W-FRI"   # weekly, Friday-anchored
ZSCORE_WINDOW_WEEKS = 26     # 6-month rolling window
SIGNAL_LAG_WEEKS = 1         # one-week lag prevents look-ahead

# RQ1 / RQ2 parameters
FORWARD_WINDOWS_DAYS = [1, 5, 20]
VOL_WINDOW_DAYS = 20
RANDOM_SEED = 42

# Alpha context parameters
PORTFOLIO_N_LONG = 2
PORTFOLIO_N_SHORT = 2
PERIODS_PER_YEAR = 52  # weekly rebalance
HAC_BASE_LAGS = 4

# API credentials
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY", "")

# ── additions for rq3 and structural_break ─────────────────────────────────
SECTOR_TICKERS: dict = {s: v["etf"] for s, v in SECTORS.items()}

SECTOR_COMMITTEES: dict = {
    "defense":    ["Armed Services", "Homeland Security"],
    "finance":    ["Banking", "Finance", "Budget"],
    "energy":     ["Energy", "Environment"],
    "healthcare": ["Health", "HELP", "Finance"],
    "tech":       ["Commerce", "Science", "Intelligence"],
}

FORWARD_RETURN_HORIZONS_DAYS: list = FORWARD_WINDOWS_DAYS
HAC_LAGS: int = HAC_BASE_LAGS
SIGNIFICANCE_LEVEL: float = 0.05

RQ1_RESULTS_FILE   = RESULTS_DIR / "rq1_inference.csv"
RQ2_RESULTS_FILE   = RESULTS_DIR / "rq2_sensitivity.csv"
RQ3_RESULTS_FILE   = RESULTS_DIR / "rq3_interaction.csv"
ALPHA_RESULTS_FILE = RESULTS_DIR / "alpha_summary.csv"
PORTFOLIO_FILE     = RESULTS_DIR / "portfolio_returns.csv"
HEAT_SCORE_FILE    = DATA_DIR / "heat_score.csv"

FF3_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
           "ftp/F-F_Research_Data_Factors_daily_CSV.zip")

SENATE_EFD_BASE_URL = "https://efdsearch.senate.gov"
