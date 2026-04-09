# Political Alpha: Congressional Trading & Sector Volatility

This project investigates whether U.S. congressional trading behavior and legislative activity serve as leading indicators for sector-level equity volatility. It constructs a composite "Political Heat Score" across five sectors (defense, finance, energy, healthcare, tech) and tests its predictive power against forward ETF returns and VIX movements.

# Data Sources

| # | Name | Source URL | Type | Fields | Format | Est. Data Size |
|---|------|-----------|------|--------|--------|---------------|
| 1 | Sector ETF Prices & VIX | finance.yahoo.com | API (yfinance) | Date, OHLCV for ITA, XLF, XLE, XLV, XLK, ^VIX | CSV | ~1,573 days x 6 tickers |
| 2 | House Financial Disclosures | disclosures-clerk.house.gov | Web scraping | Filer, Office, Date, Document ID | JSON | ~2,000+ filings/year |
| 3 | Legislative Activity | api.congress.gov | API | Bill title, sponsor, committee, policy area, status | JSON | ~250+ bills/session |

# Results
_To be completed in final submission._

# Installation
1. Copy `src/.env.example` to `src/.env` and add your Congress.gov API key (free at https://api.congress.gov/sign-up/)
2. Install dependencies: `pip install -r requirements.txt`

# Running analysis

From `src/` directory run:

`python main.py`

Results will appear in `results/` folder. All obtained data will be stored in `data/`.
