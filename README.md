# Political Alpha

**DSCI 510 — Spring 2026 — University of Southern California**
**Author:** Ananth Dhananjay
**Instructor:** Dr. Alexey Tregubov

---

## Introduction

Political Alpha investigates whether U.S. Senate stock disclosures carry predictive
signal for sector-level ETF returns. The STOCK Act (2012) requires senators to publicly
disclose trades within 45 days. This project asks: once that disclosure is public, does
it still predict forward returns? If yes, the regime is failing. If no, it is working.

The pipeline scrapes Senate Periodic Transaction Reports, tags each trade to one of five
sectors, builds a weekly political heat score, and tests its predictive power across
three research questions:

- **RQ1:** Does the heat score statistically predict forward ETF returns or realized volatility?
- **RQ2:** Does the result hold across alternative signal specifications?
- **RQ3:** Do committee-aligned trades carry more predictive power than unaligned ones?

The pipeline also builds a weekly long-short portfolio and estimates CAPM and Fama-French
3-factor alpha. A structural break analysis tests whether the signal shifted over time.

---

## Data Sources

| # | Source | Access Method | Coverage | Role |
|---|--------|---------------|----------|------|
| 1 | Senate EFD (`efdsearch.senate.gov`) | HTTP + CSRF session, HTML table parse | 7,000+ trades · 34+ senators | Trade signal |
| 2 | Yahoo Finance | `yfinance` library | 1,507 trading days × 7 tickers | ETF prices, SPY, VIX, IRX |
| 3 | Congress.gov API | REST API (free key) | 5,000 bills · 276 sector-tagged | Legislative signal |
| 4 | Ken French Data Library | Public ZIP download | 1,500+ daily rows: Mkt-RF/SMB/HML/RF | Factor controls |

**Five sectors:** Defense (ITA) · Finance (XLF) · Energy (XLE) · Healthcare (XLV) · Tech (XLK)

---

## Analysis

1. **Signal construction:** Tagged trades aggregated to weekly buy-sell counts per sector,
   z-score normalized over 26-week rolling window, combined 50/50 with bill-introduction
   signal, lagged 1 week to prevent look-ahead bias.

2. **RQ1 — OLS/HAC inference:** One regression per sector per dependent variable
   (1-week, 4-week, 12-week forward return; realized volatility). Newey-West HAC
   standard errors correct for autocorrelation.

3. **RQ2 — Sensitivity specs:** Same regression across alternative signal constructions
   (trade-only, bill-only, dollar-weighted, net-buy count).

4. **RQ3 — Committee alignment:** Trades split by committee jurisdiction overlap.
   OLS with both signals; F-test on coefficient equality.

5. **Alpha regression:** Weekly long-top-2 / short-bottom-2 sector portfolio regressed
   on CAPM and Fama-French 3 factors.

6. **Structural break:** Andrews (1993) SupF test and CUSUM test per sector. Significant
   if SupF > 9.10 (5% critical value, k=2). Rolling 60-week coefficient for visual inspection.

---

## Summary of Results

- **0 of 20** RQ1 specifications significant at p < 0.05
- One secondary spec (healthcare 20-day, raw net-buy) hit p = 0.03 but did not replicate OOS (p = 0.46)
- CAPM alpha: −0.000594 weekly (−3.0% annualized), t = −0.44, p = 0.66
- FF3 alpha: −0.000210 weekly (−1.1% annualized), t = −0.15, p = 0.88
- Portfolio cumulative return: +4.3% | Max drawdown: −15.5%

The null is consistent with the post-STOCK-Act literature. The 45-day disclosure lag
is long enough that the information content is largely arbitraged away by the time
disclosures are public.

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Python 3.10+ required. The Senate EFD scraper requires a residential IP — datacenter
IPs (Colab, AWS, GCP) are blocked with 503 errors.

### Environment setup

```bash
cp src/.env.example src/.env
# Add your Congress.gov API key (free at https://api.congress.gov/sign-up/)
```

### Full pipeline

```bash
cd src/
```

# Step 1: fetch all data (Senate scraper takes ~15-20 min, requires residential IP)
python main.py --fetch

OR: download pre-fetched data from Google Drive and skip the fetch step
https://drive.google.com/drive/folders/17XTOBd8quzmklovxYI3fAM0-X3Q9DjpN?usp=drive_link
Place the downloaded files into the data/ folder, then run:

```
# Step 2: run analysis on cached data
python main.py
```

Runs in order: tagging → heat score → RQ1 → RQ2 → RQ3 → alpha → structural break.

### Open results notebook

```bash
cd ..
jupyter notebook results.ipynb
# Run all cells — loads from data/ and results/, no re-fetching needed
```

### Run a single stage

```bash
python main.py --only rq1      # primary regressions only
python main.py --only alpha    # portfolio analysis only
python main.py --only break    # structural break only
```

### Run tests

```bash
cd src/
python -m pytest tests.py -v
```

### API keys required

| Key | Source | File |
|-----|--------|------|
| `CONGRESS_API_KEY` | https://api.congress.gov/sign-up/ (free) | `src/.env` |

---

## Repository Structure

```
politicalalpha/
├── src/
│   ├── config.py           # All constants — import from here, never hardcode
│   ├── load_market.py      # Yahoo Finance ETF/market data
│   ├── load_senate.py      # Senate EFD scraper (CSRF session + HTML parse)
│   ├── load_bills.py       # Congress.gov bill data
│   ├── load_factors.py     # Fama-French 3 factors
│   ├── tagging.py          # Sector assignment for trade records
│   ├── heat_signal.py      # Weekly heat score construction
│   ├── rq1_predictive.py   # OLS/HAC inference (RQ1)
│   ├── rq2_sensitivity.py  # Sensitivity specifications (RQ2)
│   ├── rq3_interaction.py  # Committee alignment test (RQ3)
│   ├── alpha_context.py    # Portfolio construction and factor regressions
│   ├── structural_break.py # Andrews SupF + CUSUM structural break tests
│   ├── main.py             # Pipeline orchestrator
│   ├── tests.py            # Unit tests
│   └── .env.example        # Environment variable template (no values)
├── docs/
│   ├── Ananth_Dhananjay_presentation.pdf
│   └── Ananth_Dhananjay_progress_report.pdf
├── data/                   # Raw data (gitignored — populated by --fetch)
├── results/                # Analysis outputs (gitignored — populated by main.py)
├── results.ipynb           # Results notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## AI Tools Used

Code generation assistance provided by **Claude (Anthropic, claude-sonnet-4-6)**.
All AI-generated sections labeled with `# AI generated:` comments. Research design,
signal construction logic, sector tagging rules, regression specifications, and test
cases written manually.
