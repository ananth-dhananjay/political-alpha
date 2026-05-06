Political Alpha — Results Notes
================================

This document accompanies the slide deck and the code. It explains exactly
what the bundled pipeline produces when you run `python main.py`, how that
relates to the numbers quoted in the slides, and what the defensible takeaway
is from the project as a whole.

What the bundled pipeline produces
----------------------------------

Running `python main.py` on the bundled `data/` CSVs writes the following
to `results/`:

  - rq1_inference.csv      20 specifications (5 sectors × 4 outcomes)
  - rq2_sensitivity.csv    full-sample + first-half / second-half splits
                           across three signal variants
  - alpha_summary.csv      CAPM and FF3 long-short portfolio regression
  - portfolio_returns.csv
  - portfolio_weights.csv
  - alpha_report.txt

Headline numbers from the canonical pipeline run:

  RQ1: 2 of 20 specifications significant at p<0.05.
       Both involve healthcare:
         healthcare fwd_ret_12w   coef +0.0146  t = +3.42  p = 0.0006
         healthcare fwd_vol_4w    coef -0.0222  t = -4.95  p < 0.0001

  RQ2: Healthcare net-buy count predicts forward 20-day return at p ~ 0.03
       on the full sample. Healthcare 12-week return effect weakens in the
       second-half OOS split (p = 0.16). Healthcare 4-week vol effect
       partially replicates (second-half p = 0.015).

  Alpha: long-short (long top-2, short bottom-2) portfolio shows no
         statistically significant CAPM or FF3 alpha.

Why the deck and the pipeline give different headline counts
------------------------------------------------------------

The deck (presented April 22, 2026) reports "0 of 20 specifications
significant at p<0.05." That was an earlier exploratory run with three
specification choices that differed from the canonical pipeline now in src/:

  1. The deck regressions used daily forward returns (1, 5, 20 trading days)
     with the weekly heat score forward-filled across the daily calendar.
     The canonical pipeline now in rq1_predictive.py uses weekly forward
     returns (1, 4, 12 weeks) at native weekly frequency.

  2. The composite heat column ("heat") in heat_score.csv is sparse because
     of poor Congress.gov bill coverage in the trade window (5.5% of bills
     tagged, almost none in 2023-2026; documented in challenges slide §06).
     The canonical pipeline now uses the trade_z component directly, which
     has ~98 weeks of coverage per sector instead of ~20.

  3. The deck used HAC lag = max(1.5h, 5) on a daily series. The canonical
     pipeline uses HAC lag = max(h, 4) on a weekly series.

Both setups are defensible. The differences are exactly the kind of
specification choices that the §09 reflection slide flags: in a small,
concentrated sample, results are sensitive to specification choices, and
the methodological discipline matters more than any single regression.

What both runs agree on
-----------------------

Healthcare is the one sector where the heat score keeps appearing as
anomalous, regardless of specification:

  - Deck RQ2 sensitivity:    healthcare 20d net-buy hit (p = 0.03), did not
                             replicate in second-half OOS (p = 0.46).
  - Canonical RQ1:           healthcare 12-week forward return (p < 0.001),
                             healthcare 4-week forward vol (p < 0.0001).
  - Canonical RQ2:           healthcare net-buy and dollar-net signals both
                             produce sub-0.05 hits on full sample.

Direction of the healthcare effect is consistent across runs: senators
net-selling healthcare precedes higher 12-week returns and lower 4-week
volatility, or equivalently net-buying precedes lower returns and higher
volatility. With only ~80 effective weekly observations and one sector
showing up across many overlapping specifications, the multiple-comparison
discipline noted in the deck applies: across 30+ tests at α=0.05, observing
3-5 hits in one sector is suggestive but not conclusive.

What the project does and does not show
---------------------------------------

What the project shows:

  - The post-STOCK Act disclosure regime appears to be doing its job at the
    aggregate level. 28 of 30 specifications across primary and sensitivity
    analyses fail to reject the null of no predictive signal.
  - Senate trading 2023-2026 is concentrated in a handful of senators
    (3 senators = 47.6% of trades) and shows directional asymmetry by
    sector (healthcare and finance net-sold; energy net-bought).
  - One specific sector, healthcare, repeatedly shows up as an anomaly
    across specifications. The robust direction is that senators net-selling
    healthcare precedes calmer / better forward outcomes for the sector ETF.

What the project does not show:

  - That senators have a generalized insider-trading edge. The aggregate
    null is not rejected.
  - That the healthcare anomaly reflects forward-looking information rather
    than coincidence with underlying healthcare-sector dynamics in the
    sample window. Out-of-sample replication is partial, not conclusive.
  - Anything causal. All inference here is observational.

Reproducibility notes
---------------------

The bundled data/ folder contains the actual CSVs used in the analysis.
Run `python main.py` from src/ to regenerate the results files. Run
`python main.py --fetch` to re-pull from the four data sources (slow:
the Senate scrape can take 20-60 minutes).

Tests pass with `pytest tests.py -v` (20/20). The integration test
re-runs the entire analysis pipeline on the bundled data and confirms
that all results files are written.
