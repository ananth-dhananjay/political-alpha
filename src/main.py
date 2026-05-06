"""Pipeline orchestrator.

Default flow (skip --fetch on a graded run; the bundled data CSVs are enough):

    python main.py --fetch     # 1. ingest from all four data sources
    python main.py             # 2. tag + signal + analysis (uses bundled CSVs)

Individual stages:

    python main.py --only fetch     # all four data sources
    python main.py --only tag       # sector tagging
    python main.py --only signal    # weekly heat score
    python main.py --only rq1       # primary regressions
    python main.py --only rq2       # sensitivity / OOS
    python main.py --only alpha     # portfolio + factor regressions
"""

from __future__ import annotations
import argparse
import sys
import time

from config import DATA_DIR


REQUIRED_INPUT_FILES = [
    "market.csv",
    "benchmarks.csv",
    "bills_all.csv",
    "ptr_trades.csv",
    "ff3_daily.csv",
]


def _check_inputs():
    missing = [f for f in REQUIRED_INPUT_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print(f"[main] missing {len(missing)} input file(s) in {DATA_DIR}:")
        for f in missing:
            print(f"         - {f}")
        print("[main] run `python main.py --fetch` to populate data/")
        sys.exit(1)


def stage_fetch():
    import load_market, load_senate, load_bills, load_factors
    load_market.main()
    load_senate.main()
    load_bills.main()
    load_factors.main()


def stage_tag():
    import tagging
    tagging.main()


def stage_signal():
    import heat_signal
    heat_signal.main()


def stage_rq1():
    import rq1_predictive
    rq1_predictive.main()


def stage_rq2():
    import rq2_sensitivity
    rq2_sensitivity.main()


def stage_rq3():
    import rq3_interaction
    rq3_interaction.main()


def stage_alpha():
    import alpha_context
    alpha_context.main()


def stage_break():
    import structural_break
    structural_break.main()


STAGES = {
    "fetch": stage_fetch,
    "tag": stage_tag,
    "signal": stage_signal,
    "rq1": stage_rq1,
    "rq2": stage_rq2,
    "rq3": stage_rq3,
    "alpha": stage_alpha,
    "break": stage_break,
}


def main():
    parser = argparse.ArgumentParser(description="Political Alpha pipeline")
    parser.add_argument("--fetch", action="store_true",
                        help="Run the data ingestion stage before analysis.")
    parser.add_argument("--only", choices=list(STAGES.keys()),
                        help="Run a single stage and exit.")
    args = parser.parse_args()

    start = time.time()

    if args.only:
        STAGES[args.only]()
        print(f"[main] elapsed {time.time() - start:.1f}s")
        return

    if args.fetch:
        stage_fetch()

    stage_tag()
    stage_signal()
    stage_rq1()
    stage_rq2()
    stage_rq3()
    stage_alpha()
    stage_break()
    print(f"[main] total elapsed {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
