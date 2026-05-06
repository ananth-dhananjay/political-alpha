"""Fama-French 3-factor daily ingestion (Ken French Data Library).

Pulls the public Mkt-RF / SMB / HML / RF zip from Dartmouth and converts
percentages to decimals. Output is the factor control set used in the
alpha_context module's FF3 regression.

Output:
  - data/ff3_daily.csv
"""

from __future__ import annotations
import io
import zipfile
import pandas as pd
import requests

from config import DATA_DIR, START_DATE, END_DATE, FF3_URL




def fetch_ff3() -> pd.DataFrame:
    print(f"[load_factors] fetching FF3 daily")
    r = requests.get(FF3_URL, timeout=60,
                     headers={"User-Agent": "DSCI510-PoliticalAlpha/1.0"})
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
    if csv_name is None:
        raise ValueError("No CSV found in FF3 zip: " + str(zf.namelist()))
    raw = zf.read(csv_name).decode("latin-1")

    # The raw file has a preamble plus a trailing annual block. Slice the
    # daily block: from the header row containing "Mkt-RF" to the next blank line.
    lines = raw.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if "Mkt-RF" in ln)
    end_idx = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if not lines[i].strip():
            end_idx = i
            break
    block = "\n".join(lines[header_idx:end_idx])
    df = pd.read_csv(io.StringIO(block))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df = df.loc[START_DATE:END_DATE]
    out = DATA_DIR / "ff3_daily.csv"
    df.to_csv(out)
    print(f"[load_factors] wrote {out} shape={df.shape}")
    return df


def main():
    fetch_ff3()


if __name__ == "__main__":
    main()
