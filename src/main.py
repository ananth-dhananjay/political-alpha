import os
from config import DATA_DIR, RESULTS_DIR, SECTOR_TICKERS, ETF_START_DATE, HOUSE_DISCLOSURE_YEARS, CONGRESS_SESSIONS
from load import fetch_sector_etf_data, fetch_house_disclosures, fetch_bills

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Yahoo Finance API ---
    print("--- Fetching Sector ETF Data (Yahoo Finance) ---")
    etf_data = fetch_sector_etf_data(SECTOR_TICKERS, start=ETF_START_DATE, output_dir=DATA_DIR)
    print(f"Total ETF rows: {len(etf_data)}")
    print("\n" + "=" * 50 + "\n")

    # --- House Clerk (Web Scraping) ---
    print("--- Fetching House Disclosures ---")
    for year in HOUSE_DISCLOSURE_YEARS:
        filings = fetch_house_disclosures(year, output_dir=DATA_DIR)
        print(f"  {year}: {len(filings)} filings")
    print("\n" + "=" * 50 + "\n")

    # --- Congress.gov API ---
    print("--- Fetching Bills (Congress.gov) ---")
    for congress in CONGRESS_SESSIONS:
        bills = fetch_bills(congress, output_dir=DATA_DIR)
        print(f"  Congress {congress}: {len(bills)} bills")
    print("\n" + "=" * 50 + "\n")

    print("--- Data collection complete. Check the 'data' directory. ---")
