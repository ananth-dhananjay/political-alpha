from pathlib import Path
from dotenv import load_dotenv

# project configuration from .env (secret part)
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)  # loads into os.environ

# project configuration
DATA_DIR = "../data"
RESULTS_DIR = "../results"

# data sources configuration
SECTOR_TICKERS = {
    "defense": "ITA",
    "finance": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "tech": "XLK",
    "volatility": "^VIX",
}

ETF_START_DATE = "2020-01-01"

HOUSE_DISCLOSURE_YEARS = [2020, 2021, 2022, 2023, 2024]

CONGRESS_SESSIONS = [116, 117, 118]  # 2019-2025
