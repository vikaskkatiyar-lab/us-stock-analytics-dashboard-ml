import os
from pathlib import Path

FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

TOP_N = 250
LOOKBACK_DAYS = 420
ALERT_DAILY_MOVE_PCT = 5.0
ALERT_DRAWDOWN_PCT = -20.0
