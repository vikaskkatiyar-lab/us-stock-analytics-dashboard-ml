import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from datetime import date, timedelta
from config import LOOKBACK_DAYS

class FMPClient:
    def __init__(self, api_key=None):
        pass

    def get_top_us_stocks_by_market_cap(self, limit=250):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        sp500 = tables[0]

        sp500 = sp500.rename(columns={
            "Symbol": "symbol",
            "Security": "companyName",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry"
        })

        records = []

        for _, row in sp500.iterrows():
            symbol = str(row["symbol"]).replace(".", "-")

            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                market_cap = getattr(info, "market_cap", None)

                if market_cap and market_cap > 0:
                    records.append({
                        "symbol": symbol,
                        "companyName": row.get("companyName"),
                        "marketCap": market_cap,
                        "exchange": "US",
                        "sector": row.get("sector"),
                        "industry": row.get("industry")
                    })

            except Exception:
                continue

        df = pd.DataFrame(records)

        if df.empty:
            raise RuntimeError("No stock universe data returned from Yahoo Finance.")

        df = df.sort_values("marketCap", ascending=False).drop_duplicates("symbol")

        return df.head(limit)[["symbol", "companyName", "marketCap", "exchange", "sector", "industry"]]

    def get_historical_prices(self, symbol):
        end = date.today()
        start = end - timedelta(days=LOOKBACK_DAYS)

        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=True,
            group_by="column"
        )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        if "Close" not in df.columns:
            return pd.DataFrame()

        cols = ["date", "Close"]
        if "High" in df.columns:
            cols.append("High")
        if "Low" in df.columns:
            cols.append("Low")

        df = df[cols].rename(columns={
            "Close": "adj_close",
            "High": "adj_high",
            "Low": "adj_low"
        })
        df["date"] = pd.to_datetime(df["date"])
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        if "adj_high" in df.columns:
            df["adj_high"] = pd.to_numeric(df["adj_high"], errors="coerce")
        if "adj_low" in df.columns:
            df["adj_low"] = pd.to_numeric(df["adj_low"], errors="coerce")
        df = df.dropna(subset=["adj_close"])
        df["symbol"] = symbol

        return df
