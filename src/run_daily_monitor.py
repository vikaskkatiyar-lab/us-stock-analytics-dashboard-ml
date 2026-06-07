import time
import pandas as pd
from fmp_client import FMPClient
from metrics import calculate_metrics, format_metric_columns
from reporting import generate_outputs
from config import DATA_DIR, TOP_N

PRICE_HISTORY_FILE = DATA_DIR / "price_history.csv"

def save_price_history(price_frames):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if price_frames:
        history = pd.concat(price_frames, ignore_index=True)
        history = history.sort_values(["symbol", "date"])
    else:
        history = pd.DataFrame(columns=["date", "adj_close", "adj_high", "adj_low", "symbol"])

    history.to_csv(PRICE_HISTORY_FILE, index=False)
    return PRICE_HISTORY_FILE

def main():
    client = FMPClient()

    print(f"Building top {TOP_N} US stock universe by market cap...")
    universe = client.get_top_us_stocks_by_market_cap(limit=TOP_N)

    results = []
    price_frames = []
    for _, row in universe.iterrows():
        symbol = row["symbol"]
        try:
            print(f"Fetching prices for {symbol}...")
            prices = client.get_historical_prices(symbol)
            if not prices.empty:
                price_frames.append(prices)
            metrics = calculate_metrics(prices)
            if metrics:
                results.append({**row.to_dict(), **metrics})
            time.sleep(0.2)
        except Exception as exc:
            results.append({
                **row.to_dict(),
                "error": str(exc)
            })

    df = pd.DataFrame(results)

    if "return_1y_pct" in df.columns:
        df = df.sort_values("return_1y_pct", ascending=False, na_position="last")
        df["rank_1y_performance"] = range(1, len(df) + 1)

    df = format_metric_columns(df)

    preferred = [
        "rank_1y_performance", "symbol", "companyName", "exchange", "sector", "industry",
        "marketCap", "as_of_date", "latest_price",
        "return_1d_pct", "return_5d_pct", "return_1m_pct",
        "return_3m_pct", "return_6m_pct", "return_1y_pct",
        "ytd_return_pct", "one_year_high", "drawdown_from_1y_high_pct",
        "observations", "error"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    price_history_file = save_price_history(price_frames)
    outputs = generate_outputs(df)
    print("Done.")
    print({"price_history": str(price_history_file)})
    print(outputs)

if __name__ == "__main__":
    main()
