import pandas as pd

def _return_over_period(prices: pd.Series, days: int):
    if len(prices) <= days:
        return None

    previous = float(prices.iloc[-days-1])
    current = float(prices.iloc[-1])

    if previous == 0:
        return None

    return (current / previous - 1) * 100

def calculate_metrics(price_df: pd.DataFrame) -> dict:
    if price_df.empty or len(price_df) < 2:
        return {}

    df = price_df.sort_values("date").copy()
    prices = df["adj_close"].dropna()

    if len(prices) < 2:
        return {}

    latest_price = float(prices.iloc[-1])
    one_year_high = float(prices.tail(252).max())

    ytd_df = df[df["date"].dt.year == df["date"].iloc[-1].year]

    if len(ytd_df) > 1:
        ytd_start = float(ytd_df["adj_close"].iloc[0])
        ytd_return = (latest_price / ytd_start - 1) * 100 if ytd_start else None
    else:
        ytd_return = None

    return {
        "as_of_date": df["date"].iloc[-1].date().isoformat(),
        "latest_price": round(latest_price, 4),
        "return_1d_pct": _return_over_period(prices, 1),
        "return_5d_pct": _return_over_period(prices, 5),
        "return_1m_pct": _return_over_period(prices, 21),
        "return_3m_pct": _return_over_period(prices, 63),
        "return_6m_pct": _return_over_period(prices, 126),
        "return_1y_pct": _return_over_period(prices, 252),
        "ytd_return_pct": ytd_return,
        "one_year_high": round(one_year_high, 4),
        "drawdown_from_1y_high_pct": (latest_price / one_year_high - 1) * 100 if one_year_high else None,
        "observations": len(prices)
    }

def format_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    pct_cols = [c for c in df.columns if c.endswith("_pct")]

    for col in pct_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    if "marketCap" in df.columns:
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce").round(0)

    return df
