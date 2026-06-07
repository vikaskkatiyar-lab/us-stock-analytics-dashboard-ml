import pandas as pd


FEATURE_COLUMNS = [
    "return_1d_pct",
    "return_5d_pct",
    "return_10d_pct",
    "return_20d_pct",
    "volatility_20d",
    "average_daily_movement",
    "positive_close_streak",
    "distance_from_20d_high_pct",
    "distance_from_1y_high_pct",
    "drawdown_from_1y_high_pct",
    "volume_change_pct",
]


def _profile_frame(daily_report: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["symbol", "companyName", "sector"] if col in daily_report.columns]
    if not cols:
        return pd.DataFrame(columns=["symbol", "companyName", "sector"])
    profile = daily_report[cols].drop_duplicates("symbol").copy()
    if "companyName" not in profile.columns:
        profile["companyName"] = profile["symbol"]
    if "sector" not in profile.columns:
        profile["sector"] = "Unknown"
    return profile


def build_ml_feature_frame(price_history: pd.DataFrame, daily_report: pd.DataFrame) -> pd.DataFrame:
    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df["adj_high"] = pd.to_numeric(df["adj_high"] if "adj_high" in df.columns else df["adj_close"], errors="coerce")
    df["adj_low"] = pd.to_numeric(df["adj_low"] if "adj_low" in df.columns else df["adj_close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = 0

    df = df.dropna(subset=["symbol", "date", "adj_close"]).sort_values(["symbol", "date"])
    grouped = df.groupby("symbol", group_keys=False)
    df["return_1d_pct"] = grouped["adj_close"].pct_change() * 100
    df["return_5d_pct"] = grouped["adj_close"].pct_change(5) * 100
    df["return_10d_pct"] = grouped["adj_close"].pct_change(10) * 100
    df["return_20d_pct"] = grouped["adj_close"].pct_change(20) * 100
    df["volatility_20d"] = grouped["return_1d_pct"].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    df["daily_movement_pct"] = (df["adj_high"] - df["adj_low"]) / df["adj_close"].where(df["adj_close"] != 0) * 100
    df["average_daily_movement"] = (
        grouped["daily_movement_pct"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    )
    df["is_positive_close"] = (df["return_1d_pct"] > 0).astype(int)
    df["streak_group"] = (df["is_positive_close"] == 0).groupby(df["symbol"]).cumsum()
    df["positive_close_streak"] = df.groupby(["symbol", "streak_group"])["is_positive_close"].cumsum()
    high_20 = grouped["adj_close"].rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
    high_1y = grouped["adj_close"].rolling(252, min_periods=20).max().reset_index(level=0, drop=True)
    df["distance_from_20d_high_pct"] = (df["adj_close"] / high_20 - 1) * 100
    df["distance_from_1y_high_pct"] = (df["adj_close"] / high_1y - 1) * 100
    df["drawdown_from_1y_high_pct"] = df["distance_from_1y_high_pct"]
    df["volume_change_pct"] = grouped["volume"].pct_change() * 100
    df["target_next_close_up"] = (grouped["adj_close"].shift(-1) > df["adj_close"]).astype(int)
    df["target_date"] = grouped["date"].shift(-1)
    df["target_next_close"] = grouped["adj_close"].shift(-1)
    df["target_next_high"] = grouped["adj_high"].shift(-1)
    df["target_next_low"] = grouped["adj_low"].shift(-1)

    df = df.merge(_profile_frame(daily_report), on="symbol", how="left")
    df["companyName"] = df["companyName"].fillna(df["symbol"])
    df["sector"] = df["sector"].fillna("Unknown")
    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace([pd.NA, pd.NaT, float("inf"), float("-inf")], 0)
        .fillna(0)
    )
    return df[
        [
            "date",
            "target_date",
            "symbol",
            "companyName",
            "sector",
            "adj_close",
            "adj_high",
            "adj_low",
            "target_next_close_up",
            "target_next_close",
            "target_next_high",
            "target_next_low",
        ] + FEATURE_COLUMNS
    ].copy()


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    return features.dropna(subset=["target_date"]).copy()
