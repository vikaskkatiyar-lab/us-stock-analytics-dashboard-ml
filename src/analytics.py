import pandas as pd


def _normalise_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "adj_close"}
    missing = required - set(price_history.columns)
    if missing:
        raise ValueError(f"Price history is missing required columns: {sorted(missing)}")

    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "adj_close"])
    df = df.sort_values(["symbol", "date"])
    return df


def _latest_rows_per_symbol(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if lookback_days <= 0:
        return df.iloc[0:0].copy()

    return df.groupby("symbol", group_keys=False).tail(lookback_days)


def enrich_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_price_history(price_history)
    grouped = df.groupby("symbol", group_keys=False)
    df["previous_close"] = grouped["adj_close"].shift(1)
    df["daily_move"] = df["adj_close"] - df["previous_close"]
    df["abs_daily_move"] = df["daily_move"].abs()
    df["daily_return_pct"] = (df["adj_close"] / df["previous_close"] - 1) * 100
    df["high_20d"] = grouped["adj_close"].rolling(20, min_periods=1).max().reset_index(level=0, drop=True)
    return df


def stocks_with_daily_move_between(
    price_history: pd.DataFrame,
    min_move: float,
    max_move: float,
    lookback_days: int,
) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    recent = _latest_rows_per_symbol(df, lookback_days)
    matches = recent[
        (recent["abs_daily_move"] >= min_move) &
        (recent["abs_daily_move"] <= max_move)
    ].copy()

    if matches.empty:
        return pd.DataFrame()

    summary = matches.groupby("symbol").agg(
        matching_days=("date", "count"),
        latest_match_date=("date", "max"),
        min_abs_daily_move=("abs_daily_move", "min"),
        max_abs_daily_move=("abs_daily_move", "max"),
        avg_abs_daily_move=("abs_daily_move", "mean"),
    ).reset_index()

    latest = df.groupby("symbol").tail(1)[["symbol", "date", "adj_close"]]
    latest = latest.rename(columns={"date": "latest_date", "adj_close": "latest_price"})
    summary = summary.merge(latest, on="symbol", how="left")
    return summary.sort_values(["matching_days", "max_abs_daily_move"], ascending=False)


def stocks_with_positive_closes(price_history: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    recent = _latest_rows_per_symbol(df.dropna(subset=["daily_move"]), lookback_days)

    if recent.empty:
        return pd.DataFrame()

    summary = recent.groupby("symbol").agg(
        positive_days=("daily_move", lambda moves: int((moves > 0).sum())),
        observed_days=("daily_move", "count"),
        latest_price=("adj_close", "last"),
        total_move=("daily_move", "sum"),
        total_return_pct=("daily_return_pct", lambda returns: ((returns / 100 + 1).prod() - 1) * 100),
        latest_date=("date", "last"),
    ).reset_index()

    summary = summary[
        (summary["observed_days"] >= lookback_days) &
        (summary["positive_days"] == summary["observed_days"])
    ].copy()
    return summary.sort_values("total_return_pct", ascending=False)


def stocks_with_average_daily_movement_above(
    price_history: pd.DataFrame,
    threshold: float,
    lookback_days: int,
) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    recent = _latest_rows_per_symbol(df.dropna(subset=["abs_daily_move"]), lookback_days)

    if recent.empty:
        return pd.DataFrame()

    summary = recent.groupby("symbol").agg(
        observed_days=("abs_daily_move", "count"),
        latest_price=("adj_close", "last"),
        avg_abs_daily_move=("abs_daily_move", "mean"),
        max_abs_daily_move=("abs_daily_move", "max"),
        latest_date=("date", "last"),
    ).reset_index()

    summary = summary[
        (summary["observed_days"] >= lookback_days) &
        (summary["avg_abs_daily_move"] >= threshold)
    ].copy()
    return summary.sort_values("avg_abs_daily_move", ascending=False)


def stocks_within_percent_of_20_day_high(
    price_history: pd.DataFrame,
    percent: float,
) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    latest = df.groupby("symbol").tail(1).copy()
    latest["pct_from_20d_high"] = (latest["adj_close"] / latest["high_20d"] - 1) * 100
    latest = latest[latest["pct_from_20d_high"] >= -abs(percent)].copy()
    latest = latest.rename(columns={"date": "latest_date", "adj_close": "latest_price"})
    return latest[["symbol", "latest_date", "latest_price", "high_20d", "pct_from_20d_high"]].sort_values(
        "pct_from_20d_high",
        ascending=False,
    )


def stocks_with_highest_5_day_return(price_history: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    df = _normalise_price_history(price_history)
    grouped = df.groupby("symbol", group_keys=False)
    df["close_5d_ago"] = grouped["adj_close"].shift(5)
    df["return_5d_pct"] = (df["adj_close"] / df["close_5d_ago"] - 1) * 100
    latest = df.groupby("symbol").tail(1).dropna(subset=["return_5d_pct"]).copy()
    latest = latest.rename(columns={"date": "latest_date", "adj_close": "latest_price"})
    cols = ["symbol", "latest_date", "latest_price", "close_5d_ago", "return_5d_pct"]
    return latest[cols].sort_values("return_5d_pct", ascending=False).head(top_n)


def stocks_with_predicted_next_day_range(
    price_history: pd.DataFrame,
    lookback_days: int,
    top_n: int = 25,
) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    recent = _latest_rows_per_symbol(df.dropna(subset=["abs_daily_move"]), lookback_days)

    if recent.empty:
        return pd.DataFrame()

    movement = recent.groupby("symbol").agg(
        observed_days=("abs_daily_move", "count"),
        avg_abs_daily_move=("abs_daily_move", "mean"),
        max_abs_daily_move=("abs_daily_move", "max"),
    ).reset_index()
    movement = movement[movement["observed_days"] >= lookback_days].copy()

    latest = df.groupby("symbol").tail(1)[["symbol", "date", "adj_close"]].copy()
    latest = latest.rename(columns={"date": "latest_date", "adj_close": "latest_price"})
    prediction = movement.merge(latest, on="symbol", how="left")
    prediction["predicted_next_day_low"] = prediction["latest_price"] - prediction["avg_abs_daily_move"]
    prediction["predicted_next_day_high"] = prediction["latest_price"] + prediction["avg_abs_daily_move"]
    prediction["predicted_range"] = prediction["predicted_next_day_high"] - prediction["predicted_next_day_low"]
    prediction["predicted_range_pct"] = prediction["predicted_range"] / prediction["latest_price"] * 100

    cols = [
        "symbol", "latest_date", "latest_price", "predicted_next_day_low",
        "predicted_next_day_high", "predicted_range", "predicted_range_pct",
        "avg_abs_daily_move", "max_abs_daily_move", "observed_days",
    ]
    return prediction[cols].sort_values("predicted_range_pct", ascending=False).head(top_n)


def top_intraday_play_candidates(
    price_history: pd.DataFrame,
    lookback_days: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    df = df.sort_values(["symbol", "date"])
    grouped = df.groupby("symbol", group_keys=False)
    df["ma_5"] = grouped["adj_close"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df["ma_20"] = grouped["adj_close"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    df["low_20d"] = grouped["adj_close"].rolling(20, min_periods=1).min().reset_index(level=0, drop=True)
    df["close_5d_ago"] = grouped["adj_close"].shift(5)
    df["return_5d_pct"] = (df["adj_close"] / df["close_5d_ago"] - 1) * 100

    recent = _latest_rows_per_symbol(df.dropna(subset=["abs_daily_move"]), lookback_days)
    if recent.empty:
        return pd.DataFrame()

    movement = recent.groupby("symbol").agg(
        observed_days=("abs_daily_move", "count"),
        avg_abs_daily_move=("abs_daily_move", "mean"),
        max_abs_daily_move=("abs_daily_move", "max"),
        up_days=("daily_move", lambda moves: int((moves > 0).sum())),
        avg_daily_return_pct=("daily_return_pct", "mean"),
    ).reset_index()
    movement = movement[movement["observed_days"] >= lookback_days].copy()

    latest = df.groupby("symbol").tail(1).copy()
    candidates = movement.merge(
        latest[[
            "symbol", "date", "adj_close", "high_20d", "low_20d",
            "ma_5", "ma_20", "return_5d_pct"
        ]],
        on="symbol",
        how="inner",
    )
    candidates = candidates.rename(columns={"date": "latest_date", "adj_close": "latest_price"})
    candidates["predicted_next_day_low"] = candidates["latest_price"] - candidates["avg_abs_daily_move"]
    candidates["predicted_next_day_high"] = candidates["latest_price"] + candidates["avg_abs_daily_move"]
    candidates["predicted_range"] = candidates["predicted_next_day_high"] - candidates["predicted_next_day_low"]
    candidates["predicted_range_pct"] = candidates["predicted_range"] / candidates["latest_price"] * 100
    candidates["pct_from_20d_high"] = (candidates["latest_price"] / candidates["high_20d"] - 1) * 100
    candidates["pct_above_20d_low"] = (candidates["latest_price"] / candidates["low_20d"] - 1) * 100
    candidates["trend_bonus"] = (
        (candidates["latest_price"] > candidates["ma_20"]).astype(int) +
        (candidates["ma_5"] > candidates["ma_20"]).astype(int)
    )
    candidates["intraday_pick_score"] = (
        candidates["predicted_range_pct"].rank(pct=True) * 45 +
        candidates["max_abs_daily_move"].rank(pct=True) * 20 +
        candidates["return_5d_pct"].fillna(0).rank(pct=True) * 15 +
        (100 + candidates["pct_from_20d_high"]).clip(lower=0, upper=100).rank(pct=True) * 10 +
        candidates["trend_bonus"] * 5
    )

    def build_reason(row):
        reasons = [
            f"estimated next-day range is {row['predicted_range_pct']:.2f}% of price",
            f"average daily move is ${row['avg_abs_daily_move']:.2f}",
        ]
        if pd.notna(row["return_5d_pct"]):
            direction = "positive" if row["return_5d_pct"] >= 0 else "negative"
            reasons.append(f"5-day momentum is {direction} at {row['return_5d_pct']:.2f}%")
        if pd.notna(row["pct_from_20d_high"]):
            reasons.append(f"trading {abs(row['pct_from_20d_high']):.2f}% from its 20-day high")
        if row["trend_bonus"] == 2:
            reasons.append("price and 5-day average are above the 20-day average")
        elif row["trend_bonus"] == 0:
            reasons.append("trend is weaker, so use tighter risk controls")
        return "; ".join(reasons)

    candidates["reasoning"] = candidates.apply(build_reason, axis=1)

    cols = [
        "symbol", "latest_date", "latest_price", "intraday_pick_score",
        "predicted_next_day_low", "predicted_next_day_high", "predicted_range",
        "predicted_range_pct", "avg_abs_daily_move", "max_abs_daily_move",
        "return_5d_pct", "pct_from_20d_high", "pct_above_20d_low",
        "up_days", "observed_days", "reasoning",
    ]
    return candidates[cols].sort_values("intraday_pick_score", ascending=False).head(top_n)


def sector_recent_performance(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
) -> pd.DataFrame:
    df = _normalise_price_history(price_history)
    df = df.sort_values(["symbol", "date"])
    grouped = df.groupby("symbol", group_keys=False)
    df["return_1d_pct"] = (df["adj_close"] / grouped["adj_close"].shift(1) - 1) * 100
    df["return_5d_pct"] = (df["adj_close"] / grouped["adj_close"].shift(5) - 1) * 100
    df["return_20d_pct"] = (df["adj_close"] / grouped["adj_close"].shift(20) - 1) * 100
    latest = df.groupby("symbol").tail(1).copy()

    profile_cols = [col for col in ["symbol", "companyName", "sector", "industry"] if col in daily_report.columns]
    latest = latest.merge(daily_report[profile_cols], on="symbol", how="left")
    latest = latest.dropna(subset=["sector"])

    if latest.empty:
        return pd.DataFrame()

    def top_symbols(rows, col, ascending=False):
        ranked = rows.dropna(subset=[col]).sort_values(col, ascending=ascending).head(3)
        values = []
        for _, row in ranked.iterrows():
            values.append(f"{row['symbol']} ({row[col]:.2f}%)")
        return ", ".join(values)

    rows = []
    for sector, group in latest.groupby("sector"):
        row = {
            "sector": sector,
            "stock_count": len(group),
            "avg_return_1d_pct": group["return_1d_pct"].mean(),
            "avg_return_5d_pct": group["return_5d_pct"].mean(),
            "avg_return_20d_pct": group["return_20d_pct"].mean(),
            "positive_1d_count": int((group["return_1d_pct"] > 0).sum()),
            "positive_5d_count": int((group["return_5d_pct"] > 0).sum()),
            "positive_20d_count": int((group["return_20d_pct"] > 0).sum()),
            "top_1d_stocks": top_symbols(group, "return_1d_pct"),
            "top_5d_stocks": top_symbols(group, "return_5d_pct"),
            "top_20d_stocks": top_symbols(group, "return_20d_pct"),
            "weak_1d_stocks": top_symbols(group, "return_1d_pct", ascending=True),
            "weak_5d_stocks": top_symbols(group, "return_5d_pct", ascending=True),
            "weak_20d_stocks": top_symbols(group, "return_20d_pct", ascending=True),
        }
        row["gain_reason"] = (
            f"{row['positive_5d_count']} of {row['stock_count']} stocks are positive over 5 days; "
            f"leaders: {row['top_5d_stocks'] or 'none'}; "
            f"20 day sector return is {row['avg_return_20d_pct']:.2f}%."
        )
        row["loss_reason"] = (
            f"{row['stock_count'] - row['positive_5d_count']} of {row['stock_count']} stocks are negative over 5 days; "
            f"weakest: {row['weak_5d_stocks'] or 'none'}; "
            f"1 day sector return is {row['avg_return_1d_pct']:.2f}%."
        )
        rows.append(row)

    return pd.DataFrame(rows)


def sector_forward_outlook(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    df = df.sort_values(["symbol", "date"])
    grouped = df.groupby("symbol", group_keys=False)
    df["ma_5"] = grouped["adj_close"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df["ma_20"] = grouped["adj_close"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    df["close_5d_ago"] = grouped["adj_close"].shift(5)
    df["close_20d_ago"] = grouped["adj_close"].shift(20)
    df["return_5d_pct"] = (df["adj_close"] / df["close_5d_ago"] - 1) * 100
    df["return_20d_pct"] = (df["adj_close"] / df["close_20d_ago"] - 1) * 100
    latest = df.groupby("symbol").tail(1).copy()

    recent_5 = _latest_rows_per_symbol(df.dropna(subset=["daily_return_pct"]), 5)
    recent_20 = _latest_rows_per_symbol(df.dropna(subset=["daily_return_pct"]), 20)
    avg_5 = recent_5.groupby("symbol")["daily_return_pct"].mean().rename("avg_daily_return_5d_pct")
    avg_20 = recent_20.groupby("symbol")["daily_return_pct"].mean().rename("avg_daily_return_20d_pct")
    latest = latest.merge(avg_5, on="symbol", how="left").merge(avg_20, on="symbol", how="left")

    profile_cols = [col for col in ["symbol", "companyName", "sector", "industry"] if col in daily_report.columns]
    latest = latest.merge(daily_report[profile_cols], on="symbol", how="left")
    latest = latest.dropna(subset=["sector"])
    if latest.empty:
        return pd.DataFrame()

    latest["trend_score"] = (
        (latest["adj_close"] > latest["ma_20"]).astype(int) +
        (latest["ma_5"] > latest["ma_20"]).astype(int)
    )
    latest["predicted_return_1d_pct"] = (
        latest["avg_daily_return_5d_pct"].fillna(0) * 0.55 +
        latest["avg_daily_return_20d_pct"].fillna(0) * 0.25 +
        (latest["trend_score"] - 1) * 0.15
    )
    latest["predicted_return_5d_pct"] = (
        latest["predicted_return_1d_pct"] * 3 +
        latest["return_5d_pct"].fillna(0) * 0.35
    )
    latest["predicted_return_20d_pct"] = (
        latest["predicted_return_1d_pct"] * 8 +
        latest["return_20d_pct"].fillna(0) * 0.45
    )

    rows = []
    for sector, group in latest.groupby("sector"):
        positive_5d = int((group["predicted_return_5d_pct"] > 0).sum())
        row = {
            "sector": sector,
            "stock_count": len(group),
            "predicted_return_1d_pct": group["predicted_return_1d_pct"].mean(),
            "predicted_return_5d_pct": group["predicted_return_5d_pct"].mean(),
            "predicted_return_20d_pct": group["predicted_return_20d_pct"].mean(),
            "positive_prediction_count": positive_5d,
            "negative_prediction_count": int((group["predicted_return_5d_pct"] < 0).sum()),
            "avg_trend_score": group["trend_score"].mean(),
            "avg_recent_5d_return_pct": group["return_5d_pct"].mean(),
            "avg_recent_20d_return_pct": group["return_20d_pct"].mean(),
        }
        row["gain_reason"] = (
            f"{positive_5d} of {len(group)} stocks have positive 5 day estimates; "
            f"average trend score is {row['avg_trend_score']:.2f}/2; "
            f"recent 5 day sector return is {row['avg_recent_5d_return_pct']:.2f}%."
        )
        row["loss_reason"] = (
            f"{row['negative_prediction_count']} of {len(group)} stocks have negative 5 day estimates; "
            f"average trend score is {row['avg_trend_score']:.2f}/2; "
            f"recent 20 day sector return is {row['avg_recent_20d_return_pct']:.2f}%."
        )
        rows.append(row)

    outlook = pd.DataFrame(rows)
    outlook["sector_prediction_score"] = (
        outlook["predicted_return_1d_pct"] * 0.25 +
        outlook["predicted_return_5d_pct"] * 0.35 +
        outlook["predicted_return_20d_pct"] * 0.40
    )
    return outlook.sort_values("sector_prediction_score", ascending=False)


def single_stock_trader_snapshot(
    price_history: pd.DataFrame,
    symbol: str,
    lookback_days: int = 20,
) -> dict:
    df = enrich_price_history(price_history)
    stock = df[df["symbol"] == symbol].copy()

    if stock.empty:
        return {
            "summary": pd.DataFrame(),
            "levels": pd.DataFrame(),
            "recent": pd.DataFrame(),
            "chart": pd.DataFrame(),
        }

    stock = stock.sort_values("date")
    stock["ma_5"] = stock["adj_close"].rolling(5, min_periods=1).mean()
    stock["ma_10"] = stock["adj_close"].rolling(10, min_periods=1).mean()
    stock["ma_20"] = stock["adj_close"].rolling(20, min_periods=1).mean()
    stock["ma_50"] = stock["adj_close"].rolling(50, min_periods=1).mean()
    stock["low_20d"] = stock["adj_close"].rolling(20, min_periods=1).min()
    stock["high_50d"] = stock["adj_close"].rolling(50, min_periods=1).max()
    stock["low_50d"] = stock["adj_close"].rolling(50, min_periods=1).min()

    price_changes = stock["adj_close"].diff()
    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)
    avg_gain = gains.rolling(14, min_periods=14).mean()
    avg_loss = losses.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    stock["rsi_14"] = 100 - (100 / (1 + rs))

    latest = stock.iloc[-1]
    recent_moves = stock.dropna(subset=["daily_move"]).tail(max(lookback_days, 1))
    last_5 = stock.dropna(subset=["daily_move"]).tail(5)
    last_10 = stock.dropna(subset=["daily_move"]).tail(10)
    last_20 = stock.dropna(subset=["daily_move"]).tail(20)

    def return_over(days: int):
        if len(stock) <= days:
            return None
        previous = stock["adj_close"].iloc[-days - 1]
        if previous == 0:
            return None
        return (latest["adj_close"] / previous - 1) * 100

    def avg_abs_move(rows: pd.DataFrame):
        if rows.empty:
            return None
        return rows["abs_daily_move"].mean()

    def current_streak():
        moves = stock["daily_move"].dropna()
        if moves.empty:
            return "No prior close"

        last_sign = 1 if moves.iloc[-1] > 0 else -1 if moves.iloc[-1] < 0 else 0
        if last_sign == 0:
            return "Flat today"

        count = 0
        for move in reversed(moves.tolist()):
            sign = 1 if move > 0 else -1 if move < 0 else 0
            if sign != last_sign:
                break
            count += 1

        direction = "up" if last_sign > 0 else "down"
        return f"{count} day {direction} streak"

    avg_move_5d = avg_abs_move(last_5)
    avg_move_20d = avg_abs_move(last_20)
    trend_bias = "Mixed"
    if latest["adj_close"] > latest["ma_20"] and latest["ma_5"] > latest["ma_20"]:
        trend_bias = "Bullish short-term"
    elif latest["adj_close"] < latest["ma_20"] and latest["ma_5"] < latest["ma_20"]:
        trend_bias = "Bearish short-term"

    summary = pd.DataFrame([{
        "symbol": symbol,
        "latest_date": latest["date"],
        "latest_price": latest["adj_close"],
        "previous_close": latest["previous_close"],
        "daily_move": latest["daily_move"],
        "daily_return_pct": latest["daily_return_pct"],
        "return_5d_pct": return_over(5),
        "return_10d_pct": return_over(10),
        "return_20d_pct": return_over(20),
        "ma_5": latest["ma_5"],
        "ma_10": latest["ma_10"],
        "ma_20": latest["ma_20"],
        "ma_50": latest["ma_50"],
        "price_vs_ma20_pct": (latest["adj_close"] / latest["ma_20"] - 1) * 100 if latest["ma_20"] else None,
        "rsi_14": latest["rsi_14"],
        "trend_bias": trend_bias,
        "current_streak": current_streak(),
        "up_days_last_5": int((last_5["daily_move"] > 0).sum()),
        "up_days_last_10": int((last_10["daily_move"] > 0).sum()),
        "avg_abs_move_5d": avg_move_5d,
        "avg_abs_move_10d": avg_abs_move(last_10),
        "avg_abs_move_20d": avg_move_20d,
        "volatility_20d_pct": last_20["daily_return_pct"].std(),
        "predicted_low_5d_avg_move": latest["adj_close"] - avg_move_5d if avg_move_5d is not None else None,
        "predicted_high_5d_avg_move": latest["adj_close"] + avg_move_5d if avg_move_5d is not None else None,
        "predicted_range_5d_avg_move": avg_move_5d * 2 if avg_move_5d is not None else None,
        "predicted_low_20d_avg_move": latest["adj_close"] - avg_move_20d if avg_move_20d is not None else None,
        "predicted_high_20d_avg_move": latest["adj_close"] + avg_move_20d if avg_move_20d is not None else None,
        "predicted_range_20d_avg_move": avg_move_20d * 2 if avg_move_20d is not None else None,
    }])

    levels = pd.DataFrame([{
        "latest_price": latest["adj_close"],
        "high_20d": latest["high_20d"],
        "low_20d": latest["low_20d"],
        "high_50d": latest["high_50d"],
        "low_50d": latest["low_50d"],
        "pct_from_20d_high": (latest["adj_close"] / latest["high_20d"] - 1) * 100 if latest["high_20d"] else None,
        "pct_above_20d_low": (latest["adj_close"] / latest["low_20d"] - 1) * 100 if latest["low_20d"] else None,
        "pct_from_50d_high": (latest["adj_close"] / latest["high_50d"] - 1) * 100 if latest["high_50d"] else None,
        "pct_above_50d_low": (latest["adj_close"] / latest["low_50d"] - 1) * 100 if latest["low_50d"] else None,
    }])

    recent_cols = ["date", "adj_close", "daily_move", "daily_return_pct", "ma_5", "ma_20", "rsi_14"]
    recent = stock.tail(max(lookback_days, 1))[recent_cols].copy()
    chart_cols = ["date", "adj_close", "ma_5", "ma_20", "ma_50"]
    chart = stock.tail(90)[chart_cols].copy()

    return {
        "summary": summary,
        "levels": levels,
        "recent": recent,
        "chart": chart,
    }
