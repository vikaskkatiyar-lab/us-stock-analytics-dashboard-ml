import warnings

import pandas as pd

from config import DATA_DIR, REPORTS_DIR

PRICE_HISTORY_FILE = DATA_DIR / "price_history.csv"
DAILY_REPORT_FILE = REPORTS_DIR / "daily_stock_monitor.csv"
LEARNING_SUMMARY_FILE = DATA_DIR / "historical_learning_summary.csv"
LEARNING_BACKTEST_FILE = DATA_DIR / "historical_learning_backtest.csv"
LEARNING_SIGNALS_FILE = DATA_DIR / "historical_learning_signals.csv"
LEARNING_WEIGHTS_FILE = DATA_DIR / "historical_learning_weights.csv"
LEARNING_TOP25_FILE = DATA_DIR / "historical_learning_top25_recent_accuracy.csv"
TRADING212_CFD_SYMBOLS_FILE = DATA_DIR / "trading212_cfd_symbols.csv"

FEATURE_NAMES = [
    "return_1d_lag_1", "return_1d_lag_2", "return_1d_lag_3", "return_1d_lag_4", "return_1d_lag_5",
    "return_1d_lag_6", "return_1d_lag_7", "return_1d_lag_8", "return_1d_lag_9", "return_1d_lag_10",
    "avg_return_2d", "avg_return_3d", "avg_return_5d", "avg_return_10d", "avg_return_20d",
    "sum_return_2d", "sum_return_3d", "sum_return_5d", "sum_return_10d", "sum_return_20d",
    "volatility_3d", "volatility_5d", "volatility_10d", "volatility_20d",
    "avg_abs_move_3d", "avg_abs_move_5d", "avg_abs_move_10d", "avg_abs_move_20d",
    "price_vs_ma_3d", "price_vs_ma_5d", "price_vs_ma_10d", "price_vs_ma_20d", "price_vs_ma_50d",
    "up_day_ratio_3d", "up_day_ratio_5d", "up_day_ratio_10d", "up_day_ratio_20d",
    "down_day_ratio_3d", "down_day_ratio_5d", "down_day_ratio_10d", "down_day_ratio_20d",
    "distance_from_20d_high", "distance_from_20d_low", "distance_from_50d_high", "distance_from_50d_low",
    "market_return_1d", "market_return_5d", "sector_return_1d", "sector_return_5d", "relative_strength_5d",
]


def _load_inputs():
    price_history = pd.read_csv(PRICE_HISTORY_FILE)
    daily_report = pd.read_csv(DAILY_REPORT_FILE)
    return price_history, daily_report


def _normalise(price_history: pd.DataFrame, daily_report: pd.DataFrame) -> pd.DataFrame:
    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "adj_close"]).sort_values(["symbol", "date"])

    profile = daily_report[[col for col in ["symbol", "companyName", "sector", "industry"] if col in daily_report.columns]]
    if "sector" in profile.columns:
        df = df.merge(profile, on="symbol", how="left")
    else:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].fillna("Unknown")
    return df


def build_feature_frame(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    df = _normalise(price_history, daily_report)
    grouped = df.groupby("symbol", group_keys=False)
    df["daily_return_pct"] = grouped["adj_close"].pct_change() * 100
    df["target_next_return_pct"] = grouped["daily_return_pct"].shift(-1)
    df["target_date"] = grouped["date"].shift(-1)
    df["target_up"] = (df["target_next_return_pct"] > 0).astype(int)

    for lag in range(1, 11):
        df[f"return_1d_lag_{lag}"] = grouped["daily_return_pct"].shift(lag)

    for window in [2, 3, 5, 10, 20]:
        shifted = grouped["daily_return_pct"].shift(1)
        df[f"avg_return_{window}d"] = shifted.groupby(df["symbol"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"sum_return_{window}d"] = shifted.groupby(df["symbol"]).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)

    for window in [3, 5, 10, 20]:
        shifted = grouped["daily_return_pct"].shift(1)
        df[f"volatility_{window}d"] = shifted.groupby(df["symbol"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        df[f"avg_abs_move_{window}d"] = shifted.abs().groupby(df["symbol"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

    for window in [3, 5, 10, 20, 50]:
        ma = grouped["adj_close"].shift(1).groupby(df["symbol"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"price_vs_ma_{window}d"] = (df["adj_close"] / ma - 1) * 100

    for window in [3, 5, 10, 20]:
        up = (grouped["daily_return_pct"].shift(1) > 0).astype(int)
        df[f"up_day_ratio_{window}d"] = up.groupby(df["symbol"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"down_day_ratio_{window}d"] = 1 - df[f"up_day_ratio_{window}d"]

    for window in [20, 50]:
        highs = grouped["adj_close"].shift(1).groupby(df["symbol"]).rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
        lows = grouped["adj_close"].shift(1).groupby(df["symbol"]).rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
        df[f"distance_from_{window}d_high"] = (df["adj_close"] / highs - 1) * 100
        df[f"distance_from_{window}d_low"] = (df["adj_close"] / lows - 1) * 100

    market = df.groupby("date")["daily_return_pct"].mean().sort_index()
    market_1d = market.shift(1).rename("market_return_1d")
    market_5d = market.shift(1).rolling(5, min_periods=1).mean().rename("market_return_5d")
    df = df.merge(market_1d, on="date", how="left").merge(market_5d, on="date", how="left")

    sector = df.groupby(["date", "sector"])["daily_return_pct"].mean().sort_index()
    sector_1d = sector.groupby("sector").shift(1).rename("sector_return_1d").reset_index()
    sector_5d = sector.groupby("sector").shift(1).groupby(level=1).rolling(5, min_periods=1).mean()
    sector_5d = sector_5d.reset_index(level=0, drop=True).rename("sector_return_5d").reset_index()
    df = df.merge(sector_1d, on=["date", "sector"], how="left")
    df = df.merge(sector_5d, on=["date", "sector"], how="left")
    df["relative_strength_5d"] = df["sum_return_5d"] - df["market_return_5d"]

    keep_cols = [
        "date", "symbol", "companyName", "sector", "industry", "adj_close",
        "target_date", "target_next_return_pct", "target_up",
    ] + FEATURE_NAMES
    keep_cols = [col for col in keep_cols if col in df.columns]
    features = df[keep_cols].copy()
    features[FEATURE_NAMES] = features[FEATURE_NAMES].replace([pd.NA, pd.NaT], 0).fillna(0)
    if drop_missing_target:
        features = features.dropna(subset=["target_next_return_pct"])
    return features


def _learn_weights(train: pd.DataFrame, feature_names: list[str]) -> pd.Series:
    weights = {}
    target = train["target_up"].replace({0: -1, 1: 1})
    for feature in feature_names:
        values = pd.to_numeric(train[feature], errors="coerce").fillna(0)
        std = values.std()
        if not std:
            weights[feature] = 0
            continue
        weights[feature] = float((values / std * target).mean())
    weights = pd.Series(weights)
    total = weights.abs().sum()
    if total:
        weights = weights / total
    return weights


def _score(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    values = frame[weights.index].apply(pd.to_numeric, errors="coerce").fillna(0)
    std = values.std().replace(0, 1)
    return (values / std).fillna(0).dot(weights)


def _ensure_trading212_cfd_symbols(daily_report: pd.DataFrame) -> pd.DataFrame:
    if TRADING212_CFD_SYMBOLS_FILE.exists():
        return pd.read_csv(TRADING212_CFD_SYMBOLS_FILE)

    cols = [col for col in ["symbol", "companyName", "sector", "industry"] if col in daily_report.columns]
    symbols = daily_report[cols].drop_duplicates("symbol").copy()
    symbols["trading212_cfd_available"] = True
    symbols["source"] = "seeded_from_us_top250_pending_platform_verification"
    symbols.to_csv(TRADING212_CFD_SYMBOLS_FILE, index=False)
    return symbols


def _available_cfd_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _price_accuracy_pct(predicted: pd.Series, actual: pd.Series) -> pd.Series:
    error_pct = (predicted - actual).abs() / actual.abs().where(actual != 0) * 100
    return (100 - error_pct).clip(lower=0, upper=100)


def _direction_result_label(row: pd.Series) -> str:
    predicted = row.get("latest_prediction")
    actual = row.get("latest_actual")
    if pd.isna(predicted) or pd.isna(actual):
        return pd.NA
    prefix = "Correct" if predicted == actual else "Wrong"
    return f"{prefix} predicted {predicted}"


def _prediction_direction_label(values: pd.Series) -> pd.Series:
    return values.map({1: "Up", 0: "Down"})


def _latest_prediction_ranges(price_history: pd.DataFrame) -> pd.DataFrame:
    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "adj_close"]).sort_values(["symbol", "date"])
    grouped = df.groupby("symbol", group_keys=False)
    df["daily_move_abs"] = grouped["adj_close"].diff().abs()
    recent = df.dropna(subset=["daily_move_abs"]).groupby("symbol").tail(20)
    movement = recent.groupby("symbol").agg(
        avg_abs_move_5d=("daily_move_abs", lambda values: values.tail(5).mean()),
        avg_abs_move_20d=("daily_move_abs", "mean"),
    ).reset_index()
    latest = df.groupby("symbol").tail(1)[["symbol", "date", "adj_close"]].rename(
        columns={"date": "latest_price_date", "adj_close": "current_latest_price"}
    )
    ranges = latest.merge(movement, on="symbol", how="left")
    ranges["predicted_next_day_low"] = ranges["current_latest_price"] - ranges["avg_abs_move_5d"]
    ranges["predicted_next_day_high"] = ranges["current_latest_price"] + ranges["avg_abs_move_5d"]
    ranges["predicted_next_5d_low"] = ranges["current_latest_price"] - ranges["avg_abs_move_20d"] * (5 ** 0.5)
    ranges["predicted_next_5d_high"] = ranges["current_latest_price"] + ranges["avg_abs_move_20d"] * (5 ** 0.5)
    return ranges


def _add_backtest_range_results(backtest: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    history = price_history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["adj_close"] = pd.to_numeric(history["adj_close"], errors="coerce")
    history["adj_high"] = pd.to_numeric(
        history["adj_high"] if "adj_high" in history.columns else history["adj_close"],
        errors="coerce",
    )
    history["adj_low"] = pd.to_numeric(
        history["adj_low"] if "adj_low" in history.columns else history["adj_close"],
        errors="coerce",
    )
    history = history.dropna(subset=["symbol", "date", "adj_close", "adj_high", "adj_low"])
    history = history.sort_values(["symbol", "date"])

    grouped = history.groupby("symbol", group_keys=False)
    history["up_move"] = (history["adj_high"] - history["adj_close"]).clip(lower=0)
    history["down_move"] = (history["adj_close"] - history["adj_low"]).clip(lower=0)
    history["predicted_up_move"] = (
        grouped["up_move"].shift(1).groupby(history["symbol"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    history["predicted_down_move"] = (
        grouped["down_move"].shift(1).groupby(history["symbol"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    history["predicted_next_day_low"] = history["adj_close"] - history["predicted_down_move"]
    history["predicted_next_day_high"] = history["adj_close"] + history["predicted_up_move"]

    predictions = history[
        ["symbol", "date", "adj_close", "predicted_next_day_low", "predicted_next_day_high"]
    ].rename(columns={"adj_close": "prediction_close"})
    actuals = history[["symbol", "date", "adj_low", "adj_high", "adj_close"]].rename(
        columns={
            "date": "target_date",
            "adj_low": "actual_low",
            "adj_high": "actual_high",
            "adj_close": "actual_close",
        }
    )

    results = backtest.merge(predictions, on=["symbol", "date"], how="left")
    results = results.merge(actuals, on=["symbol", "target_date"], how="left")
    results["predicted_range"] = results["predicted_next_day_high"] - results["predicted_next_day_low"]
    results["actual_range"] = results["actual_high"] - results["actual_low"]
    results["low_prediction_accuracy_pct"] = _price_accuracy_pct(results["predicted_next_day_low"], results["actual_low"])
    results["high_prediction_accuracy_pct"] = _price_accuracy_pct(results["predicted_next_day_high"], results["actual_high"])
    results["range_prediction_accuracy_pct"] = results[
        ["low_prediction_accuracy_pct", "high_prediction_accuracy_pct"]
    ].mean(axis=1)
    return results


def _merge_date_stamped_top25(summary: pd.DataFrame, path=LEARNING_TOP25_FILE) -> pd.DataFrame:
    if summary.empty or "latest_actual_date" not in summary.columns:
        return summary

    actual_dates = pd.to_datetime(summary["latest_actual_date"], errors="coerce").dropna()
    if actual_dates.empty:
        return summary

    actual_date = actual_dates.max().date().isoformat()
    metric_order = [
        "prediction_date", "predicted_direction", "actual_direction",
        "direction_result", "direction_accuracy_pct",
        "predicted_close", "actual_close", "predicted_low", "actual_low",
        "predicted_high", "actual_high", "predicted_range", "actual_range",
        "low_accuracy_pct", "high_accuracy_pct", "range_accuracy_pct",
    ]
    date_cols = {
        "latest_prediction_date": f"{actual_date}_prediction_date",
        "latest_prediction": f"{actual_date}_predicted_direction",
        "latest_actual": f"{actual_date}_actual_direction",
        "direction_result": f"{actual_date}_direction_result",
        "direction_accuracy_pct": f"{actual_date}_direction_accuracy_pct",
        "prediction_close": f"{actual_date}_predicted_close",
        "actual_close": f"{actual_date}_actual_close",
        "predicted_next_day_low": f"{actual_date}_predicted_low",
        "actual_low": f"{actual_date}_actual_low",
        "predicted_next_day_high": f"{actual_date}_predicted_high",
        "actual_high": f"{actual_date}_actual_high",
        "predicted_range": f"{actual_date}_predicted_range",
        "actual_range": f"{actual_date}_actual_range",
        "low_prediction_accuracy_pct": f"{actual_date}_low_accuracy_pct",
        "high_prediction_accuracy_pct": f"{actual_date}_high_accuracy_pct",
        "range_prediction_accuracy_pct": f"{actual_date}_range_accuracy_pct",
    }
    today_cols = ["symbol", "companyName"] + [col for col in date_cols if col in summary.columns]
    today = summary[today_cols].rename(columns=date_cols)

    if not path.exists():
        merged = today
    else:
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()

        if existing.empty or "symbol" not in existing.columns:
            merged = today
        else:
            existing = existing[existing["symbol"] != "AVERAGE"]
            keep_existing = ["symbol", "companyName"] + [
                col for col in existing.columns
                if col not in {"symbol", "companyName"} and col[:10].count("-") == 2
            ]
            existing = existing[[col for col in keep_existing if col in existing.columns]]
            drop_today_cols = [
                col for col in today.columns
                if col in existing.columns and col not in {"symbol", "companyName"}
            ]
            existing = existing.drop(columns=drop_today_cols)
            merged = existing.merge(today, on=["symbol", "companyName"], how="outer")

    date_prefixes = sorted({
        col[:10] for col in merged.columns
        if col not in {"symbol", "companyName"} and col[:10].count("-") == 2
    })
    ordered_dated_cols = []
    for prefix in date_prefixes:
        ordered_dated_cols.extend(
            [f"{prefix}_{metric}" for metric in metric_order if f"{prefix}_{metric}" in merged.columns]
        )
    sort_col = f"{actual_date}_range_accuracy_pct"
    if sort_col in merged.columns:
        merged = merged.sort_values(sort_col, ascending=False, na_position="last")
    return merged[["symbol", "companyName"] + ordered_dated_cols]


def _add_next_day_prediction_columns(
    table: pd.DataFrame,
    features: pd.DataFrame,
    weights: pd.Series,
    price_history: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return table

    latest = features.copy()
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.sort_values(["symbol", "date"]).groupby("symbol").tail(1)
    latest["next_day_predicted_direction"] = _prediction_direction_label(
        (_score(latest, weights) >= 0).astype(int)
    )

    ranges = _latest_prediction_ranges(price_history)
    next_day = latest[["symbol", "date", "next_day_predicted_direction"]].rename(
        columns={"date": "next_day_prediction_as_of_date"}
    )
    next_day = next_day.merge(
        ranges[["symbol", "predicted_next_day_low", "predicted_next_day_high"]],
        on="symbol",
        how="left",
    )
    next_day["next_day_predicted_range"] = (
        next_day["predicted_next_day_high"] - next_day["predicted_next_day_low"]
    )
    next_day = next_day.rename(
        columns={
            "predicted_next_day_low": "next_day_predicted_low",
            "predicted_next_day_high": "next_day_predicted_high",
        }
    )
    cols = [
        "symbol", "next_day_prediction_as_of_date", "next_day_predicted_direction",
        "next_day_predicted_low", "next_day_predicted_high", "next_day_predicted_range",
    ]
    table = table.drop(columns=[col for col in cols if col != "symbol" and col in table.columns])
    merged = table.merge(next_day[cols], on="symbol", how="left")
    next_cols = [col for col in cols if col != "symbol" and col in merged.columns]
    front_cols = [col for col in ["symbol", "companyName"] if col in merged.columns]
    other_cols = [col for col in merged.columns if col not in front_cols + next_cols]
    return merged[front_cols + next_cols + other_cols]


def _add_average_accuracy_row(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table

    table = table[table["symbol"] != "AVERAGE"].copy()
    average = {col: pd.NA for col in table.columns}
    average["symbol"] = "AVERAGE"
    average["companyName"] = "Average accuracy"

    for col in table.columns:
        if (
            col.endswith("_direction_accuracy_pct") or
            col.endswith("_low_accuracy_pct") or
            col.endswith("_high_accuracy_pct") or
            col.endswith("_range_accuracy_pct")
        ):
            average[col] = pd.to_numeric(table[col], errors="coerce").mean()

    result = table.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        result.loc[-1] = average
    result.index = result.index + 1
    return result.sort_index().reset_index(drop=True)



def _build_top25_recent_accuracy(
    backtest: pd.DataFrame,
    daily_report: pd.DataFrame,
    price_history: pd.DataFrame,
    latest_features: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame()

    scored = _add_backtest_range_results(backtest, price_history)
    scored["date"] = pd.to_datetime(scored["date"], errors="coerce")
    if "target_date" in scored.columns:
        scored["target_date"] = pd.to_datetime(scored["target_date"], errors="coerce")
    recent_dates = sorted(scored["date"].dropna().unique())[-5:]
    recent = scored[scored["date"].isin(recent_dates)].sort_values(["symbol", "date"]).copy()
    if recent.empty:
        return pd.DataFrame()

    summary = recent.groupby("symbol").agg(
        range_prediction_accuracy_pct=("range_prediction_accuracy_pct", "last"),
        latest_prediction_date=("date", "last"),
        latest_actual_date=("target_date", "last"),
        latest_predicted_up=("predicted_up", "last"),
        latest_actual_up=("target_up", "last"),
        prediction_close=("prediction_close", "last"),
        predicted_next_day_low=("predicted_next_day_low", "last"),
        predicted_next_day_high=("predicted_next_day_high", "last"),
        actual_low=("actual_low", "last"),
        actual_high=("actual_high", "last"),
        actual_close=("actual_close", "last"),
        predicted_range=("predicted_range", "last"),
        actual_range=("actual_range", "last"),
        low_prediction_accuracy_pct=("low_prediction_accuracy_pct", "last"),
        high_prediction_accuracy_pct=("high_prediction_accuracy_pct", "last"),
    ).reset_index()

    profile_cols = [
        "symbol", "companyName",
    ]
    profile_cols = [col for col in profile_cols if col in daily_report.columns]
    if profile_cols:
        summary = summary.merge(daily_report[profile_cols], on="symbol", how="left")

    cfd_symbols = _ensure_trading212_cfd_symbols(daily_report)
    if "trading212_cfd_available" in cfd_symbols.columns:
        cfd_symbols = cfd_symbols[_available_cfd_mask(cfd_symbols["trading212_cfd_available"])]
    cfd_cols = [col for col in ["symbol", "source"] if col in cfd_symbols.columns]
    cfd_symbols = cfd_symbols[cfd_cols].drop_duplicates("symbol")
    summary = summary.merge(
        cfd_symbols.rename(columns={"source": "trading212_cfd_source"}),
        on="symbol",
        how="inner",
    )

    summary["latest_prediction"] = summary["latest_predicted_up"].map({1: "Up", 0: "Down"})
    summary["latest_actual"] = summary["latest_actual_up"].map({1: "Up", 0: "Down"})
    summary["direction_result"] = summary.apply(_direction_result_label, axis=1)
    summary["direction_accuracy_pct"] = (summary["latest_prediction"] == summary["latest_actual"]).astype(int) * 100
    summary = summary.sort_values(
        ["range_prediction_accuracy_pct", "symbol"],
        ascending=[False, True],
    ).head(25)

    preferred = [
        "symbol", "companyName", "latest_prediction_date", "latest_actual_date",
        "latest_prediction", "latest_actual", "direction_result", "direction_accuracy_pct",
        "prediction_close", "actual_close",
        "predicted_next_day_low", "actual_low", "predicted_next_day_high",
        "actual_high", "predicted_range", "actual_range",
        "low_prediction_accuracy_pct", "high_prediction_accuracy_pct",
        "range_prediction_accuracy_pct",
    ]
    cols = [col for col in preferred if col in summary.columns]
    table = _merge_date_stamped_top25(summary[cols])
    table = _add_next_day_prediction_columns(table, latest_features, weights, price_history)
    return _add_average_accuracy_row(table)


def run_historical_learning_backtest(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
    train_days: int = 50,
    test_days: int = 50,
    max_iterations: int = 15,
    target_accuracy: float = 0.70,
) -> dict:
    features = build_feature_frame(price_history, daily_report)
    latest_features = build_feature_frame(price_history, daily_report, drop_missing_target=False)
    dates = sorted(features["date"].dropna().unique())
    if len(dates) < train_days + test_days + 1:
        raise ValueError("Not enough daily history for the requested train and backtest windows.")

    test_dates = dates[-test_days:]
    base_train_dates = dates[-(train_days + test_days):-test_days]
    train = features[features["date"].isin(base_train_dates)].copy()
    test = features[features["date"].isin(test_dates)].copy()

    weights = _learn_weights(train, FEATURE_NAMES)
    best_weights = weights.copy()
    best_accuracy = 0
    best_results = pd.DataFrame()
    summary_rows = []

    for iteration in range(1, max_iterations + 1):
        scored = test.copy()
        scored["prediction_score"] = _score(scored, weights)
        scored["predicted_up"] = (scored["prediction_score"] >= 0).astype(int)
        scored["correct"] = scored["predicted_up"] == scored["target_up"]
        accuracy = float(scored["correct"].mean())
        summary_rows.append({
            "iteration": iteration,
            "accuracy": accuracy,
            "correct_predictions": int(scored["correct"].sum()),
            "total_predictions": len(scored),
            "train_days": train_days,
            "test_days": test_days,
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_results = scored
            best_weights = weights.copy()

        if accuracy >= target_accuracy:
            break

        mistakes = scored[~scored["correct"]].copy()
        if mistakes.empty:
            break
        correction_target = mistakes["target_up"].replace({0: -1, 1: 1})
        correction = {}
        for feature in FEATURE_NAMES:
            values = pd.to_numeric(mistakes[feature], errors="coerce").fillna(0)
            std = values.std() or 1
            correction[feature] = float((values / std * correction_target).mean())
        correction = pd.Series(correction)
        weights = weights + correction * 0.05
        total = weights.abs().sum()
        if total:
            weights = weights / total

    summary = pd.DataFrame(summary_rows)
    top25 = _build_top25_recent_accuracy(best_results, daily_report, price_history, latest_features, best_weights)
    signal_weights = best_weights.sort_values(key=lambda values: values.abs(), ascending=False).reset_index()
    signal_weights.columns = ["signal", "weight"]
    signal_weights["rank"] = range(1, len(signal_weights) + 1)
    signal_weights = signal_weights[["rank", "signal", "weight"]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(LEARNING_SUMMARY_FILE, index=False)
    best_results.to_csv(LEARNING_BACKTEST_FILE, index=False)
    top25.to_csv(LEARNING_TOP25_FILE, index=False)
    signal_weights.to_csv(LEARNING_WEIGHTS_FILE, index=False)
    pd.DataFrame({"signal": FEATURE_NAMES, "description": FEATURE_NAMES}).to_csv(LEARNING_SIGNALS_FILE, index=False)

    return {
        "summary_file": str(LEARNING_SUMMARY_FILE),
        "backtest_file": str(LEARNING_BACKTEST_FILE),
        "signals_file": str(LEARNING_SIGNALS_FILE),
        "weights_file": str(LEARNING_WEIGHTS_FILE),
        "top25_file": str(LEARNING_TOP25_FILE),
        "trading212_cfd_symbols_file": str(TRADING212_CFD_SYMBOLS_FILE),
        "best_accuracy": best_accuracy,
        "iterations": len(summary),
        "total_predictions": len(best_results),
        "top25_rows": len(top25),
    }


def run_historical_learning_cycle() -> dict:
    price_history, daily_report = _load_inputs()
    return run_historical_learning_backtest(price_history, daily_report)


if __name__ == "__main__":
    print(run_historical_learning_cycle())
