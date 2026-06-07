from pathlib import Path

import pandas as pd

from analytics import enrich_price_history, sector_forward_outlook, top_intraday_play_candidates
from config import DATA_DIR, REPORTS_DIR
from historical_learning import run_historical_learning_backtest

PRICE_HISTORY_FILE = DATA_DIR / "price_history.csv"
DAILY_REPORT_FILE = REPORTS_DIR / "daily_stock_monitor.csv"
PREDICTIONS_FILE = DATA_DIR / "next_day_predictions.csv"
COMPARISONS_FILE = DATA_DIR / "next_day_prediction_results.csv"
TOP_PICKS_FILE = DATA_DIR / "top_10_intraday_picks.csv"
TOP_PICKS_RESULTS_FILE = DATA_DIR / "top_10_intraday_pick_results.csv"
SECTOR_PREDICTIONS_FILE = DATA_DIR / "sector_predictions.csv"
SECTOR_RESULTS_FILE = DATA_DIR / "sector_prediction_results.csv"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _normalise_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_next_day_predictions(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
) -> pd.DataFrame:
    df = enrich_price_history(price_history)
    recent = df.dropna(subset=["abs_daily_move"]).copy()

    if recent.empty:
        return pd.DataFrame()

    movement = recent.groupby("symbol").agg(
        avg_abs_move_5d=("abs_daily_move", lambda moves: moves.tail(5).mean()),
        avg_abs_move_20d=("abs_daily_move", lambda moves: moves.tail(20).mean()),
        observed_move_days=("abs_daily_move", "count"),
    ).reset_index()

    latest = df.groupby("symbol").tail(1)[["symbol", "date", "adj_close"]].copy()
    latest = latest.rename(columns={
        "date": "prediction_as_of_date",
        "adj_close": "latest_price",
    })

    predictions = movement.merge(latest, on="symbol", how="inner")
    predictions["predicted_low_5d"] = predictions["latest_price"] - predictions["avg_abs_move_5d"]
    predictions["predicted_high_5d"] = predictions["latest_price"] + predictions["avg_abs_move_5d"]
    predictions["predicted_low_20d"] = predictions["latest_price"] - predictions["avg_abs_move_20d"]
    predictions["predicted_high_20d"] = predictions["latest_price"] + predictions["avg_abs_move_20d"]
    predictions["predicted_range_5d"] = predictions["predicted_high_5d"] - predictions["predicted_low_5d"]
    predictions["predicted_range_20d"] = predictions["predicted_high_20d"] - predictions["predicted_low_20d"]

    profile_cols = [
        "symbol", "companyName", "sector", "industry", "marketCap",
        "rank_1y_performance", "return_1d_pct", "return_5d_pct", "return_1y_pct",
    ]
    profile_cols = [col for col in profile_cols if col in daily_report.columns]
    if profile_cols:
        predictions = predictions.merge(daily_report[profile_cols], on="symbol", how="left")

    predictions["created_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    predictions["prediction_model"] = "avg_abs_close_move"

    preferred = [
        "prediction_as_of_date", "symbol", "companyName", "sector", "industry",
        "latest_price", "predicted_low_5d", "predicted_high_5d",
        "predicted_low_20d", "predicted_high_20d", "predicted_range_5d",
        "predicted_range_20d", "avg_abs_move_5d", "avg_abs_move_20d",
        "observed_move_days", "prediction_model", "created_at",
    ]
    cols = [col for col in preferred if col in predictions.columns]
    cols += [col for col in predictions.columns if col not in cols]
    return predictions[cols].sort_values(["prediction_as_of_date", "symbol"])


def append_new_predictions(predictions: pd.DataFrame, path: Path = PREDICTIONS_FILE) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_if_exists(path)

    combined = pd.concat([existing, predictions], ignore_index=True)
    if combined.empty:
        combined.to_csv(path, index=False)
        return combined

    combined = _normalise_dates(combined, ["prediction_as_of_date"])
    combined = combined.drop_duplicates(
        subset=["prediction_as_of_date", "symbol", "prediction_model"],
        keep="last",
    )
    combined = combined.sort_values(["prediction_as_of_date", "symbol"])
    combined.to_csv(path, index=False)
    return combined


def build_prediction_comparisons(
    predictions: pd.DataFrame,
    price_history: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    predictions = _normalise_dates(predictions, ["prediction_as_of_date"])
    history = price_history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["actual_high"] = pd.to_numeric(
        history["adj_high"] if "adj_high" in history.columns else history["adj_close"],
        errors="coerce",
    )
    history["actual_low"] = pd.to_numeric(
        history["adj_low"] if "adj_low" in history.columns else history["adj_close"],
        errors="coerce",
    )
    history["actual_close"] = pd.to_numeric(history["adj_close"], errors="coerce")
    history = history.dropna(subset=["symbol", "date", "actual_high", "actual_low", "actual_close"])
    history = history.sort_values(["symbol", "date"])

    rows = []
    for _, pred in predictions.iterrows():
        symbol_history = history[
            (history["symbol"] == pred["symbol"]) &
            (history["date"] > pred["prediction_as_of_date"])
        ]
        if symbol_history.empty:
            continue

        actual = symbol_history.iloc[0]
        row = pred.to_dict()
        row.update({
            "actual_date": actual["date"],
            "actual_low": actual["actual_low"],
            "actual_high": actual["actual_high"],
            "actual_close": actual["actual_close"],
            "actual_source": "high_low" if {"adj_high", "adj_low"} <= set(price_history.columns) else "close_only_fallback",
        })

        for window in ["5d", "20d"]:
            low_col = f"predicted_low_{window}"
            high_col = f"predicted_high_{window}"
            row[f"actual_low_minus_predicted_low_{window}"] = actual["actual_low"] - pred[low_col]
            row[f"actual_high_minus_predicted_high_{window}"] = actual["actual_high"] - pred[high_col]
            row[f"actual_low_inside_prediction_{window}"] = actual["actual_low"] >= pred[low_col]
            row[f"actual_high_inside_prediction_{window}"] = actual["actual_high"] <= pred[high_col]
            row[f"actual_range_inside_prediction_{window}"] = (
                row[f"actual_low_inside_prediction_{window}"] and
                row[f"actual_high_inside_prediction_{window}"]
            )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    comparisons = pd.DataFrame(rows)
    comparisons = _normalise_dates(comparisons, ["prediction_as_of_date", "actual_date"])
    comparisons = comparisons.sort_values(["actual_date", "symbol"])
    return comparisons


def append_prediction_comparisons(
    predictions: pd.DataFrame,
    price_history: pd.DataFrame,
    path: Path = COMPARISONS_FILE,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_if_exists(path)
    new_comparisons = build_prediction_comparisons(predictions, price_history)
    combined = pd.concat([existing, new_comparisons], ignore_index=True)

    if combined.empty:
        combined.to_csv(path, index=False)
        return combined

    combined = _normalise_dates(combined, ["prediction_as_of_date", "actual_date"])
    combined = combined.drop_duplicates(
        subset=["prediction_as_of_date", "actual_date", "symbol", "prediction_model"],
        keep="last",
    )
    combined = combined.sort_values(["actual_date", "symbol"])
    combined.to_csv(path, index=False)
    return combined


def build_top_intraday_picks(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
    lookback_days: int = 5,
) -> pd.DataFrame:
    picks = top_intraday_play_candidates(price_history, lookback_days=lookback_days, top_n=10)
    if picks.empty:
        return picks

    latest_date = pd.to_datetime(picks["latest_date"], errors="coerce").max()
    picks = picks.rename(columns={"latest_date": "pick_as_of_date"})
    picks["pick_rank"] = range(1, len(picks) + 1)
    picks["pick_model"] = "intraday_range_momentum_score"
    picks["created_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

    profile_cols = [
        "symbol", "companyName", "sector", "industry", "marketCap",
        "rank_1y_performance", "return_1d_pct", "return_5d_pct", "return_1y_pct",
    ]
    profile_cols = [col for col in profile_cols if col in daily_report.columns and col not in picks.columns]
    if profile_cols:
        picks = picks.merge(daily_report[["symbol"] + profile_cols], on="symbol", how="left")

    preferred = [
        "pick_as_of_date", "pick_rank", "symbol", "companyName", "sector", "industry",
        "latest_price", "intraday_pick_score", "predicted_next_day_low",
        "predicted_next_day_high", "predicted_range", "predicted_range_pct",
        "avg_abs_daily_move", "max_abs_daily_move", "return_5d_pct",
        "pct_from_20d_high", "pct_above_20d_low", "up_days", "observed_days",
        "reasoning", "pick_model", "created_at",
    ]
    cols = [col for col in preferred if col in picks.columns]
    cols += [col for col in picks.columns if col not in cols]
    picks = picks[cols].sort_values(["pick_as_of_date", "pick_rank"])
    picks["pick_as_of_date"] = latest_date
    return picks


def append_top_intraday_picks(picks: pd.DataFrame, path: Path = TOP_PICKS_FILE) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_if_exists(path)
    combined = pd.concat([existing, picks], ignore_index=True)

    if combined.empty:
        combined.to_csv(path, index=False)
        return combined

    combined = _normalise_dates(combined, ["pick_as_of_date"])
    combined = combined.drop_duplicates(
        subset=["pick_as_of_date", "symbol", "pick_model"],
        keep="last",
    )
    combined = combined.sort_values(["pick_as_of_date", "pick_rank"])
    combined.to_csv(path, index=False)
    return combined


def build_top_pick_results(
    picks: pd.DataFrame,
    price_history: pd.DataFrame,
) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame()

    picks = _normalise_dates(picks, ["pick_as_of_date"])
    history = price_history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["actual_high"] = pd.to_numeric(
        history["adj_high"] if "adj_high" in history.columns else history["adj_close"],
        errors="coerce",
    )
    history["actual_low"] = pd.to_numeric(
        history["adj_low"] if "adj_low" in history.columns else history["adj_close"],
        errors="coerce",
    )
    history["actual_close"] = pd.to_numeric(history["adj_close"], errors="coerce")
    history = history.dropna(subset=["symbol", "date", "actual_high", "actual_low", "actual_close"])
    history = history.sort_values(["symbol", "date"])

    rows = []
    for _, pick in picks.iterrows():
        symbol_history = history[
            (history["symbol"] == pick["symbol"]) &
            (history["date"] > pick["pick_as_of_date"])
        ]
        if symbol_history.empty:
            continue

        actual = symbol_history.iloc[0]
        actual_range = actual["actual_high"] - actual["actual_low"]
        latest_price = pick["latest_price"]
        predicted_range = pick["predicted_range"]
        actual_range_pct = actual_range / latest_price * 100 if latest_price else None
        actual_close_return_pct = (actual["actual_close"] / latest_price - 1) * 100 if latest_price else None
        range_capture_pct = actual_range / predicted_range * 100 if predicted_range else None

        if range_capture_pct is None:
            result_label = "No range estimate"
        elif range_capture_pct >= 75:
            result_label = "Good"
        elif range_capture_pct >= 45:
            result_label = "Okay"
        else:
            result_label = "Weak"

        row = pick.to_dict()
        row.update({
            "actual_date": actual["date"],
            "actual_low": actual["actual_low"],
            "actual_high": actual["actual_high"],
            "actual_close": actual["actual_close"],
            "actual_intraday_range": actual_range,
            "actual_intraday_range_pct": actual_range_pct,
            "actual_close_return_pct": actual_close_return_pct,
            "range_capture_pct": range_capture_pct,
            "hit_predicted_low": actual["actual_low"] <= pick["predicted_next_day_low"],
            "hit_predicted_high": actual["actual_high"] >= pick["predicted_next_day_high"],
            "closed_up": actual["actual_close"] > latest_price,
            "pick_result": result_label,
            "actual_source": "high_low" if {"adj_high", "adj_low"} <= set(price_history.columns) else "close_only_fallback",
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    results = _normalise_dates(results, ["pick_as_of_date", "actual_date"])
    return results.sort_values(["actual_date", "pick_rank"])


def append_top_pick_results(
    picks: pd.DataFrame,
    price_history: pd.DataFrame,
    path: Path = TOP_PICKS_RESULTS_FILE,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_if_exists(path)
    new_results = build_top_pick_results(picks, price_history)
    combined = pd.concat([existing, new_results], ignore_index=True)

    if combined.empty:
        combined.to_csv(path, index=False)
        return combined

    combined = _normalise_dates(combined, ["pick_as_of_date", "actual_date"])
    combined = combined.drop_duplicates(
        subset=["pick_as_of_date", "actual_date", "symbol", "pick_model"],
        keep="last",
    )
    combined = combined.sort_values(["actual_date", "pick_rank"])
    combined.to_csv(path, index=False)
    return combined


def build_sector_predictions(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
) -> pd.DataFrame:
    outlook = sector_forward_outlook(price_history, daily_report)
    if outlook.empty:
        return outlook

    latest_date = pd.to_datetime(price_history["date"], errors="coerce").max()
    outlook = outlook.copy()
    outlook["prediction_as_of_date"] = latest_date
    outlook["sector_prediction_rank"] = range(1, len(outlook) + 1)
    outlook["prediction_model"] = "sector_momentum_trend_score"
    outlook["created_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

    preferred = [
        "prediction_as_of_date", "sector_prediction_rank", "sector", "stock_count",
        "sector_prediction_score", "predicted_return_1d_pct", "predicted_return_5d_pct",
        "predicted_return_20d_pct", "positive_prediction_count", "negative_prediction_count",
        "avg_trend_score", "avg_recent_5d_return_pct", "avg_recent_20d_return_pct",
        "gain_reason", "loss_reason", "prediction_model", "created_at",
    ]
    cols = [col for col in preferred if col in outlook.columns]
    cols += [col for col in outlook.columns if col not in cols]
    return outlook[cols].sort_values(["prediction_as_of_date", "sector_prediction_rank"])


def append_sector_predictions(
    predictions: pd.DataFrame,
    path: Path = SECTOR_PREDICTIONS_FILE,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_if_exists(path)
    combined = pd.concat([existing, predictions], ignore_index=True)

    if combined.empty:
        combined.to_csv(path, index=False)
        return combined

    combined = _normalise_dates(combined, ["prediction_as_of_date"])
    combined = combined.drop_duplicates(
        subset=["prediction_as_of_date", "sector", "prediction_model"],
        keep="last",
    )
    combined = combined.sort_values(["prediction_as_of_date", "sector_prediction_rank"])
    combined.to_csv(path, index=False)
    return combined


def build_sector_prediction_results(
    predictions: pd.DataFrame,
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    predictions = _normalise_dates(predictions, ["prediction_as_of_date"])
    history = price_history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["adj_close"] = pd.to_numeric(history["adj_close"], errors="coerce")
    profile_cols = [col for col in ["symbol", "sector"] if col in daily_report.columns]
    history = history.merge(daily_report[profile_cols], on="symbol", how="left")
    history = history.dropna(subset=["symbol", "sector", "date", "adj_close"])
    history = history.sort_values(["symbol", "date"])

    rows = []
    for _, pred in predictions.iterrows():
        sector_history = history[history["sector"] == pred["sector"]]
        if sector_history.empty:
            continue

        actual_dates = sorted(
            sector_history[sector_history["date"] > pred["prediction_as_of_date"]]["date"].dropna().unique()
        )
        if not actual_dates:
            continue

        row = pred.to_dict()
        for horizon in [1, 5, 20]:
            if len(actual_dates) < horizon:
                continue
            actual_date = actual_dates[horizon - 1]
            start = sector_history[sector_history["date"] == pred["prediction_as_of_date"]][["symbol", "adj_close"]]
            end = sector_history[sector_history["date"] == actual_date][["symbol", "adj_close"]]
            merged = start.merge(end, on="symbol", suffixes=("_start", "_end"))
            if merged.empty:
                continue
            actual_return = (merged["adj_close_end"] / merged["adj_close_start"] - 1).mean() * 100
            predicted_return = pred[f"predicted_return_{horizon}d_pct"]
            row[f"actual_date_{horizon}d"] = actual_date
            row[f"actual_return_{horizon}d_pct"] = actual_return
            row[f"prediction_error_{horizon}d_pct"] = actual_return - predicted_return
            row[f"direction_correct_{horizon}d"] = (
                (actual_return >= 0 and predicted_return >= 0) or
                (actual_return < 0 and predicted_return < 0)
            )

        if any(f"actual_return_{horizon}d_pct" in row for horizon in [1, 5, 20]):
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    date_cols = ["prediction_as_of_date", "actual_date_1d", "actual_date_5d", "actual_date_20d"]
    results = _normalise_dates(results, date_cols)
    return results.sort_values(["prediction_as_of_date", "sector_prediction_rank"])


def append_sector_prediction_results(
    predictions: pd.DataFrame,
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
    path: Path = SECTOR_RESULTS_FILE,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_if_exists(path)
    new_results = build_sector_prediction_results(predictions, price_history, daily_report)
    combined = pd.concat([existing, new_results], ignore_index=True)

    if combined.empty:
        combined.to_csv(path, index=False)
        return combined

    combined = _normalise_dates(combined, ["prediction_as_of_date"])
    combined = combined.drop_duplicates(
        subset=["prediction_as_of_date", "sector", "prediction_model"],
        keep="last",
    )
    combined = combined.sort_values(["prediction_as_of_date", "sector_prediction_rank"])
    combined.to_csv(path, index=False)
    return combined


def run_prediction_tracking_cycle() -> dict:
    price_history = pd.read_csv(PRICE_HISTORY_FILE)
    daily_report = pd.read_csv(DAILY_REPORT_FILE)
    existing_predictions = _read_csv_if_exists(PREDICTIONS_FILE)
    comparisons = append_prediction_comparisons(existing_predictions, price_history)
    new_predictions = build_next_day_predictions(price_history, daily_report)
    predictions = append_new_predictions(new_predictions)
    existing_picks = _read_csv_if_exists(TOP_PICKS_FILE)
    pick_results = append_top_pick_results(existing_picks, price_history)
    new_picks = build_top_intraday_picks(price_history, daily_report)
    picks = append_top_intraday_picks(new_picks)
    existing_sector_predictions = _read_csv_if_exists(SECTOR_PREDICTIONS_FILE)
    sector_results = append_sector_prediction_results(existing_sector_predictions, price_history, daily_report)
    new_sector_predictions = build_sector_predictions(price_history, daily_report)
    sector_predictions = append_sector_predictions(new_sector_predictions)
    learning = run_historical_learning_backtest(price_history, daily_report)

    latest_prediction_date = None
    if not new_predictions.empty:
        latest_prediction_date = pd.to_datetime(
            new_predictions["prediction_as_of_date"],
            errors="coerce",
        ).max().date().isoformat()

    return {
        "predictions_file": str(PREDICTIONS_FILE),
        "comparisons_file": str(COMPARISONS_FILE),
        "top_picks_file": str(TOP_PICKS_FILE),
        "top_picks_results_file": str(TOP_PICKS_RESULTS_FILE),
        "sector_predictions_file": str(SECTOR_PREDICTIONS_FILE),
        "sector_results_file": str(SECTOR_RESULTS_FILE),
        "new_prediction_rows": len(new_predictions),
        "total_prediction_rows": len(predictions),
        "total_comparison_rows": len(comparisons),
        "new_top_pick_rows": len(new_picks),
        "total_top_pick_rows": len(picks),
        "total_top_pick_result_rows": len(pick_results),
        "new_sector_prediction_rows": len(new_sector_predictions),
        "total_sector_prediction_rows": len(sector_predictions),
        "total_sector_result_rows": len(sector_results),
        "historical_learning_best_accuracy": learning["best_accuracy"],
        "historical_learning_iterations": learning["iterations"],
        "historical_learning_top25_rows": learning["top25_rows"],
        "latest_prediction_as_of_date": latest_prediction_date,
    }
