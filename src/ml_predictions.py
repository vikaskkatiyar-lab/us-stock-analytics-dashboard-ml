import warnings

import pandas as pd

from ml_features import FEATURE_COLUMNS
from ml_backtest import add_high_low_predictions


def _confidence_label(probability_up: float) -> str:
    confidence = abs(probability_up - 0.5)
    if confidence >= 0.25:
        return "High"
    if confidence >= 0.12:
        return "Medium"
    return "Low"


def _feature_text(row: pd.Series, positive: bool) -> str:
    values = row[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    ranked = values.sort_values(ascending=not positive)
    if positive:
        selected = ranked[ranked > 0].head(3)
    else:
        selected = ranked[ranked < 0].head(3)
    if selected.empty:
        return "No dominant feature"
    return ", ".join([f"{name}={value:.2f}" for name, value in selected.items()])


def build_daily_predictions(
    features: pd.DataFrame,
    model,
    model_name: str,
    stock_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    latest = features.copy()
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.sort_values(["symbol", "date"]).groupby("symbol").tail(1).copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")
        probability_up = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["probability_up"] = probability_up
    latest = add_high_low_predictions(latest, probability_up)
    latest["predicted_direction"] = latest["probability_up"].apply(lambda value: "Up" if value >= 0.5 else "Down or Flat")
    latest["confidence_level"] = latest["probability_up"].apply(_confidence_label)
    latest["model_name"] = model_name
    latest["latest_price"] = latest["adj_close"]
    latest["prediction_as_of_date"] = latest["date"].dt.date.astype(str)
    latest["expected_next_trading_date"] = (latest["date"] + pd.offsets.BDay(1)).dt.date.astype(str)
    latest["latest_date"] = latest["prediction_as_of_date"]
    latest["key_positive_features"] = latest.apply(lambda row: _feature_text(row, True), axis=1)
    latest["key_negative_features"] = latest.apply(lambda row: _feature_text(row, False), axis=1)

    latest = latest.merge(stock_accuracy, on="symbol", how="left")
    latest["stock_level_accuracy"] = latest["stock_level_accuracy"].fillna(0)
    latest["average_prediction_error"] = latest["average_prediction_error"].fillna(0)

    columns = [
        "symbol",
        "companyName",
        "sector",
        "prediction_as_of_date",
        "expected_next_trading_date",
        "latest_price",
        "probability_up",
        "predicted_direction",
        "confidence_level",
        "predicted_next_day_high",
        "predicted_high_move_pct",
        "predicted_next_day_low",
        "predicted_low_move_pct",
        "predicted_next_day_range",
        "model_name",
        "stock_level_accuracy",
        "average_prediction_error",
        "stock_high_accuracy_pct",
        "stock_low_accuracy_pct",
        "stock_range_accuracy_pct",
        "key_positive_features",
        "key_negative_features",
    ]
    return latest[columns].sort_values(["probability_up", "stock_level_accuracy"], ascending=[False, False])
