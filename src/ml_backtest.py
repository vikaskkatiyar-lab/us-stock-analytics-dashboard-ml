import warnings

import pandas as pd

from ml_features import FEATURE_COLUMNS


HIGH_LOW_TOLERANCE_PCT = 2.0


def _plain_result(direction_correct, high_miss_pct, low_miss_pct) -> str:
    if not bool(direction_correct):
        return "Failed: direction was wrong"
    average_miss = pd.Series([high_miss_pct, low_miss_pct]).dropna().mean()
    if pd.isna(average_miss):
        return "Passed direction; price range not scored"
    if average_miss <= 1:
        return "Strong: direction right and high/low close"
    if average_miss <= 2:
        return "Good: direction right and high/low acceptable"
    return "Mixed: direction right but high/low missed"


def add_high_low_predictions(frame: pd.DataFrame, probability_up) -> pd.DataFrame:
    result = frame.copy()
    probability = pd.Series(probability_up, index=result.index).astype(float).clip(0.0, 1.0)
    close = pd.to_numeric(result["adj_close"], errors="coerce")
    average_move = pd.to_numeric(result["average_daily_movement"], errors="coerce").fillna(0).clip(lower=0.25, upper=25)

    up_move_pct = (average_move * (0.35 + probability * 0.65)).clip(lower=0.10)
    down_move_pct = (average_move * (0.35 + (1 - probability) * 0.65)).clip(lower=0.10)

    result["predicted_next_day_high"] = close * (1 + up_move_pct / 100)
    result["predicted_next_day_low"] = close * (1 - down_move_pct / 100)
    result["predicted_high_move_pct"] = up_move_pct
    result["predicted_low_move_pct"] = -down_move_pct
    result["predicted_next_day_range"] = result["predicted_next_day_high"] - result["predicted_next_day_low"]
    return result


def add_high_low_review(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["actual_next_day_high"] = pd.to_numeric(result["target_next_high"], errors="coerce")
    result["actual_next_day_low"] = pd.to_numeric(result["target_next_low"], errors="coerce")
    result["actual_next_day_close"] = pd.to_numeric(result["target_next_close"], errors="coerce")
    result["actual_next_day_range"] = result["actual_next_day_high"] - result["actual_next_day_low"]

    for side in ["high", "low"]:
        predicted_col = f"predicted_next_day_{side}"
        actual_col = f"actual_next_day_{side}"
        error_col = f"{side}_prediction_error"
        error_pct_col = f"{side}_prediction_error_pct"
        accuracy_col = f"{side}_prediction_accuracy_pct"
        within_col = f"{side}_within_2pct"

        actual = pd.to_numeric(result[actual_col], errors="coerce")
        predicted = pd.to_numeric(result[predicted_col], errors="coerce")
        result[error_col] = predicted - actual
        result[error_pct_col] = result[error_col] / actual.where(actual != 0) * 100
        result[f"{side}_absolute_error_pct"] = result[error_pct_col].abs()
        result[accuracy_col] = (100 - result[error_pct_col].abs()).clip(lower=0)
        result[within_col] = result[error_pct_col].abs() <= HIGH_LOW_TOLERANCE_PCT

    result["range_prediction_error"] = result["predicted_next_day_range"] - result["actual_next_day_range"]
    result["range_prediction_error_pct"] = (
        result["range_prediction_error"] / result["actual_next_day_range"].where(result["actual_next_day_range"] != 0) * 100
    )
    result["range_absolute_error_pct"] = result["range_prediction_error_pct"].abs()
    result["range_prediction_accuracy_pct"] = (
        result[["high_prediction_accuracy_pct", "low_prediction_accuracy_pct"]].mean(axis=1)
    )
    result["high_low_within_2pct"] = result["high_within_2pct"] & result["low_within_2pct"]
    return result


def build_backtest(test: pd.DataFrame, model, model_name: str) -> pd.DataFrame:
    frame = test.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")
        probability_up = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    predicted = (probability_up >= 0.5).astype(int)
    actual = frame["target_next_close_up"].astype(int)
    predicted_direction = pd.Series(predicted, index=frame.index).map({1: "Up", 0: "Down or Flat"})
    actual_direction = actual.map({1: "Up", 0: "Down or Flat"})
    correct_prediction = predicted == actual
    prediction_review = [
        f"{'Correct' if is_correct else 'Wrong'} predicted {direction}"
        for is_correct, direction in zip(correct_prediction, predicted_direction)
    ]
    frame = add_high_low_predictions(frame, probability_up)
    frame = add_high_low_review(frame)
    backtest = pd.DataFrame(
        {
            "prediction_as_of_date": frame["date"],
            "actual_review_date": frame["target_date"],
            "date": frame["date"],
            "symbol": frame["symbol"],
            "companyName": frame.get("companyName", frame["symbol"]),
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "prediction_review": prediction_review,
            "probability_up": probability_up,
            "correct_prediction": correct_prediction,
            "direction_score_pct": pd.Series(correct_prediction, index=frame.index).astype(int) * 100,
            "prediction_error": (predicted - actual).astype(int),
            "prediction_close": frame["adj_close"],
            "predicted_next_day_high": frame["predicted_next_day_high"],
            "actual_next_day_high": frame["actual_next_day_high"],
            "high_prediction_error": frame["high_prediction_error"],
            "high_prediction_error_pct": frame["high_prediction_error_pct"],
            "high_absolute_error_pct": frame["high_absolute_error_pct"],
            "high_prediction_accuracy_pct": frame["high_prediction_accuracy_pct"],
            "high_within_2pct": frame["high_within_2pct"],
            "predicted_next_day_low": frame["predicted_next_day_low"],
            "actual_next_day_low": frame["actual_next_day_low"],
            "low_prediction_error": frame["low_prediction_error"],
            "low_prediction_error_pct": frame["low_prediction_error_pct"],
            "low_absolute_error_pct": frame["low_absolute_error_pct"],
            "low_prediction_accuracy_pct": frame["low_prediction_accuracy_pct"],
            "low_within_2pct": frame["low_within_2pct"],
            "predicted_next_day_range": frame["predicted_next_day_range"],
            "actual_next_day_range": frame["actual_next_day_range"],
            "range_prediction_error_pct": frame["range_prediction_error_pct"],
            "range_absolute_error_pct": frame["range_absolute_error_pct"],
            "range_prediction_accuracy_pct": frame["range_prediction_accuracy_pct"],
            "high_low_within_2pct": frame["high_low_within_2pct"],
            "model_name": model_name,
        }
    )
    backtest["combined_accuracy_pct"] = backtest[
        ["direction_score_pct", "high_prediction_accuracy_pct", "low_prediction_accuracy_pct"]
    ].mean(axis=1)
    backtest["plain_overall_accuracy_pct"] = backtest.apply(
        lambda row: 0
        if not bool(row["correct_prediction"])
        else pd.Series([row["high_prediction_accuracy_pct"], row["low_prediction_accuracy_pct"]]).mean(),
        axis=1,
    )
    backtest["plain_result"] = backtest.apply(
        lambda row: _plain_result(
            row["correct_prediction"],
            row["high_absolute_error_pct"],
            row["low_absolute_error_pct"],
        ),
        axis=1,
    )
    return backtest.sort_values(["prediction_as_of_date", "symbol"])


def build_accuracy_by_date(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame()
    frame = backtest.copy()
    frame["confidence"] = (pd.to_numeric(frame["probability_up"], errors="coerce") - 0.5).abs() * 2
    summary = frame.groupby("prediction_as_of_date").agg(
        number_of_predictions=("correct_prediction", "count"),
        correct_predictions=("correct_prediction", "sum"),
        direction_accuracy=("correct_prediction", "mean"),
        average_direction_score_pct=("direction_score_pct", "mean"),
        high_success_rate=("high_within_2pct", "mean"),
        low_success_rate=("low_within_2pct", "mean"),
        high_low_within_2pct_rate=("high_low_within_2pct", "mean"),
        average_high_miss_pct=("high_absolute_error_pct", "mean"),
        average_low_miss_pct=("low_absolute_error_pct", "mean"),
        average_high_accuracy_pct=("high_prediction_accuracy_pct", "mean"),
        average_low_accuracy_pct=("low_prediction_accuracy_pct", "mean"),
        average_range_accuracy_pct=("range_prediction_accuracy_pct", "mean"),
        average_combined_accuracy_pct=("combined_accuracy_pct", "mean"),
        average_plain_overall_accuracy_pct=("plain_overall_accuracy_pct", "mean"),
        average_probability_up=("probability_up", "mean"),
        average_confidence=("confidence", "mean"),
    ).reset_index()
    return summary.rename(columns={"prediction_as_of_date": "date"}).sort_values("date")


def build_accuracy_by_stock(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame(columns=["symbol", "stock_level_accuracy", "average_prediction_error"])
    frame = backtest.copy()
    frame["absolute_prediction_error"] = pd.to_numeric(frame["prediction_error"], errors="coerce").abs()
    return frame.groupby("symbol").agg(
        stock_level_accuracy=("correct_prediction", "mean"),
        average_prediction_error=("absolute_prediction_error", "mean"),
        stock_high_accuracy_pct=("high_prediction_accuracy_pct", "mean"),
        stock_low_accuracy_pct=("low_prediction_accuracy_pct", "mean"),
        stock_range_accuracy_pct=("range_prediction_accuracy_pct", "mean"),
        stock_plain_overall_accuracy_pct=("plain_overall_accuracy_pct", "mean"),
    ).reset_index()
