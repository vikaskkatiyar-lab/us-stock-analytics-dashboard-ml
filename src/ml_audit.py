import pandas as pd
from typing import Optional


def _stock_label(row: pd.Series) -> str:
    symbol = str(row.get("symbol", row.get("Stock", "")))
    company = row.get("companyName", "")
    if pd.isna(company) or not str(company).strip() or str(company) == symbol:
        return symbol
    return f"{symbol} - {company}"


def build_plain_prediction_audit(review_history: pd.DataFrame) -> pd.DataFrame:
    if review_history.empty:
        return pd.DataFrame()

    frame = review_history.copy()
    frame["prediction_as_of_date"] = pd.to_datetime(frame["prediction_as_of_date"], errors="coerce").dt.date.astype(str)
    frame["actual_review_date"] = pd.to_datetime(frame["actual_review_date"], errors="coerce").dt.date.astype(str)

    result = pd.DataFrame(
        {
            "Prediction Date": frame["prediction_as_of_date"],
            "Checked Against Date": frame["actual_review_date"],
            "Stock": frame.apply(_stock_label, axis=1),
            "Predicted Up/Down": frame["predicted_direction"],
            "Actual Up/Down": frame["actual_direction"],
            "Direction Result": frame["prediction_review"],
            "Direction Score %": frame["direction_score_pct"],
            "Predicted High": frame["predicted_next_day_high"],
            "Actual High": frame["actual_next_day_high"],
            "High Miss %": frame["high_absolute_error_pct"],
            "Predicted Low": frame["predicted_next_day_low"],
            "Actual Low": frame["actual_next_day_low"],
            "Low Miss %": frame["low_absolute_error_pct"],
            "Overall Accuracy %": frame["plain_overall_accuracy_pct"],
            "Overall Result": frame["plain_result"],
        }
    )
    return result


def build_daily_accuracy_trend(review_history: pd.DataFrame) -> pd.DataFrame:
    if review_history.empty:
        return pd.DataFrame()

    frame = review_history.copy()
    frame["prediction_as_of_date"] = pd.to_datetime(frame["prediction_as_of_date"], errors="coerce").dt.date.astype(str)
    grouped = frame.groupby("prediction_as_of_date").agg(
        Stocks_Reviewed=("symbol", "count"),
        Direction_Accuracy_Pct=("correct_prediction", lambda values: values.astype(bool).mean() * 100),
        Average_High_Miss_Pct=("high_absolute_error_pct", "mean"),
        Average_Low_Miss_Pct=("low_absolute_error_pct", "mean"),
        Overall_Accuracy_Pct=("plain_overall_accuracy_pct", "mean"),
    ).reset_index()
    grouped = grouped.rename(columns={"prediction_as_of_date": "Prediction Date"})
    grouped["Trend vs Previous Date"] = grouped["Overall_Accuracy_Pct"].diff().apply(
        lambda value: "First date" if pd.isna(value) else ("Improved" if value > 0 else ("Worse" if value < 0 else "No change"))
    )
    grouped["Change vs Previous Date"] = grouped["Overall_Accuracy_Pct"].diff()
    return grouped.sort_values("Prediction Date", ascending=False)


def build_stock_accuracy_trend(review_history: pd.DataFrame) -> pd.DataFrame:
    if review_history.empty:
        return pd.DataFrame()

    frame = review_history.copy()
    frame["prediction_as_of_date"] = pd.to_datetime(frame["prediction_as_of_date"], errors="coerce").dt.date.astype(str)
    return frame[[
        "symbol",
        "companyName",
        "prediction_as_of_date",
        "plain_overall_accuracy_pct",
        "plain_result",
    ]].assign(Stock=lambda value: value.apply(_stock_label, axis=1)).drop(columns=["symbol", "companyName"]).rename(
        columns={
            "prediction_as_of_date": "Prediction Date",
            "plain_overall_accuracy_pct": "Overall Accuracy %",
            "plain_result": "Overall Result",
        }
    ).sort_values(["Stock", "Prediction Date"], ascending=[True, False])


def build_stock_accuracy_pivot(review_history: pd.DataFrame, value_column: str = "plain_overall_accuracy_pct") -> pd.DataFrame:
    if review_history.empty:
        return pd.DataFrame()

    frame = review_history.copy()
    frame["prediction_as_of_date"] = pd.to_datetime(frame["prediction_as_of_date"], errors="coerce").dt.date.astype(str)
    frame["stock_label"] = frame.apply(_stock_label, axis=1)
    pivot = frame.pivot_table(
        index="stock_label",
        columns="prediction_as_of_date",
        values=value_column,
        aggfunc="first",
    )
    ordered_dates = sorted(pivot.columns.tolist(), reverse=True)
    pivot = pivot[ordered_dates].reset_index().rename(columns={"stock_label": "Stock"})
    return pivot


def _score_label(score, result) -> str:
    if pd.isna(score):
        return ""
    if float(score) <= 0:
        return "Failed"
    result_text = str(result)
    if result_text.startswith("Strong"):
        label = "Strong"
    elif result_text.startswith("Good"):
        label = "Good"
    elif result_text.startswith("Mixed"):
        label = "Mixed"
    else:
        label = "Passed"
    return f"{float(score):.1f}% {label}"


def build_stock_accuracy_verdict_pivot(review_history: pd.DataFrame) -> pd.DataFrame:
    if review_history.empty:
        return pd.DataFrame()

    frame = review_history.copy()
    frame["prediction_as_of_date"] = pd.to_datetime(frame["prediction_as_of_date"], errors="coerce").dt.date.astype(str)
    frame["display_score"] = frame.apply(
        lambda row: _score_label(row["plain_overall_accuracy_pct"], row["plain_result"]),
        axis=1,
    )
    frame["stock_label"] = frame.apply(_stock_label, axis=1)
    pivot = frame.pivot_table(
        index="stock_label",
        columns="prediction_as_of_date",
        values="display_score",
        aggfunc="first",
    )
    ordered_dates = sorted(pivot.columns.tolist(), reverse=True)
    pivot = pivot[ordered_dates].reset_index().rename(columns={"stock_label": "Stock"})
    pivot = pivot.rename(columns={date: f"{date} score" for date in ordered_dates})
    return pivot


def _direction_summary(row: pd.Series) -> str:
    predicted = row.get("predicted_direction", "")
    actual = row.get("actual_direction", "")
    if bool(row.get("correct_prediction", False)):
        return f"Correct: predicted {predicted}; actual {actual}"
    return f"Wrong: predicted {predicted}; actual {actual}"


def _price_summary(predicted, actual) -> str:
    if pd.isna(predicted) or pd.isna(actual):
        return ""
    return f"Predicted {float(predicted):.2f} (Actual {float(actual):.2f})"


def _high_low_quality(row: pd.Series) -> str:
    high_miss = row.get("high_absolute_error_pct")
    low_miss = row.get("low_absolute_error_pct")
    if pd.isna(high_miss) or pd.isna(low_miss):
        return ""
    worst_miss = max(float(high_miss), float(low_miss))
    if worst_miss <= 5:
        label = "Accurate"
    elif worst_miss <= 15:
        label = "Close"
    else:
        label = "Incorrect"
    return f"{label}: high miss {float(high_miss):.1f}%, low miss {float(low_miss):.1f}%"


def build_stock_daily_audit_wide(review_history: pd.DataFrame, max_dates: Optional[int] = None) -> pd.DataFrame:
    if review_history.empty:
        return pd.DataFrame()

    frame = review_history.copy()
    frame["prediction_as_of_date"] = pd.to_datetime(frame["prediction_as_of_date"], errors="coerce").dt.date.astype(str)
    frame["direction_cell"] = frame.apply(_direction_summary, axis=1)
    frame["high_cell"] = frame.apply(
        lambda row: _price_summary(row.get("predicted_next_day_high"), row.get("actual_next_day_high")),
        axis=1,
    )
    frame["low_cell"] = frame.apply(
        lambda row: _price_summary(row.get("predicted_next_day_low"), row.get("actual_next_day_low")),
        axis=1,
    )
    frame["quality_cell"] = frame.apply(_high_low_quality, axis=1)

    ordered_dates = sorted(frame["prediction_as_of_date"].dropna().unique().tolist(), reverse=True)
    if max_dates is not None:
        ordered_dates = ordered_dates[:max_dates]

    frame["stock_label"] = frame.apply(_stock_label, axis=1)
    stocks = sorted(frame["stock_label"].dropna().unique().tolist())
    rows = []
    for stock in stocks:
        stock_rows = frame[frame["stock_label"] == stock].set_index("prediction_as_of_date")
        output_row = {"Stock": stock}
        for date in ordered_dates:
            if date not in stock_rows.index:
                output_row[f"{date} Direction"] = ""
                output_row[f"{date} High"] = ""
                output_row[f"{date} Low"] = ""
                output_row[f"{date} High/Low Quality"] = ""
                continue
            row = stock_rows.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            output_row[f"{date} Direction"] = row["direction_cell"]
            output_row[f"{date} High"] = row["high_cell"]
            output_row[f"{date} Low"] = row["low_cell"]
            output_row[f"{date} High/Low Quality"] = row["quality_cell"]
        rows.append(output_row)

    return pd.DataFrame(rows)
