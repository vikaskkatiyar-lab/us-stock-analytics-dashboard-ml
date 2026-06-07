from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analytics import (  # noqa: E402
    sector_forward_outlook,
    sector_recent_performance,
    single_stock_trader_snapshot,
    stocks_with_average_daily_movement_above,
    stocks_with_daily_move_between,
    stocks_with_highest_5_day_return,
    stocks_with_predicted_next_day_range,
    stocks_with_positive_closes,
    stocks_within_percent_of_20_day_high,
    top_intraday_play_candidates,
)
from ml_audit import build_stock_daily_audit_wide  # noqa: E402

PRICE_HISTORY_FILE = BASE_DIR / "data" / "price_history.csv"
DAILY_REPORT_FILE = BASE_DIR / "reports" / "daily_stock_monitor.csv"
NEXT_DAY_PREDICTIONS_FILE = BASE_DIR / "data" / "next_day_predictions.csv"
NEXT_DAY_RESULTS_FILE = BASE_DIR / "data" / "next_day_prediction_results.csv"
TOP_INTRADAY_PICKS_FILE = BASE_DIR / "data" / "top_10_intraday_picks.csv"
TOP_INTRADAY_PICK_RESULTS_FILE = BASE_DIR / "data" / "top_10_intraday_pick_results.csv"
SECTOR_PREDICTIONS_FILE = BASE_DIR / "data" / "sector_predictions.csv"
SECTOR_RESULTS_FILE = BASE_DIR / "data" / "sector_prediction_results.csv"
LEARNING_SUMMARY_FILE = BASE_DIR / "data" / "historical_learning_summary.csv"
LEARNING_BACKTEST_FILE = BASE_DIR / "data" / "historical_learning_backtest.csv"
LEARNING_SIGNALS_FILE = BASE_DIR / "data" / "historical_learning_signals.csv"
LEARNING_WEIGHTS_FILE = BASE_DIR / "data" / "historical_learning_weights.csv"
LEARNING_TOP25_FILE = BASE_DIR / "data" / "historical_learning_top25_recent_accuracy.csv"
TRADING212_CFD_SYMBOLS_FILE = BASE_DIR / "data" / "trading212_cfd_symbols.csv"
ML_PREDICTIONS_FILE = BASE_DIR / "data" / "ml_predictions.csv"
ML_BACKTEST_FILE = BASE_DIR / "data" / "ml_backtest.csv"
ML_REVIEW_HISTORY_FILE = BASE_DIR / "data" / "ml_prediction_review_history.csv"
ML_ACCURACY_BY_DATE_FILE = BASE_DIR / "data" / "ml_accuracy_by_date.csv"
ML_SUMMARY_FILE = BASE_DIR / "reports" / "ml_summary.csv"
ML_HTML_REPORT_FILE = BASE_DIR / "reports" / "ml_dashboard.html"

ANALYSIS_EXPLANATIONS = {
    "One year performance": (
        "Shows the full daily monitor output, ranked by one year return when that "
        "ranking is available. Use this as the broad overview of all stocks in the latest run."
    ),
    "Daily dollar move range": (
        "Finds stocks whose absolute close-to-close daily price move fell between "
        "your minimum and maximum dollar move during the selected lookback period."
    ),
    "Positive closes": (
        "Finds stocks that closed higher than the previous trading day for every "
        "day in the selected lookback period."
    ),
    "Average daily movement": (
        "Finds stocks whose average absolute daily close-to-close dollar move is "
        "above your minimum dollar move threshold."
    ),
    "Near 20 day high": (
        "Finds stocks currently trading within your selected percentage of their "
        "highest adjusted close over the last 20 trading days."
    ),
    "Highest 5 day return": (
        "Ranks stocks by their latest five trading day return, highest first."
    ),
    "Predicted next day range": (
        "Shows the top 25 stocks with the widest estimated next day low-to-high "
        "range. This uses recent average close-to-close movement, so treat it as "
        "a simple volatility estimate rather than a guaranteed forecast."
    ),
    "Top 10 next day intraday picks": (
        "Ranks the ten strongest next-day intraday candidates using estimated "
        "range, recent movement, short-term momentum, proximity to a 20 day high, "
        "and a simple trend check. This is a planning screen, not a live trade signal."
    ),
    "Single stock trader view": (
        "Lets you select one ticker and see a compact short-term dashboard with "
        "trend, momentum, recent movement, estimated next-day range, and nearby "
        "support/resistance-style levels from the daily close history."
    ),
    "Saved next day predictions": (
        "Shows the predictions saved by the automated daily cycle. Each row is a "
        "prediction made from one stock's latest available close."
    ),
    "Prediction vs actual results": (
        "Compares saved predictions with the next available actual trading day, "
        "including whether actual lows and highs landed inside the estimated range."
    ),
    "Saved top 10 intraday picks": (
        "Shows the Top 10 next-day intraday picks saved by the automated cycle."
    ),
    "Top 10 picks vs actual results": (
        "Reviews how the saved Top 10 intraday picks performed on the next available "
        "trading day, including actual range, close return, and a Good/Okay/Weak label."
    ),
    "Sector Outlook": (
        "Shows recent sector gainers and losers, forward-looking sector estimates, "
        "and a growing comparison of sector predictions versus actual returns."
    ),
    "Historical Learning Backtest": (
        "Learns 50 close-to-close signals from historical data, tests them on a "
        "50 day walk-forward window, and repeats weight updates up to 15 times."
    ),
    "Ask a stock question": (
        "Ask a plain-English question about one stock. The answer uses the latest "
        "local analytics, and can fetch recent online headlines when you ask about news."
    ),
}

COLUMN_GLOSSARY = {
    "rank_1y_performance": "Position in the list when stocks are ranked by one year return.",
    "symbol": "Ticker symbol.",
    "companyName": "Company name.",
    "exchange": "Exchange or market label from the monitor.",
    "sector": "Broad business sector.",
    "industry": "More specific industry group.",
    "marketCap": "Company market value.",
    "as_of_date": "Latest price date used by the daily monitor.",
    "latest_date": "Latest price date used in this analysis.",
    "latest_price": "Most recent adjusted closing price.",
    "adj_close": "Adjusted closing price after accounting for splits and dividends.",
    "previous_close": "Adjusted closing price from the prior trading day.",
    "return_1d_pct": "One trading day percentage return.",
    "return_5d_pct": "Five trading day percentage return.",
    "return_1m_pct": "Approximate one month percentage return.",
    "return_3m_pct": "Approximate three month percentage return.",
    "return_6m_pct": "Approximate six month percentage return.",
    "return_1y_pct": "Approximate one year percentage return.",
    "ytd_return_pct": "Year to date percentage return.",
    "one_year_high": "Highest adjusted close over the past year of available data.",
    "drawdown_from_1y_high_pct": "How far the latest price is below the one year high.",
    "observations": "Number of price rows used by the monitor.",
    "error": "Any error captured for that stock during the monitor run.",
    "matching_days": "Number of days that matched the selected rule.",
    "latest_match_date": "Most recent date that matched the selected rule.",
    "min_abs_daily_move": "Smallest absolute daily dollar move among matching days.",
    "max_abs_daily_move": "Largest absolute daily dollar move among matching days.",
    "avg_abs_daily_move": "Average absolute daily dollar move.",
    "daily_move": "Dollar change from the previous adjusted close.",
    "abs_daily_move": "Daily dollar move without the plus or minus sign.",
    "daily_return_pct": "Daily percentage return from the previous adjusted close.",
    "positive_days": "Number of days that closed higher than the prior day.",
    "observed_days": "Number of trading days included in the calculation.",
    "total_move": "Total dollar move across the selected period.",
    "total_return_pct": "Total compounded percentage return across the selected period.",
    "high_20d": "Highest adjusted close over the latest 20 trading days.",
    "pct_from_20d_high": "How far the latest price is from the 20 day high.",
    "close_5d_ago": "Adjusted closing price five trading days before the latest price.",
    "predicted_next_day_low": "Estimated next day low based on recent average daily movement.",
    "predicted_next_day_high": "Estimated next day high based on recent average daily movement.",
    "predicted_range": "Estimated dollar gap between the predicted low and high.",
    "predicted_range_pct": "Estimated next day range as a percentage of the latest price.",
    "return_10d_pct": "Ten trading day percentage return.",
    "return_20d_pct": "Twenty trading day percentage return.",
    "ma_5": "Average adjusted close over the last 5 trading days.",
    "ma_10": "Average adjusted close over the last 10 trading days.",
    "ma_20": "Average adjusted close over the last 20 trading days.",
    "ma_50": "Average adjusted close over the last 50 trading days.",
    "price_vs_ma20_pct": "How far the latest price is above or below the 20 day average.",
    "rsi_14": "A simple 14 day momentum reading; high can mean stretched, low can mean weak.",
    "trend_bias": "Quick label based on price and short moving average versus the 20 day average.",
    "current_streak": "Current run of up days or down days.",
    "up_days_last_5": "How many of the last 5 trading days closed higher.",
    "up_days_last_10": "How many of the last 10 trading days closed higher.",
    "avg_abs_move_5d": "Average absolute daily dollar move over the last 5 trading days.",
    "avg_abs_move_10d": "Average absolute daily dollar move over the last 10 trading days.",
    "avg_abs_move_20d": "Average absolute daily dollar move over the last 20 trading days.",
    "volatility_20d_pct": "Standard deviation of daily percentage returns over the last 20 trading days.",
    "predicted_low_5d_avg_move": "Latest price minus the recent 5 day average daily move.",
    "predicted_high_5d_avg_move": "Latest price plus the recent 5 day average daily move.",
    "predicted_range_5d_avg_move": "Estimated low-to-high range using the 5 day average daily move.",
    "predicted_low_20d_avg_move": "Latest price minus the recent 20 day average daily move.",
    "predicted_high_20d_avg_move": "Latest price plus the recent 20 day average daily move.",
    "predicted_range_20d_avg_move": "Estimated low-to-high range using the 20 day average daily move.",
    "low_20d": "Lowest adjusted close over the latest 20 trading days.",
    "high_50d": "Highest adjusted close over the latest 50 trading days.",
    "low_50d": "Lowest adjusted close over the latest 50 trading days.",
    "pct_above_20d_low": "How far the latest price is above the 20 day low.",
    "pct_from_50d_high": "How far the latest price is from the 50 day high.",
    "pct_above_50d_low": "How far the latest price is above the 50 day low.",
    "prediction_as_of_date": "Date of the latest close used to make the prediction.",
    "prediction_model": "Simple method used to create the prediction.",
    "created_at": "When the prediction row was saved.",
    "predicted_low_5d": "Predicted next trading day low using the recent 5 day average move.",
    "predicted_high_5d": "Predicted next trading day high using the recent 5 day average move.",
    "predicted_low_20d": "Predicted next trading day low using the recent 20 day average move.",
    "predicted_high_20d": "Predicted next trading day high using the recent 20 day average move.",
    "predicted_range_5d": "Predicted low-to-high range using the 5 day average move.",
    "predicted_range_20d": "Predicted low-to-high range using the 20 day average move.",
    "actual_date": "Next available trading date used for comparison.",
    "actual_low": "Actual adjusted daily low for the comparison date.",
    "actual_high": "Actual adjusted daily high for the comparison date.",
    "actual_close": "Actual adjusted close for the comparison date.",
    "actual_source": "Whether the comparison used high/low data or a close-only fallback.",
    "actual_range_inside_prediction_5d": "Whether actual low and high both fit inside the 5 day estimate.",
    "actual_range_inside_prediction_20d": "Whether actual low and high both fit inside the 20 day estimate.",
    "intraday_pick_score": "Combined ranking score for next-day intraday candidates.",
    "reasoning": "Plain-English explanation of why the stock was selected.",
    "up_days": "Number of up days in the selected lookback period.",
    "pick_as_of_date": "Date of the latest close used to choose the Top 10 pick.",
    "pick_rank": "Position in the saved Top 10 intraday picks list.",
    "pick_model": "Simple method used to choose the Top 10 intraday picks.",
    "actual_intraday_range": "Actual high minus actual low on the comparison date.",
    "actual_intraday_range_pct": "Actual intraday range as a percentage of the pick date price.",
    "actual_close_return_pct": "Close-to-close return from pick date to comparison date.",
    "range_capture_pct": "Actual range as a percentage of the predicted range.",
    "hit_predicted_low": "Whether the actual low reached or went below the predicted low.",
    "hit_predicted_high": "Whether the actual high reached or went above the predicted high.",
    "closed_up": "Whether the stock closed above the pick date price.",
    "pick_result": "Simple Good/Okay/Weak review label for the pick.",
    "sector_prediction_score": "Combined sector score based on 1, 5, and 20 day estimates.",
    "sector_prediction_rank": "Position in the saved sector prediction list.",
    "predicted_return_1d_pct": "Estimated sector return for the next trading day.",
    "predicted_return_5d_pct": "Estimated sector return for the next 5 trading days.",
    "predicted_return_20d_pct": "Estimated sector return for the next 20 trading days.",
    "actual_return_1d_pct": "Actual sector return over the next trading day.",
    "actual_return_5d_pct": "Actual sector return over the next 5 trading days.",
    "actual_return_20d_pct": "Actual sector return over the next 20 trading days.",
    "prediction_error_1d_pct": "Actual 1 day sector return minus predicted 1 day return.",
    "prediction_error_5d_pct": "Actual 5 day sector return minus predicted 5 day return.",
    "prediction_error_20d_pct": "Actual 20 day sector return minus predicted 20 day return.",
    "prediction_score": "Model score; positive predicts up, negative predicts down.",
    "predicted_up": "Whether the model predicted the stock would close up the next day.",
    "target_up": "Whether the stock actually closed up the next day.",
    "correct": "Whether the model prediction matched the actual direction.",
    "accuracy": "Share of predictions that were directionally correct.",
    "weight": "Learned importance weight for a historical signal.",
    "range_prediction_accuracy_pct": "Average of low accuracy and high accuracy, based on predicted price versus actual price.",
    "low_prediction_accuracy_pct": "Accuracy of predicted low versus actual low.",
    "high_prediction_accuracy_pct": "Accuracy of predicted high versus actual high.",
    "latest_prediction_date": "Date the latest historical prediction was made from.",
    "latest_actual_date": "Next trading date used to check that latest historical prediction.",
    "latest_prediction": "Most recent historical prediction direction for this stock.",
    "latest_actual": "Actual direction for that same latest historical prediction.",
    "direction_result": "Whether the Up/Down prediction was correct, with the predicted direction named.",
    "direction_accuracy_pct": "100 if the Up/Down prediction was correct, 0 if it was wrong.",
    "prediction_close": "Adjusted close on the prediction date.",
    "predicted_next_day_low": "Predicted low for the next trading day.",
    "predicted_next_day_high": "Predicted high for the next trading day.",
    "actual_low": "Actual adjusted low on the next trading day.",
    "actual_high": "Actual adjusted high on the next trading day.",
    "actual_close": "Actual adjusted close on the next trading day.",
    "predicted_range": "Predicted high minus predicted low.",
    "actual_range": "Actual high minus actual low.",
    "probability_up": "ML-estimated probability that the next trading day close is higher than the latest close.",
    "prediction_as_of_date": "Exact market date used to make the prediction.",
    "actual_review_date": "Exact next trading date used to compare the historical prediction with actual prices.",
    "expected_next_trading_date": "For the latest unevaluated prediction, this is the next trading session after the prediction date.",
    "predicted_direction": "ML direction estimate: Up or Down or Flat.",
    "actual_direction": "Actual next trading day direction used to review an ML prediction.",
    "prediction_review": "Plain-English review of whether the ML Up/Down prediction was correct.",
    "confidence_level": "Plain-English confidence bucket based on distance from 50% probability.",
    "predicted_high_move_pct": "ML-estimated next-day upside move from the latest close.",
    "predicted_low_move_pct": "ML-estimated next-day downside move from the latest close.",
    "predicted_next_day_range": "Predicted high minus predicted low for the next trading day.",
    "actual_next_day_high": "Actual adjusted high on the next trading day used to review the ML high prediction.",
    "actual_next_day_low": "Actual adjusted low on the next trading day used to review the ML low prediction.",
    "high_prediction_error_pct": "Signed high prediction error percent. Positive means predicted high was above actual; negative means below actual.",
    "low_prediction_error_pct": "Signed low prediction error percent. Positive means predicted low was above actual; negative means below actual.",
    "high_prediction_accuracy_pct": "High price prediction score. 100 is exact; lower is worse; 0 means a very large miss.",
    "low_prediction_accuracy_pct": "Low price prediction score. 100 is exact; lower is worse; 0 means a very large miss.",
    "range_prediction_accuracy_pct": "Average of high and low prediction accuracy percentages.",
    "high_within_2pct": "Whether predicted high was within 2% of the actual next-day high.",
    "low_within_2pct": "Whether predicted low was within 2% of the actual next-day low.",
    "high_low_within_2pct": "Whether both predicted high and predicted low were within 2% of actual. This is a loose tolerance check, not a perfect prediction.",
    "high_success_rate": "Share of predictions where next-day high was within 2% of actual high.",
    "low_success_rate": "Share of predictions where next-day low was within 2% of actual low.",
    "high_low_within_2pct_rate": "Share of predictions where both high and low were within 2% of actual.",
    "stock_level_accuracy": "Backtested direction accuracy for this stock in the ML test window.",
    "average_prediction_error": "Average absolute direction error for this stock in the ML test window.",
    "key_positive_features": "Feature values that most strongly lean positive for the latest row.",
    "key_negative_features": "Feature values that most strongly lean negative for the latest row.",
    "correct_prediction": "Whether the ML predicted direction matched the actual next-day direction.",
    "direction_accuracy": "Share of ML predictions that matched actual direction for a date.",
    "average_confidence": "Average distance from a neutral 50% probability for a date.",
}


@st.cache_data
def load_csv(path: Path, modified_at: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_optional_csv(path: Path, modified_at) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def file_mtime(path: Path):
    return path.stat().st_mtime if path.exists() else None


def apply_profile_filters(
    results: pd.DataFrame,
    report: pd.DataFrame,
    min_stock_price: float,
    selected_sectors: list[str],
) -> pd.DataFrame:
    if results.empty:
        return results

    profile_cols = [
        "symbol", "companyName", "sector", "industry", "marketCap",
        "rank_1y_performance", "return_1d_pct", "return_5d_pct", "return_1y_pct",
    ]
    profile_cols = [col for col in profile_cols if col in report.columns]
    profile_cols = ["symbol"] + [col for col in profile_cols if col != "symbol" and col not in results.columns]
    merged = results.merge(report[profile_cols], on="symbol", how="left")

    if "latest_price" in merged.columns:
        merged = merged[pd.to_numeric(merged["latest_price"], errors="coerce") >= min_stock_price]

    if selected_sectors and "sector" in merged.columns:
        merged = merged[merged["sector"].isin(selected_sectors)]

    front_cols = [col for col in ["symbol", "companyName", "sector", "industry", "latest_price"] if col in merged.columns]
    other_cols = [col for col in merged.columns if col not in front_cols]
    return merged[front_cols + other_cols]


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in display.columns:
        if "date" in col:
            display[col] = pd.to_datetime(display[col], errors="coerce").dt.date
        elif pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].round(2)
    return display


def style_learning_table(df: pd.DataFrame):
    def cell_style(value):
        if isinstance(value, str):
            if value.startswith("Correct"):
                return "background-color: #d9f7e5; color: #14532d; font-weight: 600"
            if value.startswith("Wrong"):
                return "background-color: #fde2e2; color: #7f1d1d; font-weight: 600"
            if value == "Up":
                return "background-color: #e0f2fe; color: #075985"
            if value == "Down":
                return "background-color: #fff7ed; color: #9a3412"
        if isinstance(value, (int, float)) and not pd.isna(value):
            if value >= 95:
                return "background-color: #dcfce7; color: #166534"
            if value >= 80:
                return "background-color: #fef9c3; color: #854d0e"
            if value >= 50:
                return "background-color: #ffedd5; color: #9a3412"
            if value >= 0:
                return "background-color: #fee2e2; color: #991b1b"
        return ""

    def row_style(row):
        if row.get("symbol") == "AVERAGE":
            return ["font-weight: 700; background-color: #eef2ff; color: #312e81"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(row_style, axis=1)
    style_cols = [
        col for col in df.columns
        if "accuracy_pct" in col or "direction_result" in col or "direction" in col
    ]
    return styled.map(cell_style, subset=style_cols)


def format_value(value, suffix=""):
    if pd.isna(value):
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:,.2f}{suffix}"


def latest_date_label(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "not available"
    dates = pd.to_datetime(df[column], errors="coerce").dropna()
    if dates.empty:
        return "not available"
    return dates.max().date().isoformat()


def date_list_label(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "none"
    dates = pd.to_datetime(df[column], errors="coerce").dropna()
    if dates.empty:
        return "none"
    values = sorted(dates.dt.date.astype(str).unique().tolist())
    return ", ".join(values[-8:])


def render_prediction_date_status(
    price_history: pd.DataFrame,
    saved_predictions: pd.DataFrame,
    prediction_results: pd.DataFrame,
):
    latest_price_date = latest_date_label(price_history, "date")
    latest_prediction_date = latest_date_label(saved_predictions, "prediction_as_of_date")
    latest_actual_date = latest_date_label(prediction_results, "actual_date")

    cols = st.columns(4)
    cols[0].metric("Latest price date", latest_price_date)
    cols[1].metric("Latest prediction date", latest_prediction_date)
    cols[2].metric("Latest evaluated actual date", latest_actual_date)
    cols[3].metric("Evaluation rows", f"{len(prediction_results):,}")

    st.caption(
        "Prediction dates available: "
        f"{date_list_label(saved_predictions, 'prediction_as_of_date')}. "
        "Actual evaluation dates available: "
        f"{date_list_label(prediction_results, 'actual_date')}."
    )

    if latest_prediction_date != "not available" and latest_actual_date == "not available":
        if latest_price_date == "not available":
            st.warning("No price history is available yet, so predictions cannot be evaluated.")
        elif latest_price_date <= latest_prediction_date:
            st.warning(
                "No prediction evaluation is available yet because the latest saved price date "
                f"({latest_price_date}) is not later than the latest prediction date "
                f"({latest_prediction_date}). The next completed market day must be downloaded first."
            )
        else:
            st.warning(
                "There is newer price history than the latest prediction date, but no comparison rows were saved. "
                "Run the prediction cycle again to build the evaluation table."
            )


def render_prediction_result_summary(prediction_results: pd.DataFrame):
    if prediction_results.empty:
        return

    df = prediction_results.copy()
    df["prediction_as_of_date"] = pd.to_datetime(df["prediction_as_of_date"], errors="coerce").dt.date
    df["actual_date"] = pd.to_datetime(df["actual_date"], errors="coerce").dt.date
    for col in [
        "actual_range_inside_prediction_5d",
        "actual_range_inside_prediction_20d",
        "actual_low_inside_prediction_5d",
        "actual_high_inside_prediction_5d",
    ]:
        if col in df.columns:
            df[col] = df[col].map(lambda value: str(value).lower() == "true" if pd.notna(value) else pd.NA)

    summary = df.groupby(["prediction_as_of_date", "actual_date"], dropna=False).agg(
        evaluated_stocks=("symbol", "count"),
        range_hit_5d_pct=("actual_range_inside_prediction_5d", "mean"),
        range_hit_20d_pct=("actual_range_inside_prediction_20d", "mean"),
        low_inside_5d_pct=("actual_low_inside_prediction_5d", "mean"),
        high_inside_5d_pct=("actual_high_inside_prediction_5d", "mean"),
    ).reset_index()

    for col in [col for col in summary.columns if col.endswith("_pct")]:
        summary[col] = summary[col] * 100

    st.markdown("**Daily Prediction Evaluation Summary**")
    st.dataframe(format_for_display(summary), width="stretch", hide_index=True, height=260)


def render_top_pick_date_status(
    price_history: pd.DataFrame,
    saved_top_picks: pd.DataFrame,
    top_pick_results: pd.DataFrame,
):
    latest_price_date = latest_date_label(price_history, "date")
    latest_pick_date = latest_date_label(saved_top_picks, "pick_as_of_date")
    latest_actual_date = latest_date_label(top_pick_results, "actual_date")

    cols = st.columns(4)
    cols[0].metric("Latest price date", latest_price_date)
    cols[1].metric("Latest Top 10 pick date", latest_pick_date)
    cols[2].metric("Latest reviewed actual date", latest_actual_date)
    cols[3].metric("Review rows", f"{len(top_pick_results):,}")

    st.caption(
        "Top 10 pick dates available: "
        f"{date_list_label(saved_top_picks, 'pick_as_of_date')}. "
        "Actual review dates available: "
        f"{date_list_label(top_pick_results, 'actual_date')}."
    )

    if latest_pick_date != "not available" and latest_actual_date == "not available":
        if latest_price_date <= latest_pick_date:
            st.warning(
                "No Top 10 pick review is available yet because the latest saved price date "
                f"({latest_price_date}) is not later than the latest pick date ({latest_pick_date})."
            )
        else:
            st.warning(
                "There is newer price history than the latest Top 10 pick date, but no review rows were saved. "
                "Run the prediction cycle again to build the review table."
            )


def render_top_pick_result_summary(top_pick_results: pd.DataFrame):
    if top_pick_results.empty:
        return

    df = top_pick_results.copy()
    df["pick_as_of_date"] = pd.to_datetime(df["pick_as_of_date"], errors="coerce").dt.date
    df["actual_date"] = pd.to_datetime(df["actual_date"], errors="coerce").dt.date
    df["closed_up"] = df["closed_up"].map(lambda value: str(value).lower() == "true" if pd.notna(value) else pd.NA)

    summary = df.groupby(["pick_as_of_date", "actual_date"], dropna=False).agg(
        reviewed_picks=("symbol", "count"),
        good_picks=("pick_result", lambda values: (values == "Good").sum()),
        okay_picks=("pick_result", lambda values: (values == "Okay").sum()),
        weak_picks=("pick_result", lambda values: (values == "Weak").sum()),
        closed_up_pct=("closed_up", "mean"),
        avg_actual_intraday_range_pct=("actual_intraday_range_pct", "mean"),
        avg_close_return_pct=("actual_close_return_pct", "mean"),
    ).reset_index()
    summary["closed_up_pct"] = summary["closed_up_pct"] * 100

    st.markdown("**Daily Top 10 Pick Review Summary**")
    st.dataframe(format_for_display(summary), width="stretch", hide_index=True, height=260)


def stock_label(symbol: str, report: pd.DataFrame) -> str:
    match = report[report["symbol"] == symbol]
    if match.empty or "companyName" not in report.columns:
        return symbol
    company = match.iloc[0]["companyName"]
    return f"{symbol} - {company}" if pd.notna(company) else symbol


@st.cache_data(ttl=900)
def fetch_stock_news(symbol: str) -> list[dict]:
    try:
        news = yf.Ticker(symbol).news or []
    except Exception:
        return []

    items = []
    for item in news[:8]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title", "")
        link = content.get("canonicalUrl", {}).get("url") or item.get("link", "")
        publisher = content.get("provider", {}).get("displayName") or item.get("publisher", "")
        published = content.get("pubDate") or item.get("providerPublishTime", "")
        if title:
            items.append({
                "title": title,
                "publisher": publisher,
                "published": published,
                "link": link,
            })
    return items


def detect_symbol_from_question(question: str, symbols: list[str], report: pd.DataFrame):
    text = question.upper()
    for symbol in sorted(symbols, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(symbol.upper())}(?![A-Z0-9])", text):
            return symbol

    question_lower = question.lower()
    if "companyName" in report.columns:
        for _, row in report.dropna(subset=["companyName"]).iterrows():
            company = str(row["companyName"]).lower()
            if company and company in question_lower:
                return row["symbol"]

    return None


def recent_market_movers(price_history: pd.DataFrame, daily_report: pd.DataFrame, days: int = 1) -> pd.DataFrame:
    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "adj_close"]).sort_values(["symbol", "date"])
    grouped = df.groupby("symbol", group_keys=False)
    df[f"return_{days}d_pct"] = (df["adj_close"] / grouped["adj_close"].shift(days) - 1) * 100
    latest = df.groupby("symbol").tail(1).dropna(subset=[f"return_{days}d_pct"]).copy()
    latest["abs_return_pct"] = latest[f"return_{days}d_pct"].abs()
    cols = [col for col in ["symbol", "companyName", "sector", "industry", "latest_price"] if col in daily_report.columns]
    profile = daily_report[[col for col in ["symbol", "companyName", "sector", "industry", "latest_price"] if col in daily_report.columns]]
    latest = latest.merge(profile, on="symbol", how="left", suffixes=("", "_report"))
    latest = latest.rename(columns={"date": "latest_date", "adj_close": "latest_close"})
    return latest[[
        "symbol", "companyName", "sector", "industry", "latest_date", "latest_close",
        f"return_{days}d_pct", "abs_return_pct"
    ]].sort_values("abs_return_pct", ascending=False)


def render_market_query_answer(question: str, price_history: pd.DataFrame, daily_report: pd.DataFrame, learning_top25: pd.DataFrame, saved_top_picks: pd.DataFrame, sector_prediction_results: pd.DataFrame):
    text = question.lower()
    days = 1
    label = "1 day"
    if any(phrase in text for phrase in ["5 day", "5-day", "5 days", "five day", "five days", "last five"]):
        days = 5
        label = "5 day"
    elif any(phrase in text for phrase in ["20 day", "20-day", "20 days", "twenty day", "twenty days", "last twenty"]):
        days = 20
        label = "20 day"

    if any(word in text for word in ["fluctuation", "fluctuate", "mover", "movement", "volatile", "biggest move"]):
        movers = recent_market_movers(price_history, daily_report, days=days).head(25)
        st.markdown(f"**Top stock fluctuations over the latest {label} period**")
        st.caption("Ranked by absolute close-to-close percentage move from the saved daily history.")
        st.dataframe(format_for_display(movers), width="stretch", hide_index=True, height=620)
        st.download_button(
            "Download market movers answer",
            data=format_for_display(movers).to_csv(index=False).encode("utf-8"),
            file_name=f"top_stock_fluctuations_{days}d.csv",
            mime="text/csv",
        )
        return

    if "gainer" in text or "gain" in text or "up" in text:
        gainers = recent_market_movers(price_history, daily_report, days=days).sort_values(f"return_{days}d_pct", ascending=False).head(25)
        st.markdown(f"**Top gainers over the latest {label} period**")
        st.dataframe(format_for_display(gainers), width="stretch", hide_index=True, height=620)
        return

    if "loser" in text or "lost" in text or "down" in text:
        losers = recent_market_movers(price_history, daily_report, days=days).sort_values(f"return_{days}d_pct", ascending=True).head(25)
        st.markdown(f"**Top losers over the latest {label} period**")
        st.dataframe(format_for_display(losers), width="stretch", hide_index=True, height=620)
        return

    if "top 25" in text and not learning_top25.empty:
        st.markdown("**Top 25 learned-model stocks**")
        st.dataframe(format_for_display(learning_top25), width="stretch", hide_index=True, height=620)
        return

    if "top 10" in text and not saved_top_picks.empty:
        st.markdown("**Latest saved Top 10 intraday picks**")
        st.dataframe(format_for_display(saved_top_picks.tail(10)), width="stretch", hide_index=True, height=520)
        return

    if "sector" in text:
        recent = sector_recent_performance(price_history, daily_report)
        outlook = sector_forward_outlook(price_history, daily_report)
        st.markdown("**Sector snapshot**")
        st.dataframe(format_for_display(outlook.head(10)), width="stretch", hide_index=True, height=420)
        st.markdown("**Recent sector performance**")
        st.dataframe(format_for_display(recent.head(10)), width="stretch", hide_index=True, height=420)
        return

    st.info(
        "I can answer market-wide questions like `top stock fluctuation yesterday`, "
        "`top gainers last 5 days`, `top losers last 20 days`, `top 25 learned stocks`, "
        "or single-stock questions like `next day range for GOOGL`."
    )


def latest_stock_context(symbol: str, price_history: pd.DataFrame, daily_report: pd.DataFrame, learning_top25: pd.DataFrame, saved_predictions: pd.DataFrame, saved_top_picks: pd.DataFrame) -> dict:
    context = {"symbol": symbol}
    profile = daily_report[daily_report["symbol"] == symbol]
    if not profile.empty:
        context.update(profile.iloc[0].to_dict())

    prices = price_history[price_history["symbol"] == symbol].copy()
    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
        prices = prices.dropna(subset=["date", "adj_close"]).sort_values("date")
        latest = prices.iloc[-1]
        context["latest_price_date"] = latest["date"].date().isoformat()
        context["latest_close"] = latest["adj_close"]
        for days in [1, 5, 20]:
            if len(prices) > days:
                previous = prices["adj_close"].iloc[-days - 1]
                context[f"return_{days}d_pct_local"] = (latest["adj_close"] / previous - 1) * 100 if previous else None

    learning = learning_top25[learning_top25["symbol"] == symbol] if not learning_top25.empty else pd.DataFrame()
    if not learning.empty:
        context.update({f"learning_{key}": value for key, value in learning.iloc[0].to_dict().items()})

    predictions = saved_predictions[saved_predictions["symbol"] == symbol] if not saved_predictions.empty else pd.DataFrame()
    if not predictions.empty and "prediction_as_of_date" in predictions.columns:
        predictions = predictions.sort_values("prediction_as_of_date")
        context.update({f"saved_prediction_{key}": value for key, value in predictions.iloc[-1].to_dict().items()})

    top_pick = saved_top_picks[saved_top_picks["symbol"] == symbol] if not saved_top_picks.empty else pd.DataFrame()
    if not top_pick.empty:
        context.update({f"top_pick_{key}": value for key, value in top_pick.iloc[-1].to_dict().items()})

    return context


def render_stock_query_answer(question: str, context: dict, include_news: bool):
    symbol = context.get("symbol", "")
    name = context.get("companyName", symbol)
    st.markdown(f"**Answer for {symbol} - {name}**")

    bullets = []
    if "latest_close" in context:
        bullets.append(f"Latest adjusted close: **{format_value(context['latest_close'])}** on {context.get('latest_price_date', 'latest date')}.")
    for label, key in [("1 day", "return_1d_pct_local"), ("5 day", "return_5d_pct_local"), ("20 day", "return_20d_pct_local")]:
        if key in context and pd.notna(context[key]):
            bullets.append(f"{label} return: **{format_value(context[key], '%')}**.")
    if "learning_prediction_accuracy_last_5d_pct" in context:
        bullets.append(
            f"Historical learning accuracy for this stock over the latest 5 backtest days: "
            f"**{format_value(context['learning_prediction_accuracy_last_5d_pct'], '%')}**."
        )
    if "learning_predicted_next_day_low" in context:
        bullets.append(
            f"Learning-table next day range: **{format_value(context['learning_predicted_next_day_low'])} "
            f"to {format_value(context['learning_predicted_next_day_high'])}**."
        )
    if "learning_predicted_next_5d_low" in context:
        bullets.append(
            f"Learning-table next 5 day range: **{format_value(context['learning_predicted_next_5d_low'])} "
            f"to {format_value(context['learning_predicted_next_5d_high'])}**."
        )
    if "saved_prediction_predicted_low_5d" in context:
        bullets.append(
            f"Saved daily prediction range: **{format_value(context['saved_prediction_predicted_low_5d'])} "
            f"to {format_value(context['saved_prediction_predicted_high_5d'])}** using the 5 day move model."
        )
    if "top_pick_reasoning" in context:
        bullets.append(f"Top-pick reasoning: {context['top_pick_reasoning']}")

    if not bullets:
        bullets.append("I found the stock in the universe, but there is not enough saved analytics yet for a detailed answer.")

    for bullet in bullets:
        st.markdown(f"- {bullet}")

    if include_news:
        st.markdown("**Recent online headlines**")
        news = fetch_stock_news(symbol)
        if not news:
            st.info("I could not fetch recent headlines for this ticker right now.")
        else:
            for item in news:
                label = item["title"]
                if item.get("publisher"):
                    label += f" - {item['publisher']}"
                if item.get("link"):
                    st.markdown(f"- [{label}]({item['link']})")
                else:
                    st.markdown(f"- {label}")


def render_stock_query_view(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
    learning_top25: pd.DataFrame,
    saved_predictions: pd.DataFrame,
    saved_top_picks: pd.DataFrame,
    sector_prediction_results: pd.DataFrame,
    selected_symbol: str,
    symbols: list[str],
):
    st.caption(
        "Ask about one stock or the whole market. Examples: "
        "`top stock fluctuation yesterday`, `top gainers last 5 days`, "
        "`next day range for GOOGL`, or `news about Nvidia`."
    )
    question = st.text_input(
        "Ask a question",
        placeholder="Example: top stock fluctuation yesterday",
    )
    st.selectbox(
        "Optional stock context",
        options=symbols,
        index=symbols.index(selected_symbol) if selected_symbol in symbols else 0,
        format_func=lambda symbol: stock_label(symbol, daily_report),
        help="Use this only as a browsing helper. Market-wide questions are no longer forced into this stock.",
    )
    if not question:
        st.info("Type a question above to get an immediate answer.")
        return

    symbol = detect_symbol_from_question(question, symbols, daily_report)
    if symbol is None:
        render_market_query_answer(
            question,
            price_history,
            daily_report,
            learning_top25,
            saved_top_picks,
            sector_prediction_results,
        )
        return

    include_news = any(word in question.lower() for word in ["news", "latest", "research", "headline", "headlines", "why"])
    context = latest_stock_context(symbol, price_history, daily_report, learning_top25, saved_predictions, saved_top_picks)
    render_stock_query_answer(question, context, include_news)


def render_single_stock_view(price_history: pd.DataFrame, daily_report: pd.DataFrame, symbol: str, lookback_days: int):
    snapshot = single_stock_trader_snapshot(price_history, symbol=symbol, lookback_days=lookback_days)
    summary = snapshot["summary"]

    if summary.empty:
        st.warning("No price history is available for the selected stock.")
        return

    row = summary.iloc[0]
    st.caption(
        "This view uses daily adjusted closes from the generated price history. "
        "It is useful for short-term planning, but it does not include live intraday candles, bid/ask, volume, or news."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Latest price", format_value(row["latest_price"]))
    metric_cols[1].metric("1 day move", format_value(row["daily_move"]), format_value(row["daily_return_pct"], "%"))
    metric_cols[2].metric("5 day return", format_value(row["return_5d_pct"], "%"))
    metric_cols[3].metric("RSI 14", format_value(row["rsi_14"]))
    metric_cols[4].metric("Trend", row["trend_bias"])

    st.markdown("**Next Day Prediction**")
    st.caption(
        "Estimated low and high for the next trading day, based on recent average daily close-to-close movement."
    )
    prediction_cols = st.columns(4)
    prediction_cols[0].metric("Predicted low, 5 day", format_value(row["predicted_low_5d_avg_move"]))
    prediction_cols[1].metric("Predicted high, 5 day", format_value(row["predicted_high_5d_avg_move"]))
    prediction_cols[2].metric("Predicted range, 5 day", format_value(row["predicted_range_5d_avg_move"]))
    prediction_cols[3].metric("Current streak", row["current_streak"])

    prediction_table = pd.DataFrame([
        {
            "basis": "5 day average move",
            "predicted_low": row["predicted_low_5d_avg_move"],
            "predicted_high": row["predicted_high_5d_avg_move"],
            "predicted_range": row["predicted_range_5d_avg_move"],
            "average_daily_move_used": row["avg_abs_move_5d"],
        },
        {
            "basis": "20 day average move",
            "predicted_low": row["predicted_low_20d_avg_move"],
            "predicted_high": row["predicted_high_20d_avg_move"],
            "predicted_range": row["predicted_range_20d_avg_move"],
            "average_daily_move_used": row["avg_abs_move_20d"],
        },
    ])
    st.dataframe(format_for_display(prediction_table), width="stretch", hide_index=True, height=145)

    chart = snapshot["chart"].copy()
    chart["date"] = pd.to_datetime(chart["date"])
    chart = chart.set_index("date")
    st.line_chart(chart, height=320, width="stretch")

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Short-Term Signals**")
        signals = summary[[
            "latest_date", "return_5d_pct", "return_10d_pct", "return_20d_pct",
            "price_vs_ma20_pct", "up_days_last_5", "up_days_last_10",
            "avg_abs_move_10d", "avg_abs_move_20d", "volatility_20d_pct",
        ]]
        st.dataframe(format_for_display(signals), width="stretch", hide_index=True, height=180)

    with right_col:
        st.markdown("**Nearby Levels**")
        st.dataframe(format_for_display(snapshot["levels"]), width="stretch", hide_index=True, height=180)

    st.markdown("**Recent Trading Days**")
    st.dataframe(format_for_display(snapshot["recent"]), width="stretch", hide_index=True, height=360)

    st.download_button(
        "Download selected stock analytics",
        data=pd.concat(
            [
                summary.assign(section="summary"),
                snapshot["levels"].assign(section="levels"),
            ],
            ignore_index=True,
            sort=False,
        ).to_csv(index=False).encode("utf-8"),
        file_name=f"{symbol.lower()}_single_stock_analytics.csv",
        mime="text/csv",
    )


def render_sector_outlook(
    price_history: pd.DataFrame,
    daily_report: pd.DataFrame,
    sector_results: pd.DataFrame,
):
    recent = sector_recent_performance(price_history, daily_report)
    outlook = sector_forward_outlook(price_history, daily_report)

    if recent.empty or outlook.empty:
        st.warning("Sector data is not available yet.")
        return

    gain_cols = [
        "sector", "avg_return_1d_pct", "avg_return_5d_pct", "avg_return_20d_pct",
        "positive_1d_count", "positive_5d_count", "positive_20d_count", "gain_reason",
    ]
    loss_cols = [
        "sector", "avg_return_1d_pct", "avg_return_5d_pct", "avg_return_20d_pct",
        "weak_1d_stocks", "weak_5d_stocks", "weak_20d_stocks", "loss_reason",
    ]
    forward_cols = [
        "sector", "sector_prediction_score", "predicted_return_1d_pct",
        "predicted_return_5d_pct", "predicted_return_20d_pct",
        "positive_prediction_count", "avg_trend_score", "gain_reason",
    ]
    forward_loss_cols = [
        "sector", "sector_prediction_score", "predicted_return_1d_pct",
        "predicted_return_5d_pct", "predicted_return_20d_pct",
        "negative_prediction_count", "avg_trend_score", "loss_reason",
    ]

    st.caption(
        "Sector estimates use daily adjusted closes and simple momentum/trend scoring. "
        "They are planning signals, not guaranteed forecasts."
    )

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("**1. Sector Stocks Gained: 1D / 5D / 20D**")
        gainers = recent.sort_values("avg_return_5d_pct", ascending=False).head(6)
        st.dataframe(format_for_display(gainers[gain_cols]), width="stretch", hide_index=True, height=310)

    with top_right:
        st.markdown("**2. Sector Stocks Lost: 1D / 5D / 20D**")
        losers = recent.sort_values("avg_return_5d_pct", ascending=True).head(6)
        st.dataframe(format_for_display(losers[loss_cols]), width="stretch", hide_index=True, height=310)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("**3. Sectors Estimated To Gain: Next 1D / 5D / 20D**")
        gain_outlook = outlook.sort_values("sector_prediction_score", ascending=False).head(6)
        st.dataframe(format_for_display(gain_outlook[forward_cols]), width="stretch", hide_index=True, height=310)

    with bottom_right:
        st.markdown("**4. Sectors Estimated To Lose: Next 1D / 5D / 20D**")
        loss_outlook = outlook.sort_values("sector_prediction_score", ascending=True).head(6)
        st.dataframe(format_for_display(loss_outlook[forward_loss_cols]), width="stretch", hide_index=True, height=310)

    st.markdown("**Improvement Model: Sector Predictions vs Actuals**")
    st.caption(
        "The scheduled daily cycle saves sector predictions and compares them with actual "
        "1, 5, and 20 trading day sector returns as future data becomes available."
    )
    if sector_results.empty:
        st.warning("No sector prediction comparisons yet. This will populate after future trading days are downloaded.")
    else:
        display = format_for_display(sector_results)
        st.dataframe(display, width="stretch", hide_index=True, height=420)
        st.download_button(
            "Download sector prediction results",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="sector_prediction_results.csv",
            mime="text/csv",
        )

    combined = pd.concat(
        [
            recent.assign(section="recent_sector_performance"),
            outlook.assign(section="sector_forward_outlook"),
        ],
        ignore_index=True,
        sort=False,
    )
    st.download_button(
        "Download sector outlook",
        data=format_for_display(combined).to_csv(index=False).encode("utf-8"),
        file_name="sector_outlook.csv",
        mime="text/csv",
    )


def render_historical_learning_view(summary: pd.DataFrame, top25: pd.DataFrame, signals: pd.DataFrame, weights: pd.DataFrame):
    if summary.empty or top25.empty or weights.empty:
        st.warning("No historical learning results yet. Run `python3 src/historical_learning.py` once.")
        return

    best = summary.sort_values("accuracy", ascending=False).iloc[0]
    final = summary.iloc[-1]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Best accuracy", f"{best['accuracy'] * 100:.2f}%")
    metric_cols[1].metric("Best iteration", int(best["iteration"]))
    metric_cols[2].metric("Iterations run", int(final["iteration"]))
    metric_cols[3].metric("Backtest predictions", f"{int(best['total_predictions']):,}")

    if best["accuracy"] < 0.70:
        st.warning("The model did not reach 70% accuracy before the 15 iteration stop. The displayed result is the honest best backtest result.")
    else:
        st.success("The model reached the 70% directional accuracy target on this historical backtest.")

    st.markdown("**Top 25 Prediction Accuracy**")
    st.caption(
        "The first columns show the latest next-day prediction. Each completed day adds dated columns, "
        "with an AVERAGE row summarising direction and price-range accuracy for that date."
    )
    st.dataframe(style_learning_table(format_for_display(top25)), width="stretch", hide_index=True, height=620)

    with st.expander("Model details"):
        st.markdown("**Iteration Results**")
        st.dataframe(format_for_display(summary), width="stretch", hide_index=True, height=260)

        st.markdown("**Top Learned Signals**")
        st.dataframe(format_for_display(weights.head(50)), width="stretch", hide_index=True, height=520)

    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            "Download learning summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="historical_learning_summary.csv",
            mime="text/csv",
        )
    with download_cols[1]:
        st.download_button(
            "Download learned signals",
            data=weights.to_csv(index=False).encode("utf-8"),
            file_name="historical_learning_weights.csv",
            mime="text/csv",
        )
    with download_cols[2]:
        st.download_button(
            "Download top 25 accuracy",
            data=top25.to_csv(index=False).encode("utf-8"),
            file_name="historical_learning_top25_recent_accuracy.csv",
            mime="text/csv",
        )


def render_ml_signals_view(predictions: pd.DataFrame, accuracy_by_date: pd.DataFrame):
    if predictions.empty:
        st.warning("No ML predictions yet. Run `python3 src/run_ml_pipeline.py` in the ML project.")
        return

    model_accuracy = None
    if not accuracy_by_date.empty and "direction_accuracy" in accuracy_by_date.columns:
        model_accuracy = pd.to_numeric(accuracy_by_date["direction_accuracy"], errors="coerce").mean()

    cols = st.columns(5)
    cols[0].metric("Stocks analysed", f"{len(predictions):,}")
    cols[1].metric("Predicted Up", f"{(predictions['predicted_direction'] == 'Up').sum():,}")
    cols[2].metric("Down or Flat", f"{(predictions['predicted_direction'] != 'Up').sum():,}")
    cols[3].metric("Avg probability Up", f"{pd.to_numeric(predictions['probability_up'], errors='coerce').mean() * 100:.2f}%")
    cols[4].metric("Model accuracy", "n/a" if pd.isna(model_accuracy) else f"{model_accuracy * 100:.2f}%")

    filter_cols = st.columns(5)
    sector_options = sorted(predictions["sector"].dropna().unique().tolist()) if "sector" in predictions.columns else []
    selected_ml_sectors = filter_cols[0].multiselect("ML sector", options=sector_options, default=sector_options)
    confidence_options = sorted(predictions["confidence_level"].dropna().unique().tolist())
    selected_confidence = filter_cols[1].multiselect("Confidence", options=confidence_options, default=confidence_options)
    direction_options = sorted(predictions["predicted_direction"].dropna().unique().tolist())
    selected_direction = filter_cols[2].multiselect("Direction", options=direction_options, default=direction_options)
    min_probability = filter_cols[3].number_input("Min probability Up", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    min_stock_accuracy = filter_cols[4].number_input("Min stock accuracy", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    filtered = predictions.copy()
    if selected_ml_sectors:
        filtered = filtered[filtered["sector"].isin(selected_ml_sectors)]
    if selected_confidence:
        filtered = filtered[filtered["confidence_level"].isin(selected_confidence)]
    if selected_direction:
        filtered = filtered[filtered["predicted_direction"].isin(selected_direction)]
    filtered = filtered[pd.to_numeric(filtered["probability_up"], errors="coerce") >= min_probability]
    filtered = filtered[pd.to_numeric(filtered["stock_level_accuracy"], errors="coerce") >= min_stock_accuracy]

    display_cols = [
        "symbol", "companyName", "sector", "prediction_as_of_date",
        "expected_next_trading_date", "latest_price", "probability_up",
        "predicted_direction", "confidence_level",
        "predicted_next_day_high",
        "predicted_next_day_low",
        "predicted_next_day_range",
        "stock_level_accuracy", "stock_range_accuracy_pct",
    ]
    display_cols = [col for col in display_cols if col in filtered.columns]
    st.markdown("**Next day predictions with dates**")
    st.caption(
        "These are the latest ML predictions. prediction_as_of_date is the exact market date used; expected_next_trading_date means the next market session after that date."
    )
    display = format_for_display(filtered[display_cols])
    st.dataframe(display, width="stretch", hide_index=True, height=680)
    st.download_button(
        "Download ML predictions",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="ml_predictions.csv",
        mime="text/csv",
    )


def render_ml_prediction_review_view(review_history: pd.DataFrame, accuracy_by_date: pd.DataFrame):
    if review_history.empty:
        st.warning("No ML prediction review history yet. Run `python3 src/run_ml_pipeline.py` in the ML project.")
        return

    st.markdown(
        """
        <style>
          .review-nav-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 6px 0 18px;
          }
          .review-nav-card {
            border: 1px solid #d7dde8;
            border-radius: 4px;
            background: #ffffff;
            padding: 14px 14px 12px;
            min-height: 128px;
          }
          .review-nav-card h4 {
            margin: 0 0 8px;
            font-size: 15px;
            color: #1f2937;
          }
          .review-nav-card p {
            margin: 0 0 10px;
            font-size: 13px;
            line-height: 1.35;
            color: #4b5563;
          }
          .review-nav-card a {
            font-size: 13px;
            font-weight: 600;
            color: #1d4ed8;
            text-decoration: none;
          }
          @media (max-width: 1100px) {
            .review-nav-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 700px) {
            .review-nav-grid { grid-template-columns: 1fr; }
          }
        </style>
        <div class="review-nav-grid">
          <div class="review-nav-card">
            <h4>Stock Audit</h4>
            <p>Review one stock over the selected lookback period with predicted direction, actual direction, high, low, misses, and final result.</p>
            <a href="#stock-audit">Open stock audit</a>
          </div>
          <div class="review-nav-card">
            <h4>Selected Stock Trend</h4>
            <p>See whether the selected stock's prediction quality is improving or failing across recent prediction dates.</p>
            <a href="#selected-stock-trend">Open trend</a>
          </div>
          <div class="review-nav-card">
            <h4>Market Scorecard</h4>
            <p>Compare day-by-day performance across all reviewed stocks: direction accuracy, high miss, low miss, and overall score.</p>
            <a href="#market-scorecard">Open scorecard</a>
          </div>
          <div class="review-nav-card">
            <h4>Stock By Date</h4>
            <p>Audit every stock in a wide table where each date adds close direction, high, low, and dollar movement comparison columns.</p>
            <a href="#stock-by-date">Open wide table</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = review_history.copy()
    prediction_date_col = "prediction_as_of_date" if "prediction_as_of_date" in history.columns else "date"
    history[prediction_date_col] = pd.to_datetime(history[prediction_date_col], errors="coerce")
    if "actual_review_date" in history.columns:
        history["actual_review_date"] = pd.to_datetime(history["actual_review_date"], errors="coerce")

    symbols = sorted(history["symbol"].dropna().unique().tolist())
    default_symbol_index = symbols.index("AAPL") if "AAPL" in symbols else 0
    filter_cols = st.columns([2, 1])
    selected_symbol = filter_cols[0].selectbox("Stock", options=symbols, index=default_symbol_index)
    lookback_days = filter_cols[1].selectbox("Look back", options=[5, 10, 20, 50], index=0, format_func=lambda value: f"Last {value} reviewed days")

    stock_history = history[history["symbol"] == selected_symbol].sort_values(prediction_date_col, ascending=False).copy()
    recent_dates = stock_history[prediction_date_col].dropna().dt.date.drop_duplicates().head(lookback_days)
    audit = stock_history[stock_history[prediction_date_col].dt.date.isin(recent_dates)].sort_values(prediction_date_col, ascending=False).copy()

    correct = pd.to_numeric(audit["correct_prediction"], errors="coerce").fillna(False).astype(bool)
    high_miss = pd.to_numeric(audit.get("high_absolute_error_pct"), errors="coerce")
    low_miss = pd.to_numeric(audit.get("low_absolute_error_pct"), errors="coerce")
    combined_score = pd.to_numeric(audit.get("combined_accuracy_pct"), errors="coerce")

    cols = st.columns(5)
    cols[0].metric("Reviewed days", f"{len(audit):,}")
    cols[1].metric("Direction accuracy", f"{correct.mean() * 100:.2f}%")
    cols[2].metric("Avg high miss", f"{high_miss.mean():.2f}%")
    cols[3].metric("Avg low miss", f"{low_miss.mean():.2f}%")
    plain_score = pd.to_numeric(audit.get("plain_overall_accuracy_pct"), errors="coerce")
    cols[4].metric("Overall score", f"{plain_score.mean():.2f}%")

    audit_display = pd.DataFrame(
        {
            "Prediction Date": audit[prediction_date_col].dt.date.astype(str),
            "Actual Date Checked": audit["actual_review_date"].dt.date.astype(str) if "actual_review_date" in audit.columns else "",
            "Stock": audit["symbol"],
            "Predicted Direction": audit["predicted_direction"],
            "Actual Direction": audit["actual_direction"],
            "Direction Result": audit["prediction_review"],
            "Direction Score %": audit.get("direction_score_pct"),
            "Predicted High": audit.get("predicted_next_day_high"),
            "Actual High": audit.get("actual_next_day_high"),
            "High Miss %": audit.get("high_absolute_error_pct"),
            "Predicted Low": audit.get("predicted_next_day_low"),
            "Actual Low": audit.get("actual_next_day_low"),
            "Low Miss %": audit.get("low_absolute_error_pct"),
            "Overall Accuracy %": audit.get("plain_overall_accuracy_pct"),
            "Overall Result": audit.get("plain_result"),
        }
    )

    st.markdown('<a id="stock-audit"></a>', unsafe_allow_html=True)
    st.markdown("**Prediction audit by stock**")
    st.caption(
        "This is the human audit view: predicted values are next to actual values. If the Up/Down direction is wrong, the overall result is a failure even when high/low prices are close."
    )
    st.dataframe(format_for_display(audit_display), width="stretch", hide_index=True, height=620)

    st.markdown('<a id="selected-stock-trend"></a>', unsafe_allow_html=True)
    st.markdown("**Selected stock recent accuracy**")
    stock_trend_display = audit_display[
        ["Prediction Date", "Overall Accuracy %", "Overall Result", "Direction Result", "High Miss %", "Low Miss %"]
    ].copy()
    st.dataframe(format_for_display(stock_trend_display), width="stretch", hide_index=True, height=360)

    if not accuracy_by_date.empty:
        daily_summary = accuracy_by_date.sort_values("date", ascending=False).head(lookback_days).copy()
        summary_cols = [
            "date", "number_of_predictions", "direction_accuracy",
            "average_high_miss_pct", "average_low_miss_pct", "average_plain_overall_accuracy_pct",
        ]
        summary_cols = [col for col in summary_cols if col in daily_summary.columns]
        st.markdown('<a id="market-scorecard"></a>', unsafe_allow_html=True)
        st.markdown("**Daily prediction performance**")
        st.dataframe(format_for_display(daily_summary[summary_cols]), width="stretch", hide_index=True, height=360)

    wide_audit = build_stock_daily_audit_wide(history, max_dates=lookback_days)
    if not wide_audit.empty:
        st.markdown('<a id="stock-by-date"></a>', unsafe_allow_html=True)
        st.markdown("**Stock prediction audit by date**")
        st.caption(
            "Each prediction date adds four columns: Daily Close Direction, High, Low, and High/Low $ Movement. "
            "Daily Close Direction is green when predicted Up/Down matched the next trading day's closing direction. "
            "Movement is green within +/- $1, red when actual movement was at least $2 less than predicted, and blue when actual movement was at least $2 more than predicted."
        )

        def highlight_direction_cells(value):
            text = str(value)
            if text.startswith("Correct") or text.startswith("Close"):
                return "background-color: #dcfce7; color: #166534;"
            if text.startswith("Wrong") or text.startswith("Less than predicted"):
                return "background-color: #fee2e2; color: #991b1b;"
            if text.startswith("More than predicted"):
                return "background-color: #dbeafe; color: #1e40af;"
            if text.startswith("Near"):
                return "background-color: #fef9c3; color: #854d0e;"
            return ""

        styled_wide_audit = wide_audit.style.map(highlight_direction_cells)
        st.dataframe(styled_wide_audit, width="stretch", hide_index=True, height=780)

    st.download_button(
        "Download stock prediction audit by date",
        data=(wide_audit if "wide_audit" in locals() and not wide_audit.empty else audit_display).to_csv(index=False).encode("utf-8"),
        file_name="ml_prediction_stock_daily_audit_wide.csv",
        mime="text/csv",
    )


def render_ml_backtest_view(backtest: pd.DataFrame, accuracy_by_date: pd.DataFrame):
    if backtest.empty:
        st.warning("No ML backtest yet. Run `python3 src/run_ml_pipeline.py` in the ML project.")
        return

    accuracy = pd.to_numeric(backtest["correct_prediction"], errors="coerce").mean()
    st.metric("Overall ML test accuracy", f"{accuracy * 100:.2f}%")
    st.markdown("**Accuracy by date**")
    if not accuracy_by_date.empty:
        st.dataframe(format_for_display(accuracy_by_date), width="stretch", hide_index=True, height=320)

    st.markdown("**Accuracy by stock**")
    stock_accuracy = backtest.groupby("symbol").agg(
        predictions=("correct_prediction", "count"),
        stock_level_accuracy=("correct_prediction", "mean"),
        average_probability_up=("probability_up", "mean"),
    ).reset_index().sort_values("stock_level_accuracy", ascending=False)
    st.dataframe(format_for_display(stock_accuracy), width="stretch", hide_index=True, height=360)

    st.markdown("**Backtest rows**")
    display = format_for_display(backtest)
    st.dataframe(display, width="stretch", hide_index=True, height=780)
    st.download_button(
        "Download ML backtest",
        data=backtest.to_csv(index=False).encode("utf-8"),
        file_name="ml_backtest.csv",
        mime="text/csv",
    )


def render_agentic_insights_view(summary: pd.DataFrame):
    if summary.empty:
        st.warning("No agentic insight summary yet. Run `python3 src/run_ml_pipeline.py` in the ML project.")
        return
    for _, row in summary.iterrows():
        title = str(row["section"]).replace("_", " ").title()
        st.markdown(f"**{title}**")
        st.write(row["insight"])


def render_model_explanation_view():
    st.markdown("**What the ML model estimates**")
    st.write(
        "This ML model uses historical price behaviour to estimate whether the next trading day close "
        "may be higher than the current close. It is a statistical screening tool. It does not predict "
        "guaranteed returns and should not be used as investment advice."
    )
    st.markdown("**Features used**")
    st.write(
        "The model uses recent returns, volatility, average daily movement, positive close streak, "
        "distance from recent highs, drawdown from one-year high, optional volume change, sector, "
        "and company metadata where available."
    )
    st.markdown("**Models compared**")
    st.write(
        "The pipeline compares logistic regression and random forest using a time-based train/test split. "
        "The selected model is whichever performs best on test-set direction accuracy, with precision "
        "and confidence used as tie-breakers."
    )


def render_model_limitations_view():
    st.warning(
        "This dashboard is for screening and research only. It is not investment advice, "
        "does not execute trades, and does not guarantee future performance."
    )
    st.write(
        "The ML layer uses historical CSV data and does not know about live news, earnings surprises, "
        "liquidity, corporate actions beyond adjusted prices, macro shocks, transaction costs, or your "
        "personal risk profile. Treat high-confidence rows as items to review, not actions to take."
    )


st.set_page_config(page_title="ML Stock Prediction Dashboard", layout="wide")
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      [data-testid="collapsedControl"] { display: none; }
      .block-container { padding-top: 2rem; max-width: 98vw; }
    </style>
    """,
    unsafe_allow_html=True,
)
if ML_HTML_REPORT_FILE.exists():
    components.html(ML_HTML_REPORT_FILE.read_text(encoding="utf-8"), height=12000, scrolling=True)
    st.stop()

st.title("ML Stock Prediction Dashboard")
st.write(
    "Review next-day ML predictions, compare predicted Up/Down, high, and low against actual market results, "
    "and track whether prediction quality improves or worsens by date."
)

if not PRICE_HISTORY_FILE.exists() or not DAILY_REPORT_FILE.exists():
    st.warning("Run the daily monitor first to generate the latest CSV files.")
    st.code("python3 src/run_daily_monitor.py")
    st.stop()

price_history = load_csv(PRICE_HISTORY_FILE, file_mtime(PRICE_HISTORY_FILE))
daily_report = load_csv(DAILY_REPORT_FILE, file_mtime(DAILY_REPORT_FILE))
saved_predictions = load_optional_csv(NEXT_DAY_PREDICTIONS_FILE, file_mtime(NEXT_DAY_PREDICTIONS_FILE))
prediction_results = load_optional_csv(NEXT_DAY_RESULTS_FILE, file_mtime(NEXT_DAY_RESULTS_FILE))
saved_top_picks = load_optional_csv(TOP_INTRADAY_PICKS_FILE, file_mtime(TOP_INTRADAY_PICKS_FILE))
top_pick_results = load_optional_csv(TOP_INTRADAY_PICK_RESULTS_FILE, file_mtime(TOP_INTRADAY_PICK_RESULTS_FILE))
sector_prediction_results = load_optional_csv(SECTOR_RESULTS_FILE, file_mtime(SECTOR_RESULTS_FILE))
learning_summary = load_optional_csv(LEARNING_SUMMARY_FILE, file_mtime(LEARNING_SUMMARY_FILE))
learning_backtest = load_optional_csv(LEARNING_BACKTEST_FILE, file_mtime(LEARNING_BACKTEST_FILE))
learning_signals = load_optional_csv(LEARNING_SIGNALS_FILE, file_mtime(LEARNING_SIGNALS_FILE))
learning_weights = load_optional_csv(LEARNING_WEIGHTS_FILE, file_mtime(LEARNING_WEIGHTS_FILE))
learning_top25 = load_optional_csv(LEARNING_TOP25_FILE, file_mtime(LEARNING_TOP25_FILE))
ml_predictions = load_optional_csv(ML_PREDICTIONS_FILE, file_mtime(ML_PREDICTIONS_FILE))
ml_backtest = load_optional_csv(ML_BACKTEST_FILE, file_mtime(ML_BACKTEST_FILE))
ml_review_history = load_optional_csv(ML_REVIEW_HISTORY_FILE, file_mtime(ML_REVIEW_HISTORY_FILE))
ml_accuracy_by_date = load_optional_csv(ML_ACCURACY_BY_DATE_FILE, file_mtime(ML_ACCURACY_BY_DATE_FILE))
ml_summary = load_optional_csv(ML_SUMMARY_FILE, file_mtime(ML_SUMMARY_FILE))

review_tab, signals_tab, backtest_tab, insights_tab, explanation_tab, limitations_tab = st.tabs(
    [
        "Daily Review",
        "Next-Day Predictions",
        "Backtest",
        "Insights",
        "Model Explanation",
        "Limitations",
    ]
)

with review_tab:
    render_ml_prediction_review_view(ml_review_history, ml_accuracy_by_date)
with signals_tab:
    render_ml_signals_view(ml_predictions, ml_accuracy_by_date)
with backtest_tab:
    render_ml_backtest_view(ml_backtest, ml_accuracy_by_date)
with insights_tab:
    render_agentic_insights_view(ml_summary)
with explanation_tab:
    render_model_explanation_view()
with limitations_tab:
    render_model_limitations_view()

st.stop()

analysis_type = st.sidebar.selectbox(
    "Analysis type",
    [
        "ML daily prediction review",
        "ML signals",
        "ML backtest",
        "Agentic insights",
        "Model explanation",
        "Model limitations",
        "One year performance",
        "Daily dollar move range",
        "Positive closes",
        "Average daily movement",
        "Near 20 day high",
        "Highest 5 day return",
        "Predicted next day range",
        "Top 10 next day intraday picks",
        "Single stock trader view",
        "Saved next day predictions",
        "Prediction vs actual results",
        "Saved top 10 intraday picks",
        "Top 10 picks vs actual results",
        "Sector Outlook",
        "Historical Learning Backtest",
        "Ask a stock question",
    ],
)
lookback_days = st.sidebar.number_input("Lookback days", min_value=1, max_value=252, value=5, step=1)
min_dollar_move = st.sidebar.number_input("Minimum dollar move", min_value=0.0, value=1.0, step=0.25)
max_dollar_move = st.sidebar.number_input("Maximum dollar move", min_value=0.0, value=10.0, step=0.25)
min_stock_price = st.sidebar.number_input("Minimum stock price", min_value=0.0, value=0.0, step=1.0)
within_high_pct = st.sidebar.number_input("Within percent of 20 day high", min_value=0.0, value=3.0, step=0.5)

sectors = []
if "sector" in daily_report.columns:
    sectors = sorted(daily_report["sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect("Sector filter", options=sectors, default=sectors)

symbols = sorted(price_history["symbol"].dropna().unique().tolist())
selected_symbol = st.sidebar.selectbox(
    "Stock for single stock view",
    options=symbols,
    format_func=lambda symbol: stock_label(symbol, daily_report),
)

st.header(analysis_type)
if analysis_type in ANALYSIS_EXPLANATIONS:
    st.caption(ANALYSIS_EXPLANATIONS[analysis_type])
if analysis_type == "Single stock trader view":
    st.subheader(stock_label(selected_symbol, daily_report))

with st.expander("How to use this dashboard"):
    st.write(
        "Start with ML daily prediction review to audit what the model predicted versus what actually happened. "
        "Use ML signals for the latest unevaluated next-trading-day predictions. The original rule-based screens "
        "are still available lower in the sidebar for comparison."
    )
    for name, explanation in ANALYSIS_EXPLANATIONS.items():
        st.markdown(f"**{name}:** {explanation}")

with st.expander("Column glossary"):
    glossary = pd.DataFrame(
        [{"Column": column, "Meaning": meaning} for column, meaning in COLUMN_GLOSSARY.items()]
    )
    st.dataframe(glossary, width="stretch", hide_index=True, height=520)

if analysis_type == "Single stock trader view":
    render_single_stock_view(price_history, daily_report, selected_symbol, lookback_days)
    st.stop()

if analysis_type == "Sector Outlook":
    render_sector_outlook(price_history, daily_report, sector_prediction_results)
    st.stop()

if analysis_type == "Historical Learning Backtest":
    render_historical_learning_view(learning_summary, learning_top25, learning_signals, learning_weights)
    st.stop()

if analysis_type == "ML signals":
    render_ml_signals_view(ml_predictions, ml_accuracy_by_date)
    st.stop()

if analysis_type == "ML daily prediction review":
    render_ml_prediction_review_view(ml_review_history, ml_accuracy_by_date)
    st.stop()

if analysis_type == "ML backtest":
    render_ml_backtest_view(ml_backtest, ml_accuracy_by_date)
    st.stop()

if analysis_type == "Agentic insights":
    render_agentic_insights_view(ml_summary)
    st.stop()

if analysis_type == "Model explanation":
    render_model_explanation_view()
    st.stop()

if analysis_type == "Model limitations":
    render_model_limitations_view()
    st.stop()

if analysis_type == "Ask a stock question":
    render_stock_query_view(
        price_history,
        daily_report,
        learning_top25,
        saved_predictions,
        saved_top_picks,
        sector_prediction_results,
        selected_symbol,
        symbols,
    )
    st.stop()

if analysis_type == "Saved next day predictions":
    if saved_predictions.empty:
        st.warning("No saved predictions yet. Run `python3 src/run_daily_prediction_cycle.py` once, then this table will populate.")
        st.stop()
    display = format_for_display(saved_predictions)
    st.caption(f"{len(display):,} saved prediction rows from {NEXT_DAY_PREDICTIONS_FILE.name}.")
    st.info("Tip: scroll horizontally inside the table to see every column.")
    st.dataframe(display, width="stretch", hide_index=True, height=720)
    st.download_button(
        "Download saved next day predictions",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="next_day_predictions.csv",
        mime="text/csv",
    )
    st.stop()

if analysis_type == "Prediction vs actual results":
    render_prediction_date_status(price_history, saved_predictions, prediction_results)
    if prediction_results.empty:
        st.warning(
            "No prediction comparisons yet. The daily cycle needs at least one saved prediction "
            "and then a later trading day with actual data."
        )
        st.stop()
    render_prediction_result_summary(prediction_results)
    display = format_for_display(prediction_results)
    st.caption(f"{len(display):,} prediction comparison rows from {NEXT_DAY_RESULTS_FILE.name}.")
    st.info("Tip: scroll horizontally inside the table to see every column.")
    st.dataframe(display, width="stretch", hide_index=True, height=720)
    st.download_button(
        "Download prediction vs actual results",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="next_day_prediction_results.csv",
        mime="text/csv",
    )
    st.stop()

if analysis_type == "Saved top 10 intraday picks":
    if saved_top_picks.empty:
        st.warning("No saved Top 10 picks yet. Run `python3 src/run_daily_prediction_cycle.py` once, then this table will populate.")
        st.stop()
    display = format_for_display(saved_top_picks)
    st.caption(f"{len(display):,} saved Top 10 pick rows from {TOP_INTRADAY_PICKS_FILE.name}.")
    st.info("Tip: scroll horizontally inside the table to see every column.")
    st.dataframe(display, width="stretch", hide_index=True, height=720)
    st.download_button(
        "Download saved Top 10 intraday picks",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="top_10_intraday_picks.csv",
        mime="text/csv",
    )
    st.stop()

if analysis_type == "Top 10 picks vs actual results":
    render_top_pick_date_status(price_history, saved_top_picks, top_pick_results)
    if top_pick_results.empty:
        st.warning(
            "No Top 10 pick reviews yet. The daily cycle needs saved picks and then "
            "a later trading day with actual high/low data."
        )
        st.stop()
    render_top_pick_result_summary(top_pick_results)
    display = format_for_display(top_pick_results)
    st.caption(f"{len(display):,} Top 10 pick review rows from {TOP_INTRADAY_PICK_RESULTS_FILE.name}.")
    st.info("Tip: scroll horizontally inside the table to see every column.")
    st.dataframe(display, width="stretch", hide_index=True, height=720)
    st.download_button(
        "Download Top 10 picks vs actual results",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="top_10_intraday_pick_results.csv",
        mime="text/csv",
    )
    st.stop()

if analysis_type == "One year performance":
    raw_results = daily_report.copy()
elif analysis_type == "Daily dollar move range":
    raw_results = stocks_with_daily_move_between(
        price_history,
        min_move=min_dollar_move,
        max_move=max_dollar_move,
        lookback_days=lookback_days,
    )
elif analysis_type == "Positive closes":
    raw_results = stocks_with_positive_closes(price_history, lookback_days=lookback_days)
elif analysis_type == "Average daily movement":
    raw_results = stocks_with_average_daily_movement_above(
        price_history,
        threshold=min_dollar_move,
        lookback_days=lookback_days,
    )
elif analysis_type == "Near 20 day high":
    raw_results = stocks_within_percent_of_20_day_high(price_history, percent=within_high_pct)
elif analysis_type == "Predicted next day range":
    raw_results = stocks_with_predicted_next_day_range(
        price_history,
        lookback_days=lookback_days,
        top_n=25,
    )
elif analysis_type == "Top 10 next day intraday picks":
    raw_results = top_intraday_play_candidates(
        price_history,
        lookback_days=lookback_days,
        top_n=10,
    )
else:
    raw_results = stocks_with_highest_5_day_return(price_history, top_n=250)

results = apply_profile_filters(raw_results, daily_report, min_stock_price, selected_sectors)
display = format_for_display(results)

st.caption(
    f"Using {PRICE_HISTORY_FILE.name} and {DAILY_REPORT_FILE.name}. "
    f"{len(display):,} matching rows."
)
st.info("Tip: scroll horizontally inside the table to see every column.")
st.dataframe(display, width="stretch", hide_index=True, height=720)

csv_data = display.to_csv(index=False).encode("utf-8")
download_cols = st.columns(3)
with download_cols[0]:
    st.download_button(
        "Download current filtered view",
        data=csv_data,
        file_name=f"{analysis_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
with download_cols[1]:
    st.download_button(
        "Download full one year performance",
        data=daily_report.to_csv(index=False).encode("utf-8"),
        file_name="daily_stock_monitor.csv",
        mime="text/csv",
    )
with download_cols[2]:
    st.download_button(
        "Download full price history",
        data=price_history.to_csv(index=False).encode("utf-8"),
        file_name="price_history.csv",
        mime="text/csv",
    )
