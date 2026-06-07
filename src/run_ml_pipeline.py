import pandas as pd

from agentic_insights import build_agentic_insights, insights_to_frame
from config import DATA_DIR, REPORTS_DIR
from ml_audit import (
    build_daily_accuracy_trend,
    build_plain_prediction_audit,
    build_stock_accuracy_pivot,
    build_stock_accuracy_trend,
    build_stock_accuracy_verdict_pivot,
    build_stock_daily_audit_wide,
)
from ml_backtest import build_accuracy_by_date, build_accuracy_by_stock, build_backtest
from ml_features import build_ml_feature_frame, training_rows
from ml_html_report import build_ml_html_report
from ml_model import train_and_select_model
from ml_predictions import build_daily_predictions


PRICE_HISTORY_FILE = DATA_DIR / "price_history.csv"
DAILY_REPORT_FILE = REPORTS_DIR / "daily_stock_monitor.csv"
ML_PREDICTIONS_FILE = DATA_DIR / "ml_predictions.csv"
ML_BACKTEST_FILE = DATA_DIR / "ml_backtest.csv"
ML_REVIEW_HISTORY_FILE = DATA_DIR / "ml_prediction_review_history.csv"
ML_PLAIN_AUDIT_FILE = DATA_DIR / "ml_prediction_audit_simple.csv"
ML_DAILY_TREND_FILE = DATA_DIR / "ml_prediction_daily_trend.csv"
ML_STOCK_TREND_FILE = DATA_DIR / "ml_prediction_stock_trend.csv"
ML_STOCK_TREND_PIVOT_FILE = DATA_DIR / "ml_prediction_stock_trend_pivot.csv"
ML_STOCK_TREND_VERDICT_PIVOT_FILE = DATA_DIR / "ml_prediction_stock_trend_verdict_pivot.csv"
ML_STOCK_DAILY_AUDIT_WIDE_FILE = DATA_DIR / "ml_prediction_stock_daily_audit_wide.csv"
ML_ACCURACY_BY_DATE_FILE = DATA_DIR / "ml_accuracy_by_date.csv"
ML_SUMMARY_FILE = REPORTS_DIR / "ml_summary.csv"
ML_HTML_REPORT_FILE = REPORTS_DIR / "ml_dashboard.html"


def run_ml_pipeline() -> dict:
    price_history = pd.read_csv(PRICE_HISTORY_FILE)
    daily_report = pd.read_csv(DAILY_REPORT_FILE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_features = build_ml_feature_frame(price_history, daily_report)
    model_features = training_rows(all_features)
    trained = train_and_select_model(model_features)

    backtest = build_backtest(trained["test"], trained["model"], trained["model_name"])
    review_history = backtest.sort_values(["date", "symbol"], ascending=[False, True]).copy()
    plain_audit = build_plain_prediction_audit(review_history)
    daily_trend = build_daily_accuracy_trend(review_history)
    stock_trend = build_stock_accuracy_trend(review_history)
    stock_trend_pivot = build_stock_accuracy_pivot(review_history)
    stock_trend_verdict_pivot = build_stock_accuracy_verdict_pivot(review_history)
    stock_daily_audit_wide = build_stock_daily_audit_wide(review_history)
    accuracy_by_date = build_accuracy_by_date(backtest)
    accuracy_by_stock = build_accuracy_by_stock(backtest)
    predictions = build_daily_predictions(
        all_features,
        trained["model"],
        trained["model_name"],
        accuracy_by_stock,
    )
    insights = build_agentic_insights(predictions, backtest, daily_report)
    summary = insights_to_frame(insights)

    predictions.to_csv(ML_PREDICTIONS_FILE, index=False)
    backtest.to_csv(ML_BACKTEST_FILE, index=False)
    review_history.to_csv(ML_REVIEW_HISTORY_FILE, index=False)
    plain_audit.to_csv(ML_PLAIN_AUDIT_FILE, index=False)
    daily_trend.to_csv(ML_DAILY_TREND_FILE, index=False)
    stock_trend.to_csv(ML_STOCK_TREND_FILE, index=False)
    stock_trend_pivot.to_csv(ML_STOCK_TREND_PIVOT_FILE, index=False)
    stock_trend_verdict_pivot.to_csv(ML_STOCK_TREND_VERDICT_PIVOT_FILE, index=False)
    stock_daily_audit_wide.to_csv(ML_STOCK_DAILY_AUDIT_WIDE_FILE, index=False)
    accuracy_by_date.to_csv(ML_ACCURACY_BY_DATE_FILE, index=False)
    summary.to_csv(ML_SUMMARY_FILE, index=False)
    build_ml_html_report(predictions, backtest, accuracy_by_date, summary, ML_HTML_REPORT_FILE)

    return {
        "model_name": trained["model_name"],
        "model_file": str(trained["model_file"]),
        "metadata_file": str(trained["metadata_file"]),
        "ml_predictions_file": str(ML_PREDICTIONS_FILE),
        "ml_backtest_file": str(ML_BACKTEST_FILE),
        "ml_review_history_file": str(ML_REVIEW_HISTORY_FILE),
        "ml_plain_audit_file": str(ML_PLAIN_AUDIT_FILE),
        "ml_daily_trend_file": str(ML_DAILY_TREND_FILE),
        "ml_stock_trend_file": str(ML_STOCK_TREND_FILE),
        "ml_stock_trend_pivot_file": str(ML_STOCK_TREND_PIVOT_FILE),
        "ml_stock_trend_verdict_pivot_file": str(ML_STOCK_TREND_VERDICT_PIVOT_FILE),
        "ml_stock_daily_audit_wide_file": str(ML_STOCK_DAILY_AUDIT_WIDE_FILE),
        "ml_accuracy_by_date_file": str(ML_ACCURACY_BY_DATE_FILE),
        "ml_summary_file": str(ML_SUMMARY_FILE),
        "ml_html_report_file": str(ML_HTML_REPORT_FILE),
        "prediction_rows": len(predictions),
        "backtest_rows": len(backtest),
        "accuracy_dates": len(accuracy_by_date),
    }


if __name__ == "__main__":
    print(run_ml_pipeline())
