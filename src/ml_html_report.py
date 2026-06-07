import html

import pandas as pd

from ml_audit import (
    build_daily_accuracy_trend,
    build_plain_prediction_audit,
    build_stock_daily_audit_wide,
)


def _format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
    return html.escape(str(value))


def _cell_style(value) -> str:
    text = str(value)
    if text.startswith("Correct") or text.startswith("Accurate"):
        return " style='background:#dcfce7;color:#166534;'"
    if text.startswith("Wrong") or text.startswith("Incorrect"):
        return " style='background:#fee2e2;color:#991b1b;'"
    if text.startswith("Close"):
        return " style='background:#fef9c3;color:#854d0e;'"
    return ""


def _table_html(df: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if df.empty:
        return "<p class='muted'>No rows available.</p>"
    cols = [col for col in columns if col in df.columns]
    headers = "".join([f"<th>{html.escape(col)}</th>" for col in cols])
    rows = []
    for _, row in df[cols].head(max_rows).iterrows():
        cells = "".join([f"<td{_cell_style(row[col])}>{_format_value(row[col])}</td>" for col in cols])
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_ml_html_report(
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    accuracy_by_date: pd.DataFrame,
    summary: pd.DataFrame,
    output_file,
) -> str:
    direction_accuracy = backtest["correct_prediction"].mean() if not backtest.empty else None
    up_count = int((predictions["predicted_direction"] == "Up").sum()) if not predictions.empty else 0
    down_count = int((predictions["predicted_direction"] != "Up").sum()) if not predictions.empty else 0
    avg_probability = predictions["probability_up"].mean() if not predictions.empty else None

    insight_blocks = ""
    if not summary.empty:
        for _, row in summary.iterrows():
            title = str(row["section"]).replace("_", " ").title()
            insight_blocks += (
                f"<section><h3>{html.escape(title)}</h3>"
                f"<p>{html.escape(str(row['insight']))}</p></section>"
            )

    next_day_cols = [
        "symbol", "companyName", "sector", "prediction_as_of_date",
        "expected_next_trading_date", "latest_price", "probability_up",
        "predicted_direction", "confidence_level", "predicted_next_day_high",
        "predicted_next_day_low", "predicted_next_day_range", "stock_level_accuracy",
        "stock_range_accuracy_pct",
    ]
    accuracy_cols = [
        "date", "number_of_predictions", "direction_accuracy",
        "average_high_miss_pct", "average_low_miss_pct",
        "average_plain_overall_accuracy_pct",
    ]
    sort_cols = ["prediction_as_of_date", "symbol"] if "prediction_as_of_date" in backtest.columns else ["date", "symbol"]
    review_history = backtest.sort_values(sort_cols, ascending=[False, True])
    plain_audit = build_plain_prediction_audit(review_history)
    daily_trend = build_daily_accuracy_trend(review_history)
    stock_daily_audit_wide = build_stock_daily_audit_wide(review_history, max_dates=5)
    audit_cols = [
        "Prediction Date", "Checked Against Date", "Stock",
        "Predicted Up/Down", "Actual Up/Down", "Direction Result",
        "Direction Score %", "Predicted High", "Actual High", "High Miss %",
        "Predicted Low", "Actual Low", "Low Miss %",
        "Overall Accuracy %", "Overall Result",
    ]
    trend_cols = [
        "Prediction Date", "Stocks_Reviewed", "Direction_Accuracy_Pct",
        "Average_High_Miss_Pct", "Average_Low_Miss_Pct",
        "Overall_Accuracy_Pct", "Trend vs Previous Date", "Change vs Previous Date",
    ]
    wide_audit_cols = stock_daily_audit_wide.columns.tolist() if not stock_daily_audit_wide.empty else []

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Stock Analytics Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #6b7280; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; background: #f9fafb; }}
    .card strong {{ display: block; font-size: 22px; margin-top: 6px; }}
    .nav-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0; }}
    .nav-card {{ border: 1px solid #d7dde8; border-radius: 4px; background: #fff; padding: 14px; min-height: 128px; }}
    .nav-card h3 {{ margin: 0 0 8px; font-size: 15px; color: #1f2937; }}
    .nav-card p {{ margin: 0 0 10px; font-size: 13px; line-height: 1.35; color: #4b5563; }}
    .nav-card a {{ font-size: 13px; font-weight: 600; color: #1d4ed8; text-decoration: none; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 7px; text-align: right; vertical-align: top; }}
    th {{ background: #eef2ff; color: #312e81; position: sticky; top: 0; }}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2), td:nth-child(3), th:nth-child(3) {{ text-align: left; }}
    section {{ margin: 18px 0; }}
    .notice {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px; border-radius: 8px; }}
    @media (max-width: 1100px) {{ .nav-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 700px) {{ .nav-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>ML Stock Analytics Report</h1>
  <p class="muted">Static HTML report generated from the local CSV-based ML pipeline.</p>
  <div class="notice">
    This is a statistical screening report only. It is not investment advice and does not execute trades.
  </div>
  <div class="cards">
    <div class="card">Stocks analysed<strong>{len(predictions):,}</strong></div>
    <div class="card">Predicted Up<strong>{up_count:,}</strong></div>
    <div class="card">Down or Flat<strong>{down_count:,}</strong></div>
    <div class="card">Average probability Up<strong>{'' if avg_probability is None else f'{avg_probability:.2%}'}</strong></div>
    <div class="card">Backtest accuracy<strong>{'' if direction_accuracy is None else f'{direction_accuracy:.2%}'}</strong></div>
  </div>
  <div class="nav-grid">
    <div class="nav-card">
      <h3>Next Day Predictions</h3>
      <p>Latest unevaluated predictions with the exact prediction date, expected next trading date, direction, high, low, and range.</p>
      <a href="#next-day-predictions">Open table</a>
    </div>
    <div class="nav-card">
      <h3>Daily Trend</h3>
      <p>Market-wide day-by-day performance across reviewed stocks, including direction accuracy and high/low miss percentages.</p>
      <a href="#daily-trend">Open trend</a>
    </div>
    <div class="nav-card">
      <h3>Stock By Date</h3>
      <p>Wide audit table where every prediction date adds Direction, High, Low, and High/Low Quality columns.</p>
      <a href="#stock-by-date">Open wide table</a>
    </div>
    <div class="nav-card">
      <h3>Audit History</h3>
      <p>Detailed row-by-row history showing predicted values beside actual values and the final human-readable result.</p>
      <a href="#audit-history">Open audit</a>
    </div>
  </div>
  <h2>Agentic Insights</h2>
  {insight_blocks}
  <h2 id="next-day-predictions">Next Day Predictions</h2>
  <p class="muted">These rows are the latest unevaluated predictions. The prediction_as_of_date column shows the exact data date used. expected_next_trading_date is listed as after that date until new market data identifies the actual next trading date.</p>
  {_table_html(predictions, next_day_cols, 100)}
  <h2 id="market-scorecard">Market-Wide Daily Scorecard</h2>
  {_table_html(accuracy_by_date, accuracy_cols, 80)}
  <h2 id="daily-trend">Day-On-Day Accuracy Trend</h2>
  <p class="muted">This shows whether overall prediction quality improved or got worse from one prediction date to the next.</p>
  {_table_html(daily_trend, trend_cols, 80)}
  <h2 id="stock-by-date">Stock Prediction Audit By Date</h2>
  <p class="muted">Each prediction date adds four columns: Direction, High, Low, and High/Low Quality. Direction says whether Up/Down was correct. High and Low show predicted value with actual value in brackets. Quality is Accurate within 5%, Close within 15%, and Incorrect above 15%.</p>
  {_table_html(stock_daily_audit_wide, wide_audit_cols, 250)}
  <h2 id="audit-history">Prediction Audit History</h2>
  <p class="muted">This table is built for looking back by stock and date. Direction score is 100 when Up/Down was correct and 0 when wrong. If direction is wrong, the overall result is a failure even when high/low prices are close.</p>
  {_table_html(plain_audit, audit_cols, 500)}
</body>
</html>
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_text, encoding="utf-8")
    return str(output_file)
