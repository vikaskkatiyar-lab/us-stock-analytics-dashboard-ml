# US Top 250 Stock Analytics Dashboard ML

This is a separate machine-learning version of the existing non-ML Streamlit stock analytics dashboard.

The project keeps the original rule-based dashboard, CSV outputs, daily monitor, reports, filters, and tables. The ML layer is added separately so the original screening workflow remains available.

## Purpose

The dashboard helps review the top US-listed stocks using:

- rule-based analytics from historical price and report CSVs
- a baseline ML direction model
- cautious plain-English agentic insights

It does not execute trades and does not provide investment advice.

## Rule-Based Analytics vs ML Analytics

Rule-based analytics use deterministic calculations such as recent returns, positive closes, average daily movement, near-high filters, and sector summaries.

ML analytics train models on historical price behaviour to estimate whether the next trading day close may be higher than the current close. The ML output is a statistical screening signal only.

## How The ML Model Works

The ML pipeline uses `data/price_history.csv` and `reports/daily_stock_monitor.csv`.

Features include:

- 1, 5, 10, and 20 day returns
- 20 day volatility
- average daily movement
- positive close streak
- distance from 20 day high
- distance and drawdown from one-year high
- volume change when volume is available
- sector and company metadata where available

The target is:

```text
1 = next trading day close is higher than current close
0 = next trading day close is down or flat
```

The pipeline compares logistic regression and random forest using a time-based train/test split. It chooses the model with the best test-set direction accuracy, using precision and confidence as tie-breakers.

## Agentic Insights

The agentic insight layer converts rule-based and ML output into cautious plain-English screening notes:

- strongest upward ML signals
- weakest signals
- sector-level patterns
- rule-based and ML agreement
- high-probability but weak historical accuracy cases
- strong rule-based but weak ML confidence cases
- overall caution points

These insights are not recommendations.

## Run Daily Monitor

```bash
python3 src/run_daily_monitor.py
```

This refreshes the rule-based CSV files in `data/` and `reports/`.

## Run ML Pipeline

```bash
python3 src/run_ml_pipeline.py
```

This creates or refreshes:

```text
models/latest_model.pkl
models/latest_model_metadata.json
data/ml_predictions.csv
data/ml_backtest.csv
data/ml_prediction_review_history.csv
data/ml_accuracy_by_date.csv
reports/ml_summary.csv
reports/ml_dashboard.html
```

`reports/ml_dashboard.html` is a static HTML report you can open directly in a browser or upload/share as a simple report.
It includes a clear Next Day Predictions table and a Daily Prediction Review History table.

## Launch Dashboard Locally

```bash
python3 -m streamlit run app.py
```

If using a local virtual environment:

```bash
.venv/bin/python -m streamlit run app.py
```

## Deploy On Streamlit Cloud

1. Push this ML project to its own GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the ML GitHub repository.
4. Set the branch to `main`.
5. Set the main file path to `app.py`.
6. Deploy.

For the baseline ML deployment, no API key is required if the CSV files are committed.

## Automatic Daily Refresh

This project includes a GitHub Actions workflow:

```text
.github/workflows/daily-refresh.yml
```

It can be run manually from GitHub, and it is also scheduled to run at `01:00 UTC` Tuesday through Saturday. That timing is intended to refresh the dashboard after the prior US trading day has closed and data is available.

The workflow does three things:

1. runs `python src/run_daily_monitor.py`
2. runs `python src/run_ml_pipeline.py`
3. commits refreshed files in `data/`, `reports/`, and `models/`

After the workflow commits updated files to GitHub, Streamlit Community Cloud should redeploy or refresh the app from the latest repository state.

## Limitations

- The ML layer uses historical CSV data only.
- It cannot know future news, earnings surprises, macro events, liquidity changes, or transaction costs.
- Accuracy can change materially over time.
- High probability does not mean a stock should be bought.
- No trade execution is included.

## Disclaimer

This dashboard is a statistical screening tool for research. It is not financial advice, investment advice, or a trading system.
