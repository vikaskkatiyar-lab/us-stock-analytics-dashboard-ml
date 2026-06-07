import pandas as pd


def _top_symbols(frame: pd.DataFrame, count: int = 5) -> str:
    if frame.empty:
        return "None available"
    labels = []
    for _, row in frame.head(count).iterrows():
        labels.append(f"{row['symbol']} ({row.get('probability_up', 0):.1%} up probability)")
    return ", ".join(labels)


def build_agentic_insights(
    ml_predictions: pd.DataFrame,
    ml_backtest: pd.DataFrame,
    daily_report: pd.DataFrame,
) -> dict:
    predictions = ml_predictions.copy()
    if predictions.empty:
        return {
            "summary": "No ML prediction rows are available yet. Run the ML pipeline first.",
            "strongest_upward_signals": "None available",
            "weakest_signals": "None available",
            "sector_patterns": "None available",
            "rule_ml_agreement": "None available",
            "caution_list": "No model output is available to review.",
            "limitations": "This is a statistical screening layer only, not investment advice.",
        }

    up = predictions[predictions["predicted_direction"] == "Up"].sort_values("probability_up", ascending=False)
    weak = predictions.sort_values("probability_up", ascending=True)
    sector = predictions.groupby("sector").agg(
        stocks=("symbol", "count"),
        up_signals=("predicted_direction", lambda values: (values == "Up").sum()),
        average_probability_up=("probability_up", "mean"),
    ).reset_index()
    sector = sector.sort_values(["up_signals", "average_probability_up"], ascending=False)

    report = daily_report.copy()
    agreement = pd.DataFrame()
    if "return_5d_pct" in report.columns:
        report["rule_direction"] = pd.to_numeric(report["return_5d_pct"], errors="coerce").apply(
            lambda value: "Up" if value > 0 else "Down or Flat"
        )
        agreement = predictions.merge(report[["symbol", "rule_direction"]], on="symbol", how="left")
        agreement = agreement[agreement["predicted_direction"] == agreement["rule_direction"]]

    caution = predictions[
        (predictions["probability_up"] >= 0.65) &
        (pd.to_numeric(predictions["stock_level_accuracy"], errors="coerce") < 0.5)
    ]
    rule_strong_ml_weak = pd.DataFrame()
    if "return_5d_pct" in report.columns:
        strong_rule = report[pd.to_numeric(report["return_5d_pct"], errors="coerce") > 3][["symbol", "return_5d_pct"]]
        rule_strong_ml_weak = predictions.merge(strong_rule, on="symbol", how="inner")
        rule_strong_ml_weak = rule_strong_ml_weak[rule_strong_ml_weak["confidence_level"] == "Low"]

    direction_accuracy = ml_backtest["correct_prediction"].mean() if not ml_backtest.empty else None
    caution_points = [
        "The model uses historical price behaviour only and cannot account for news, earnings surprises, liquidity, or macro shocks.",
        "High probability should be treated as a screening signal, not a recommendation.",
    ]
    if direction_accuracy is not None:
        caution_points.insert(0, f"Recent test-set direction accuracy is {direction_accuracy:.1%}.")
    if not caution.empty:
        caution_points.append(
            "High-probability stocks with weaker stock-level history: " + _top_symbols(caution, 5)
        )
    if not rule_strong_ml_weak.empty:
        caution_points.append(
            "Strong recent rule-based performers with low ML confidence: " + _top_symbols(rule_strong_ml_weak, 5)
        )

    return {
        "summary": (
            "This ML layer estimates whether the next trading day close may be higher than the current close. "
            "It is a statistical screening signal only and is not investment advice."
        ),
        "strongest_upward_signals": _top_symbols(up, 8),
        "weakest_signals": _top_symbols(weak, 8),
        "sector_patterns": "; ".join(
            [
                f"{row['sector']}: {int(row['up_signals'])}/{int(row['stocks'])} Up, avg probability {row['average_probability_up']:.1%}"
                for _, row in sector.head(6).iterrows()
            ]
        ),
        "rule_ml_agreement": _top_symbols(agreement.sort_values("probability_up", ascending=False), 8),
        "caution_list": " ".join(caution_points),
        "limitations": (
            "Do not use this output for automated trading or trade execution. "
            "It does not guarantee returns and should be combined with independent review."
        ),
    }


def insights_to_frame(insights: dict) -> pd.DataFrame:
    return pd.DataFrame([{"section": key, "insight": value} for key, value in insights.items()])
