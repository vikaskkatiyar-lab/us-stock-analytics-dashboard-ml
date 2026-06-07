import pandas as pd
from jinja2 import Template
from config import REPORTS_DIR, ALERT_DAILY_MOVE_PCT, ALERT_DRAWDOWN_PCT

HTML_TEMPLATE = """
<html>
<head>
<title>US Top 250 Stock Monitor</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
h1 { margin-bottom: 0; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border: 1px solid #ddd; padding: 6px; text-align: right; }
th { background: #f4f4f4; }
td.left, th.left { text-align: left; }
</style>
</head>
<body>
<h1>US Top 250 Stock Monitor</h1>
<p>Daily view of one year performance, ranked by one year return.</p>
<table>
<thead>
<tr>
{% for col in columns %}
<th class="{{ 'left' if col in left_cols else '' }}">{{ col }}</th>
{% endfor %}
</tr>
</thead>
<tbody>
{% for row in rows %}
<tr>
{% for col in columns %}
<td class="{{ 'left' if col in left_cols else '' }}">{{ row[col] }}</td>
{% endfor %}
</tr>
{% endfor %}
</tbody>
</table>
</body>
</html>
"""

def generate_outputs(df: pd.DataFrame):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = REPORTS_DIR / "daily_stock_monitor.csv"
    output_html = REPORTS_DIR / "daily_stock_monitor.html"
    alerts_csv = REPORTS_DIR / "alerts.csv"

    df.to_csv(output_csv, index=False)

    left_cols = {"symbol", "companyName", "exchange", "sector", "industry", "as_of_date"}
    template = Template(HTML_TEMPLATE)

    html = template.render(
        columns=list(df.columns),
        rows=df.fillna("").to_dict(orient="records"),
        left_cols=left_cols
    )

    output_html.write_text(html, encoding="utf-8")

    if "return_1d_pct" in df.columns and "drawdown_from_1y_high_pct" in df.columns:
        alerts = df[
            (pd.to_numeric(df["return_1d_pct"], errors="coerce").abs() >= ALERT_DAILY_MOVE_PCT) |
            (pd.to_numeric(df["drawdown_from_1y_high_pct"], errors="coerce") <= ALERT_DRAWDOWN_PCT)
        ].copy()
    else:
        alerts = pd.DataFrame()

    alerts.to_csv(alerts_csv, index=False)

    return {
        "csv": str(output_csv),
        "html": str(output_html),
        "alerts": str(alerts_csv),
        "alert_count": len(alerts)
    }
