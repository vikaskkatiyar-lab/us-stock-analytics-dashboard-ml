#!/usr/bin/env bash
cd /Users/vikaskatiyar/Documents/us_top250_stock_monitor
mkdir -p logs

if lsof -nP -iTCP:8502 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Dashboard is already running at http://127.0.0.1:8502/"
  exit 0
fi

nohup ./.venv/bin/streamlit run app.py \
  --server.port 8502 \
  --server.address 127.0.0.1 \
  >> logs/streamlit-dashboard.log 2>&1 &

echo "Dashboard started at http://127.0.0.1:8502/"
