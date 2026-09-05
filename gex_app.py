"""
gex_app.py

A small local website for computing GEX (gamma exposure) proxies from free
yfinance options data. Run it once, then use it from your browser like a
normal website — enter a ticker, get a chart, change the ticker, repeat.

This is a RESEARCH / EDUCATIONAL PROXY, not observed dealer positioning.

Install:
    pip install flask yfinance scipy pandas numpy plotly

Run:
    python gex_app.py

Then open http://127.0.0.1:5000 in Chrome (it should also auto-open for you).
"""

import os
import threading
import webbrowser
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from flask import Flask, render_template_string, request
from scipy.stats import norm

app = Flask(__name__)

CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.045

PAGE = """
<!doctype html>
<html>
<head>
  <title>GEX Terminal</title>
  <style>
    body { background:#111; color:#eee; font-family: -apple-system, Segoe UI, sans-serif;
           max-width: 1000px; margin: 30px auto; padding: 0 20px; }
    h1 { font-size: 22px; }
    form { display:flex; gap:10px; align-items:center; margin-bottom: 20px; flex-wrap: wrap; }
    input, select { background:#222; color:#eee; border:1px solid #444; border-radius:4px;
                    padding:8px 10px; font-size:14px; }
    button { background:#2ecc71; border:none; color:#111; font-weight:600; border-radius:4px;
             padding:8px 16px; cursor:pointer; font-size:14px; }
    button:hover { background:#27ae60; }
    .stats { display:flex; gap:24px; margin: 16px 0; flex-wrap: wrap; }
    .stat { background:#1c1c1c; border:1px solid #333; border-radius:6px; padding:10px 16px; }
    .stat .label { color:#888; font-size:12px; text-transform:uppercase; }
    .stat .value { font-size:18px; font-weight:600; }
    .error { background:#3a1c1c; border:1px solid #822; color:#f88; padding:12px; border-radius:6px; }
    table { border-collapse: collapse; margin-top: 10px; }
    td, th { padding: 4px 12px; text-align: right; border-bottom: 1px solid #333; }
    .note { color:#777; font-size:12px; margin-top: 30px; }
  </style>
</head>
<body>
  <h1>GEX Terminal (proxy)</h1>
  <form method="get" action="/">
    <input name="ticker" placeholder="Ticker (e.g. SPY)" value="{{ ticker or '' }}" required>
    <input name="expiry" placeholder="Expiry YYYY-MM-DD (optional)" value="{{ expiry or '' }}">
    <button type="submit">Load</button>
  </form>

  {% if error %}
    <div class="error">{{ error }}</div>
  {% endif %}

  {% if result %}
    <div class="stats">
      <div class="stat"><div class="label">Spot</div><div class="value">{{ "%.2f"|format(result.spot) }}</div></div>
      <div class="stat"><div class="label">Expiry</div><div class="value">{{ result.expiry }}</div></div>
      <div class="stat"><div class="label">Total GEX</div><div class="value">{{ "{:,.0f}".format(result.total_gex) }}</div></div>
      <div class="stat"><div class="label">Zero-gamma flip</div><div class="value">{{ "%.2f"|format(result.flip) if result.flip else "n/a" }}</div></div>
    </div>

    {{ result.chart_html | safe }}

    <h3>Top gamma walls</h3>
    <table>
      <tr><th>Strike</th><th>GEX</th></tr>
      {% for row in result.top_walls %}
        <tr><td>{{ row.strike }}</td><td>{{ "{:,.0f}".format(row.gex) }}</td></tr>
      {% endfor %}
    </table>
  {% endif %}

  <div class="note">
    Delayed data via yfinance (Yahoo Finance). GEX is a model-based proxy using open interest
    and Black-Scholes gamma with the standard dealer-short-gamma convention — not observed
    dealer positioning.
  </div>
</body>
</html>
"""


def bs_gamma(spot, strike, t_years, iv, r=RISK_FREE_RATE):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * np.sqrt(t_years))
    return norm.pdf(d1) / (spot * iv * np.sqrt(t_years))


def years_to_expiry(expiry_str):
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(hour=16, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((expiry - now).total_seconds() / 86400, 0) / 365.0


def fetch_chain(ticker_symbol, expiry):
    ticker = yf.Ticker(ticker_symbol)

    spot_hist = ticker.history(period="1d")
    if spot_hist.empty:
        raise RuntimeError(f"Could not fetch spot price for '{ticker_symbol}'. Check the ticker.")
    spot = float(spot_hist["Close"].iloc[-1])

    available = ticker.options
    if not available:
        raise RuntimeError(
            f"No option expiries returned for '{ticker_symbol}'. "
            "Yahoo's feed occasionally returns empty results — try again in a moment."
        )

    if not expiry:
        expiry = available[0]
    elif expiry not in available:
        raise RuntimeError(f"Expiry {expiry} not available. Choices: {', '.join(available)}")

    chain = ticker.option_chain(expiry)
    return spot, expiry, chain.calls, chain.puts


def compute_gex(spot, expiry, calls, puts):
    t = years_to_expiry(expiry)
    rows = []
    for df, sign in [(calls, +1), (puts, -1)]:
        for _, row in df.iterrows():
            oi = row.get("openInterest", 0) or 0
            iv = row.get("impliedVolatility", 0) or 0
            strike = row["strike"]
            if oi <= 0 or iv <= 0:
                continue
            gamma = bs_gamma(spot, strike, t, iv)
            gex = sign * gamma * oi * CONTRACT_MULTIPLIER * (spot ** 2) * 0.01
            rows.append({"strike": strike, "gex": gex})
    return pd.DataFrame(rows)


def find_zero_gamma_flip(by_strike):
    signs = np.sign(by_strike["cumulative_gex"].values)
    changes = np.where(np.diff(signs) != 0)[0]
    if len(changes) == 0:
        return None
    return float(by_strike["strike"].iloc[changes[0]])


def build_chart_html(ticker_symbol, expiry, spot, by_strike, flip):
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in by_strike["gex"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_strike["strike"], y=by_strike["gex"], marker_color=colors,
        hovertemplate="Strike: %{x}<br>GEX: %{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=spot, line_color="#3498db", annotation_text=f"Spot {spot:.2f}", annotation_position="top")
    if flip:
        fig.add_vline(x=flip, line_dash="dash", line_color="#f39c12",
                       annotation_text=f"Flip {flip:.2f}", annotation_position="bottom")
    fig.update_layout(
        title=f"{ticker_symbol} Gamma Exposure — expiry {expiry}",
        xaxis_title="Strike", yaxis_title="GEX",
        template="plotly_dark", height=550, margin=dict(t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.route("/")
def index():
    ticker = request.args.get("ticker", "").strip().upper()
    expiry = request.args.get("expiry", "").strip()

    if not ticker:
        return render_template_string(PAGE, ticker=None, expiry=None, result=None, error=None)

    try:
        spot, resolved_expiry, calls, puts = fetch_chain(ticker, expiry or None)
        gex_df = compute_gex(spot, resolved_expiry, calls, puts)
        if gex_df.empty:
            raise RuntimeError("No strikes had usable open interest / implied volatility data.")

        by_strike = gex_df.groupby("strike", as_index=False)["gex"].sum().sort_values("strike")
        by_strike["cumulative_gex"] = by_strike["gex"].cumsum()
        flip = find_zero_gamma_flip(by_strike)
        total_gex = by_strike["gex"].sum()

        top_walls = by_strike.reindex(
            by_strike["gex"].abs().sort_values(ascending=False).index
        ).head(8).to_dict("records")

        chart_html = build_chart_html(ticker, resolved_expiry, spot, by_strike, flip)

        result = {
            "spot": spot, "expiry": resolved_expiry, "total_gex": total_gex,
            "flip": flip, "top_walls": top_walls, "chart_html": chart_html,
        }
        return render_template_string(PAGE, ticker=ticker, expiry=expiry, result=result, error=None)

    except RuntimeError as e:
        return render_template_string(PAGE, ticker=ticker, expiry=expiry, result=None, error=str(e))


def open_browser(port):
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Only auto-open a browser tab when running locally, not on a hosting platform.
    if not os.environ.get("PORT"):
        threading.Timer(1.0, lambda: open_browser(port)).start()
    app.run(debug=False, host="0.0.0.0", port=port)
