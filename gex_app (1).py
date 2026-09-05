"""
gex_app.py

A small local website for GEX (gamma exposure) analysis using free data:
    - GEX Terminal: options chain -> gamma exposure by strike, via yfinance
    - Dark Pool Ratio: daily short-sale volume as % of total volume, via
      FINRA's free public Daily Short Sale Volume Files

Both are research proxies, not observed dealer positioning or true
off-exchange volume. See in-app notes for details.

Install:
    pip install flask yfinance scipy pandas numpy plotly requests

Run:
    python gex_app.py

Then open http://127.0.0.1:5000 in Chrome (it should also auto-open for you).
"""

import os
import threading
import webbrowser
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from flask import Flask, request
from scipy.stats import norm

app = Flask(__name__)


def render(template: str, **tokens) -> str:
    """Plain __TOKEN__ substitution — deliberately not Jinja/str.format, since
    the injected chart HTML from Plotly contains its own literal curly braces."""
    html = template
    for key, value in tokens.items():
        html = html.replace(f"__{key.upper()}__", str(value))
    return html


CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.045
FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"

# ---------------------------------------------------------------------------
# Shared page shell
# ---------------------------------------------------------------------------

BASE_CSS = """
  :root {
    --bg: #0a0e14; --panel: #10151f; --border: #1f2733; --text: #e8ecf1;
    --muted: #7d8899; --green: #2ecc71; --red: #e74c3c; --blue: #3498db;
    --orange: #f39c12; --purple: #9b59b6;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", sans-serif;
         margin: 0; }
  .topnav { display:flex; align-items:center; gap: 24px; padding: 14px 24px; border-bottom: 1px solid var(--border);
            background: var(--panel); }
  .topnav .brand { font-weight: 700; font-size: 16px; }
  .topnav a { color: var(--muted); text-decoration: none; font-size: 14px; font-weight: 600;
              padding: 6px 12px; border-radius: 6px; }
  .topnav a.active { color: var(--text); background: #1a2230; }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
  form.controls { display:flex; gap:10px; align-items:center; margin-bottom: 20px; flex-wrap: wrap; }
  input, select { background:#151b26; color:var(--text); border:1px solid var(--border); border-radius:6px;
                  padding:9px 12px; font-size:14px; }
  button { background: var(--green); border:none; color:#08130d; font-weight:700; border-radius:6px;
           padding:9px 18px; cursor:pointer; font-size:14px; }
  button:hover { filter: brightness(1.1); }
  .error { background:#2a1414; border:1px solid #5c2626; color:#ff8c8c; padding:12px 16px; border-radius:8px;
           margin-bottom: 16px; }
  .grid { display:grid; grid-template-columns: 1fr 280px; gap: 20px; align-items:start; }
  .card { background: var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .stat-card .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .stat-card .value { font-size:20px; font-weight:700; margin-top:4px; }
  .sidebar { display:flex; flex-direction:column; gap:12px; }
  table { border-collapse: collapse; width:100%; margin-top: 10px; }
  td, th { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--border); font-size:13px; }
  th:first-child, td:first-child { text-align: left; color: var(--muted); }
  .note { color: var(--muted); font-size:12px; line-height:1.6; margin-top: 24px; }
  h1 { font-size: 20px; margin: 0 0 16px; }
  h3 { font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing:.04em; margin: 20px 0 8px; }
"""

NAV_TEMPLATE = """
<div class="topnav">
  <div class="brand">GEX Terminal</div>
  <a href="/" class="__GEX_ACTIVE__">GEX Terminal</a>
  <a href="/darkpool" class="__DP_ACTIVE__">Dark Pool Ratio</a>
</div>
"""


def build_nav(active: str) -> str:
    return render(NAV_TEMPLATE, gex_active="active" if active == "gex" else "",
                   dp_active="active" if active == "darkpool" else "")

GEX_PAGE = """
<!doctype html><html><head><title>GEX Terminal</title><style>__CSS__</style></head><body>
__NAV__
<div class="wrap">
  <form class="controls" method="get" action="/">
    <input name="ticker" placeholder="Ticker (e.g. SPY)" value="__TICKER__" required>
    <input name="expiry" placeholder="Expiry YYYY-MM-DD (optional)" value="__EXPIRY__">
    <button type="submit">Load</button>
  </form>

  __ERROR_BLOCK__

  __BODY__

  <div class="note">
    Delayed data via yfinance (Yahoo Finance). GEX is a model-based proxy using open interest
    and Black-Scholes gamma with the standard dealer-short-gamma convention — not observed
    dealer positioning.
  </div>
</div>
</body></html>
"""

DP_PAGE = """
<!doctype html><html><head><title>Dark Pool Ratio</title><style>__CSS__</style></head><body>
__NAV__
<div class="wrap">
  <form class="controls" method="get" action="/darkpool">
    <input name="ticker" placeholder="Ticker (e.g. SPY)" value="__TICKER__" required>
    <input name="days" type="number" min="5" max="90" value="__DAYS__" style="width:90px" title="Trading days">
    <button type="submit">Load</button>
  </form>

  __ERROR_BLOCK__

  __BODY__

  <div class="note">
    Source: FINRA Daily Short Sale Volume Files (free, public, ~1 trading-day lag).
    This is short-sale volume as a % of total consolidated volume — a common free proxy for
    off-exchange / "dark pool" style activity, not a direct feed of actual dark pool prints
    (real-time ATS-level dark pool data is not freely available).
  </div>
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# GEX Terminal logic
# ---------------------------------------------------------------------------

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


def build_gex_chart_html(ticker_symbol, expiry, spot, by_strike, flip):
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
        template="plotly_dark", height=480, margin=dict(t=50, b=40),
        paper_bgcolor="#10151f", plot_bgcolor="#10151f",
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.route("/")
def index():
    ticker = request.args.get("ticker", "").strip().upper()
    expiry = request.args.get("expiry", "").strip()
    nav = build_nav("gex")

    if not ticker:
        return render(GEX_PAGE, css=BASE_CSS, nav=nav, ticker="", expiry="",
                                       error_block="", body="")

    try:
        spot, resolved_expiry, calls, puts = fetch_chain(ticker, expiry or None)
        gex_df = compute_gex(spot, resolved_expiry, calls, puts)
        if gex_df.empty:
            raise RuntimeError("No strikes had usable open interest / implied volatility data.")

        by_strike = gex_df.groupby("strike", as_index=False)["gex"].sum().sort_values("strike")
        by_strike["cumulative_gex"] = by_strike["gex"].cumsum()
        flip = find_zero_gamma_flip(by_strike)
        total_gex = by_strike["gex"].sum()

        call_wall_row = by_strike.loc[by_strike["gex"].idxmax()]
        put_wall_row = by_strike.loc[by_strike["gex"].idxmin()]

        top_walls = by_strike.reindex(
            by_strike["gex"].abs().sort_values(ascending=False).index
        ).head(8).to_dict("records")

        chart_html = build_gex_chart_html(ticker, resolved_expiry, spot, by_strike, flip)
        regime = "POSITIVE" if total_gex >= 0 else "NEGATIVE"
        regime_color = "var(--green)" if total_gex >= 0 else "var(--red)"

        walls_rows = "".join(
            f"<tr><td>{r['strike']}</td><td>{r['gex']:,.0f}</td></tr>" for r in top_walls
        )

        body = f"""
        <div class="grid">
          <div class="card">
            {chart_html}
            <h3>Top gamma walls</h3>
            <table><tr><th>Strike</th><th>GEX</th></tr>{walls_rows}</table>
          </div>
          <div class="sidebar">
            <div class="card stat-card">
              <div class="label">Regime</div>
              <div class="value" style="color:{regime_color}">{regime}</div>
            </div>
            <div class="card stat-card">
              <div class="label">Spot</div>
              <div class="value">{spot:.2f}</div>
            </div>
            <div class="card stat-card">
              <div class="label">Zero-gamma flip</div>
              <div class="value" style="color:var(--orange)">{f"{flip:.2f}" if flip else "n/a"}</div>
            </div>
            <div class="card stat-card">
              <div class="label">Call wall</div>
              <div class="value" style="color:var(--green)">{call_wall_row['strike']:.2f}</div>
              <div class="label">GEX {call_wall_row['gex']:,.0f}</div>
            </div>
            <div class="card stat-card">
              <div class="label">Put wall</div>
              <div class="value" style="color:var(--red)">{put_wall_row['strike']:.2f}</div>
              <div class="label">GEX {put_wall_row['gex']:,.0f}</div>
            </div>
            <div class="card stat-card">
              <div class="label">Net GEX</div>
              <div class="value">{total_gex:,.0f}</div>
            </div>
            <div class="card stat-card">
              <div class="label">Expiry</div>
              <div class="value">{resolved_expiry}</div>
            </div>
          </div>
        </div>
        """
        return render(GEX_PAGE, css=BASE_CSS, nav=nav, ticker=ticker, expiry=expiry,
                                       error_block="", body=body)

    except RuntimeError as e:
        error_block = f'<div class="error">{e}</div>'
        return render(GEX_PAGE, css=BASE_CSS, nav=nav, ticker=ticker, expiry=expiry,
                                       error_block=error_block, body="")


# ---------------------------------------------------------------------------
# Dark Pool Ratio logic (FINRA daily short-sale volume, free & public)
# ---------------------------------------------------------------------------

def fetch_finra_day_text(date_str: str):
    url = FINRA_URL.format(date=date_str)
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    text = resp.text
    if not text or "Access Denied" in text[:300] or "<html" in text[:100].lower():
        return None
    return text


def parse_finra_day_text(text: str, symbol: str):
    for line in text.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4 or parts[1].upper() != symbol:
            continue
        try:
            short_vol = int(parts[2])
            total_vol = int(parts[3])
        except ValueError:
            return None
        if total_vol <= 0:
            return None
        return {"date": parts[0], "short_volume": short_vol, "total_volume": total_vol,
                "ratio": short_vol / total_vol}
    return None


def get_dark_pool_series(symbol: str, target_days: int, max_lookback: int = 200):
    symbol = symbol.upper()
    results = []
    d = datetime.now() - timedelta(days=1)  # today's file usually isn't posted yet
    checked = 0
    while len(results) < target_days and checked < max_lookback:
        checked += 1
        if d.weekday() < 5:  # skip weekends
            text = fetch_finra_day_text(d.strftime("%Y%m%d"))
            if text:
                row = parse_finra_day_text(text, symbol)
                if row:
                    results.append(row)
        d -= timedelta(days=1)
    results.sort(key=lambda r: r["date"])
    return results


def build_darkpool_chart_html(symbol, rows):
    dates = [r["date"] for r in rows]
    ratios = [r["ratio"] * 100 for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=ratios, mode="lines+markers", name="Short volume %",
                              line=dict(color="#9b59b6")))
    fig.add_hline(y=50, line_dash="dash", line_color="#555",
                  annotation_text="50% — heavy short/dark-style flow", annotation_position="top left")
    fig.update_layout(
        title=f"{symbol}: Short Volume as % of Total (FINRA, daily)",
        xaxis_title="Date", yaxis_title="% of total volume",
        template="plotly_dark", height=480, margin=dict(t=50, b=40),
        paper_bgcolor="#10151f", plot_bgcolor="#10151f",
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.route("/darkpool")
def darkpool():
    ticker = request.args.get("ticker", "").strip().upper()
    days = request.args.get("days", "30").strip()
    nav = build_nav("darkpool")

    try:
        days_n = max(5, min(90, int(days)))
    except ValueError:
        days_n = 30

    if not ticker:
        return render(DP_PAGE, css=BASE_CSS, nav=nav, ticker="", days=days_n,
                                       error_block="", body="")

    try:
        rows = get_dark_pool_series(ticker, days_n)
        if not rows:
            raise RuntimeError(
                f"No FINRA short-sale volume data found for '{ticker}' in the lookback window. "
                "Check the ticker, or FINRA's feed may be temporarily unavailable."
            )

        avg_ratio = sum(r["ratio"] for r in rows) / len(rows)
        latest_ratio = rows[-1]["ratio"]
        trend = "up" if latest_ratio > avg_ratio else "down"
        trend_color = "var(--orange)" if trend == "up" else "var(--blue)"

        chart_html = build_darkpool_chart_html(ticker, rows)

        recent_rows = "".join(
            f"<tr><td>{r['date']}</td><td>{r['ratio']*100:.1f}%</td>"
            f"<td>{r['short_volume']:,}</td><td>{r['total_volume']:,}</td></tr>"
            for r in reversed(rows[-15:])
        )

        body = f"""
        <div class="grid">
          <div class="card">
            {chart_html}
            <h3>Recent days</h3>
            <table><tr><th>Date</th><th>Short %</th><th>Short Vol</th><th>Total Vol</th></tr>{recent_rows}</table>
          </div>
          <div class="sidebar">
            <div class="card stat-card">
              <div class="label">Latest</div>
              <div class="value" style="color:{trend_color}">{latest_ratio*100:.1f}%</div>
            </div>
            <div class="card stat-card">
              <div class="label">{days_n}-day average</div>
              <div class="value">{avg_ratio*100:.1f}%</div>
            </div>
            <div class="card stat-card">
              <div class="label">Trend vs average</div>
              <div class="value" style="color:{trend_color}">{trend.upper()}</div>
            </div>
            <div class="card stat-card">
              <div class="label">Days loaded</div>
              <div class="value">{len(rows)}</div>
            </div>
          </div>
        </div>
        """
        return render(DP_PAGE, css=BASE_CSS, nav=nav, ticker=ticker, days=days_n,
                                       error_block="", body=body)

    except RuntimeError as e:
        error_block = f'<div class="error">{e}</div>'
        return render(DP_PAGE, css=BASE_CSS, nav=nav, ticker=ticker, days=days_n,
                                       error_block=error_block, body="")


def open_browser(port):
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if not os.environ.get("PORT"):
        threading.Timer(1.0, lambda: open_browser(port)).start()
    app.run(debug=False, host="0.0.0.0", port=port)
