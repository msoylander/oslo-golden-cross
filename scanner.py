from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import html
import os
import time

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

RESULTS_DIR = Path("results")
LOOKBACK = "2y"
FAST = 50
SLOW = 200


def discover_oslo_universe() -> pd.DataFrame:
    q = EquityQuery("and", [
        EquityQuery("eq", ["region", "no"]),
        EquityQuery("eq", ["exchange", "OSL"]),
    ])
    response = yf.screen(q, size=250, sortField="ticker", sortAsc=True)
    quotes = response.get("quotes", [])
    rows = []
    for x in quotes:
        symbol = x.get("symbol")
        quote_type = str(x.get("quoteType") or "").upper()
        name = x.get("shortName") or x.get("longName") or symbol or ""

        if not symbol:
            continue

        # Yahoo's OSL screener also returns bonds/professional-market
        # instruments. Keep ordinary equity symbols and reject obvious
        # non-share instruments before downloading price history.
        upper = symbol.upper()
        if "-PRO" in upper or upper.endswith("O.OL"):
            continue
        if quote_type and quote_type != "EQUITY":
            continue

        rows.append({
            "ticker": symbol,
            "company": name,
            "market_cap": x.get("marketCap"),
        })
    if not rows:
        raise RuntimeError("Yahoo screener returned no Oslo shares after filtering")
    return pd.DataFrame(rows).drop_duplicates("ticker").sort_values("ticker")


def analyse(ticker: str, company: str) -> dict:
    df = yf.download(ticker, period=LOOKBACK, interval="1d", auto_adjust=True,
                     progress=False, threads=False)
    if df.empty:
        raise ValueError("no Yahoo Finance price data")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < SLOW + 2:
        raise ValueError(f"only {len(close)} valid sessions")

    s50 = close.rolling(FAST).mean()
    s200 = close.rolling(SLOW).mean()
    latest50, latest200 = float(s50.iloc[-1]), float(s200.iloc[-1])
    prev50, prev200 = float(s50.iloc[-2]), float(s200.iloc[-2])
    spread = (latest50 / latest200 - 1) * 100

    new_cross = prev50 <= prev200 and latest50 > latest200
    death_cross = prev50 >= prev200 and latest50 < latest200

    # "Approaching" means SMA50 is below SMA200 but within 2%.
    if new_cross:
        status = "NEW GOLDEN CROSS"
    elif death_cross:
        status = "NEW DEATH CROSS"
    elif 0 < spread <= 2:
        status = "ABOVE / CLOSE"
    elif -2 <= spread <= 0:
        status = "APPROACHING"
    elif spread > 0:
        status = "ABOVE"
    else:
        status = "BELOW"

    return {
        "company": company,
        "ticker": ticker,
        "date": close.index[-1].strftime("%Y-%m-%d"),
        "close": round(float(close.iloc[-1]), 4),
        "sma50": round(latest50, 4),
        "sma200": round(latest200, 4),
        "distance_pct": round(spread, 3),
        "status": status,
    }


def make_dashboard(df: pd.DataFrame, errors: list[dict], universe_count: int) -> str:
    new = df[df.status == "NEW GOLDEN CROSS"]
    approaching = df[(df.distance_pct <= 0) & (df.distance_pct >= -2)].sort_values("distance_pct", ascending=False)
    rows = []
    for _, r in df.sort_values(["status", "distance_pct"], ascending=[True, False]).iterrows():
        cls = "gold" if r.status == "NEW GOLDEN CROSS" else ("near" if r.status == "APPROACHING" else "")
        rows.append(f"""<tr class="{cls}"><td>{html.escape(str(r.company))}</td><td>{html.escape(str(r.ticker))}</td>
<td>{r.close:.2f}</td><td>{r.sma50:.2f}</td><td>{r.sma200:.2f}</td>
<td data-order="{r.distance_pct}">{r.distance_pct:+.2f}%</td><td>{html.escape(str(r.status))}</td></tr>""")

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Oslo Golden Cross Dashboard</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:28px;background:#f6f7f9;color:#17202a}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}} .card{{background:white;padding:16px 22px;border-radius:12px;box-shadow:0 1px 4px #bbb}}
.big{{font-size:28px;font-weight:700}} table{{border-collapse:collapse;width:100%;background:white}} th,td{{padding:9px 11px;border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{{text-align:left}}
th{{position:sticky;top:0;background:#17202a;color:white;cursor:pointer}} .gold{{background:#fff3b0;font-weight:700}} .near{{background:#e8f4ff}}
input{{padding:10px;width:min(420px,90%);margin:10px 0 16px;border:1px solid #aaa;border-radius:8px}}
</style></head><body>
<h1>Oslo Golden Cross Dashboard</h1>
<p>Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} from Yahoo Finance daily price data.</p>
<div class="cards"><div class="card"><div class="big">{universe_count}</div>Oslo shares discovered</div>
<div class="card"><div class="big">{len(df)}</div>Successfully analysed</div>
<div class="card"><div class="big">{len(new)}</div>New Golden Crosses</div>
<div class="card"><div class="big">{len(approaching)}</div>Within 2% below crossover</div>
<div class="card"><div class="big">{len(errors)}</div>Data errors</div></div>
<input id="q" placeholder="Search company, ticker or status…" onkeyup="filterTable()">
<table id="stocks"><thead><tr><th>Company</th><th>Ticker</th><th>Price</th><th>SMA50</th><th>SMA200</th><th>50 vs 200</th><th>Status</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>
function filterTable(){{let q=document.getElementById('q').value.toLowerCase();document.querySelectorAll('#stocks tbody tr').forEach(r=>r.style.display=r.innerText.toLowerCase().includes(q)?'':'none')}}
document.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{{let tb=th.closest('table').tBodies[0],rs=[...tb.rows],asc=th.dataset.asc!=='1';rs.sort((a,b)=>{{let x=a.cells[i].dataset.order??a.cells[i].innerText,y=b.cells[i].dataset.order??b.cells[i].innerText;let nx=parseFloat(x),ny=parseFloat(y);return (!isNaN(nx)&&!isNaN(ny)?nx-ny:x.localeCompare(y))*(asc?1:-1)}});rs.forEach(r=>tb.appendChild(r));th.dataset.asc=asc?'1':'0'}})
</script></body></html>"""


def write_summary(df: pd.DataFrame, universe_count: int, errors: list[dict]) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    new = df[df.status == "NEW GOLDEN CROSS"]
    near = df[(df.distance_pct <= 0) & (df.distance_pct >= -2)].sort_values("distance_pct", ascending=False).head(15)
    lines = [
        "# Oslo Golden Cross Dashboard",
        f"**Universe discovered:** {universe_count} · **Analysed:** {len(df)} · **Errors:** {len(errors)}",
        f"## New Golden Crosses ({len(new)})",
    ]
    if new.empty:
        lines.append("No new Golden Cross today.")
    else:
        lines += ["| Company | Ticker | Price | SMA50 | SMA200 | Distance |", "|---|---|---:|---:|---:|---:|"]
        for _, r in new.iterrows():
            lines.append(f"| {r.company} | {r.ticker} | {r.close:.2f} | {r.sma50:.2f} | {r.sma200:.2f} | {r.distance_pct:+.2f}% |")
    lines += ["", "## Closest below Golden Cross", "| Company | Ticker | Distance |", "|---|---|---:|"]
    for _, r in near.iterrows():
        lines.append(f"| {r.company} | {r.ticker} | {r.distance_pct:+.2f}% |")
    lines += ["", "The downloadable **oslo-golden-cross-dashboard** artifact contains the full searchable dashboard and CSV."]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    universe = discover_oslo_universe()
    print(f"Yahoo screener discovered {len(universe)} OSL equities.")
    results, errors = [], []
    for i, r in enumerate(universe.itertuples(index=False), 1):
        try:
            result = analyse(r.ticker, r.company)
            results.append(result)
            if result["status"] == "NEW GOLDEN CROSS":
                print(f"GOLDEN CROSS: {r.ticker} {r.company} ({result['date']})")
        except Exception as exc:
            errors.append({"ticker": r.ticker, "company": r.company, "error": str(exc)})
            print(f"WARN {r.ticker}: {exc}")
        time.sleep(0.03)

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No equities could be analysed")
    df = df.sort_values("distance_pct", ascending=False)
    df.to_csv(RESULTS_DIR / "oslo_universe.csv", index=False)
    pd.DataFrame(errors).to_csv(RESULTS_DIR / "errors.csv", index=False)
    (RESULTS_DIR / "dashboard.html").write_text(make_dashboard(df, errors, len(universe)), encoding="utf-8")
    write_summary(df, len(universe), errors)
    print(f"Successfully analysed {len(df)}/{len(universe)} equities.")
    print(f"New Golden Crosses: {(df.status == 'NEW GOLDEN CROSS').sum()}")


if __name__ == "__main__":
    main()
