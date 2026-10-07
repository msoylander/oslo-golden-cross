from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import time

import pandas as pd
import yfinance as yf

TICKERS_FILE = Path("tickers.txt")
RESULTS_DIR = Path("results")
LOOKBACK = "2y"
FAST = 50
SLOW = 200


def load_tickers() -> list[str]:
    tickers = []
    for line in TICKERS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(line)
    return list(dict.fromkeys(tickers))


def scan(ticker: str) -> dict | None:
    df = yf.download(
        ticker,
        period=LOOKBACK,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError("no Yahoo Finance data")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    if len(close) < SLOW + 2:
        raise ValueError(f"only {len(close)} valid sessions")

    sma50 = close.rolling(FAST).mean()
    sma200 = close.rolling(SLOW).mean()

    # A NEW golden cross occurred on the latest completed Yahoo daily bar:
    # previous bar SMA50 <= SMA200 and latest bar SMA50 > SMA200.
    crossed = (
        sma50.iloc[-2] <= sma200.iloc[-2]
        and sma50.iloc[-1] > sma200.iloc[-1]
    )
    if not crossed:
        return None

    return {
        "ticker": ticker,
        "date": close.index[-1].strftime("%Y-%m-%d"),
        "close": round(float(close.iloc[-1]), 4),
        "sma50": round(float(sma50.iloc[-1]), 4),
        "sma200": round(float(sma200.iloc[-1]), 4),
        "spread_pct": round(float((sma50.iloc[-1] / sma200.iloc[-1] - 1) * 100), 4),
    }


def main() -> None:
    tickers = load_tickers()
    crosses = []
    errors = []

    print(f"Scanning {len(tickers)} Oslo-listed Yahoo tickers...")
    for i, ticker in enumerate(tickers, start=1):
        try:
            result = scan(ticker)
            if result:
                crosses.append(result)
                print(f"GOLDEN CROSS: {ticker} ({result['date']})")
        except Exception as exc:
            errors.append((ticker, str(exc)))
            print(f"WARN {ticker}: {exc}")
        time.sleep(0.05)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = RESULTS_DIR / f"golden_cross_{stamp}.csv"

    fields = ["ticker", "date", "close", "sma50", "sma200", "spread_pct"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(crosses)

    print(f"\nFound {len(crosses)} new golden cross(es).")
    print(f"Saved: {out}")
    if errors:
        print(f"{len(errors)} ticker(s) returned errors; scan continued.")


if __name__ == "__main__":
    main()
