# Oslo Golden Cross

Automated Golden Cross scanner for an initial universe of Oslo-listed shares using Yahoo Finance via `yfinance`.

## Signal definition

A **new Golden Cross** is reported only when:

- previous daily bar: SMA(50) <= SMA(200)
- latest daily bar: SMA(50) > SMA(200)

This intentionally does **not** report every share whose 50-day SMA is already above its 200-day SMA.

## Run locally

```bash
pip install -r requirements.txt
python scanner.py
```

Results are written to `results/golden_cross_YYYY-MM-DD.csv`.

## Automation

GitHub Actions runs the scanner on weekdays after the Oslo market close and can also be started manually with **Run workflow**.

## Ticker universe

`tickers.txt` is the initial Yahoo Finance Oslo universe. Yahoo's Oslo symbols generally use `.OL`. The list should be expanded and periodically maintained as Euronext Oslo listings change.

## Data caveat

Yahoo Finance is convenient for screening but is not an exchange-grade market-data feed. Signals should be verified before making investment decisions.
