"""Fundamental value screening for the Oslo equity universe. Research tool, not advice."""
from __future__ import annotations
import json, math, time
from datetime import datetime, timezone
from pathlib import Path
import yfinance as yf
from scanner import discover_oslo_universe

OUT = Path("value")
OUT.mkdir(exist_ok=True)

def number(x):
    try:
        x=float(x)
        return round(x,4) if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def rank(value, low, high, invert=False):
    if value is None: return None
    v=max(0,min(1,(value-low)/(high-low)))
    return round(100*(1-v if invert else v),1)

def dividend_fraction(raw):
    """Yahoo dividendYield is commonly percentage points (e.g. 7.27 for 7.27%)."""
    value=number(raw)
    if value is None or value < 0: return None
    return round(value / 100, 6) if value <= 100 else None

def score(info):
    sector=str(info.get("sector") or "")
    industry=str(info.get("industry") or "")
    financial=sector=="Financial Services"
    metrics={
      "pe":number(info.get("trailingPE")),
      "forward_pe":number(info.get("forwardPE")),
      "pb":number(info.get("priceToBook")),
      "ev_ebitda":number(info.get("enterpriseToEbitda")),
      "fcf_yield":None,
      "roe":number(info.get("returnOnEquity")),
      "operating_margin":number(info.get("operatingMargins")),
      "debt_equity":number(info.get("debtToEquity")),
      "current_ratio":number(info.get("currentRatio")),
      "dividend_yield":dividend_fraction(info.get("dividendYield")),
      "market_cap":number(info.get("marketCap")),
      "price":number(info.get("currentPrice") or info.get("regularMarketPrice")),
      "currency":info.get("currency"),
    }
    cap=metrics["market_cap"]; fcf=number(info.get("freeCashflow"))
    if cap and cap>0 and fcf is not None: metrics["fcf_yield"]=round(fcf/cap,4)
    # Implausible valuation ratios often indicate stale prices or broken vendor fields.
    for key in ("pe", "forward_pe", "pb", "ev_ebitda"):
        v=metrics[key]
        if v is not None and (v <= 0 or v > 500 or (key in ("pe","ev_ebitda") and v < 1)):
            metrics[key]=None
    parts={}
    if financial:
        parts["valuation"]=[rank(metrics["pe"],5,25,True),rank(metrics["pb"],0.5,3,True)]
        parts["quality"]=[rank(metrics["roe"],0,0.25)]
    else:
        parts["valuation"]=[rank(metrics["pe"],5,30,True),rank(metrics["ev_ebitda"],4,20,True),rank(metrics["fcf_yield"],0,0.12)]
        parts["quality"]=[rank(metrics["roe"],0,0.3),rank(metrics["operating_margin"],0,0.3)]
        parts["balance"]=[rank(metrics["debt_equity"],0,200,True),rank(metrics["current_ratio"],0.7,3)]
    parts["income"]=[rank(metrics["dividend_yield"],0,0.08)]
    weights={"valuation":0.45,"quality":0.30,"balance":0.20,"income":0.05} if not financial else {"valuation":0.60,"quality":0.35,"income":0.05}
    available={k:sum(v)/len(v) for k,vals in parts.items() if (v:=[x for x in vals if x is not None])}
    coverage=round(sum(weights[k] for k in available),2)
    final=round(sum(available[k]*weights[k] for k in available)/coverage,1) if coverage>=0.55 else None
    return metrics,final,coverage,("financial" if financial else "operating"),sector,industry

def main():
    universe=discover_oslo_universe()
    rows=[]
    for i,r in enumerate(universe.to_dict("records"),1):
        ticker=r["ticker"]
        try:
            info=yf.Ticker(ticker).info
            metrics,s,coverage,model,sector,industry=score(info)
            rows.append({"ticker":ticker,"company":r["company"],"market":r["market"],"sector":sector,"industry":industry,
                         "model":model,"score":s,"coverage":coverage,**metrics,"error":None})
        except Exception as exc:
            rows.append({"ticker":ticker,"company":r["company"],"market":r["market"],"score":None,"coverage":0,"error":str(exc)[:160]})
        if i%25==0: print(f"Processed {i}/{len(universe)}",flush=True)
        time.sleep(0.25)
    payload={"generated_utc":datetime.now(timezone.utc).isoformat(),"universe_count":len(universe),
             "screened_count":len(rows),"scored_count":sum(x["score"] is not None for x in rows),
             "method":"heuristic relative-value score; NOT intrinsic value","rows":rows}
    (OUT/"data.json").write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")
    print(f"Saved {len(rows)} securities to value/data.json")

if __name__=="__main__": main()
