"""Telecharge les bougies M1 XAUUSD de Deriv (meme source que l'API live) au format HistData.

Sortie : DAT_DERIV_XAUUSD_M1_<annee>.csv  (separateur ';', heure EST fixe = UTC-5 comme HistData,
pour que build_dataset.load_m1 les lise tel quel).
Usage : python fetch_deriv_m1.py 2025-07-01 [fin]
"""
import asyncio, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import websockets

URL = "wss://api.derivws.com/trading/v1/options/ws/public?app_id=34xSuGBG6QR7xUOC8Kv4P"
OUT = Path(r"C:\Users\chris\Desktop\DATA")

async def fetch(start: int, end: int):
    rows = []
    async with websockets.connect(URL, ping_interval=20) as ws:
        cur = start
        while cur < end:
            req = {"ticks_history": "frxXAUUSD", "style": "candles", "granularity": 60,
                   "start": cur, "end": min(cur + 1000 * 60, end), "count": 1000, "adjust_start_time": 1}
            await ws.send(json.dumps(req))
            r = json.loads(await ws.recv())
            if "error" in r:
                raise RuntimeError(r["error"]["message"])
            c = r.get("candles", [])
            rows += c
            cur = min(cur + 1000 * 60, end)
            last = c[-1]["epoch"] if c else cur
            print(f"  {datetime.fromtimestamp(last, timezone.utc):%Y-%m-%d %H:%M}  total={len(rows):,}", flush=True)
    return rows

if __name__ == "__main__":
    start = int(pd.Timestamp(sys.argv[1], tz="UTC").timestamp())
    end = int(pd.Timestamp(sys.argv[2], tz="UTC").timestamp()) if len(sys.argv) > 2 else int(datetime.now(timezone.utc).timestamp())
    rows = asyncio.run(fetch(start, end))
    df = pd.DataFrame(rows).drop_duplicates("epoch").sort_values("epoch")
    est = pd.to_datetime(df["epoch"], unit="s") - pd.Timedelta(hours=5)
    df["dt"] = est.dt.strftime("%Y%m%d %H%M%S")
    df["vol"] = 0
    for year, part in df.groupby(est.dt.year):
        p = OUT / f"DAT_DERIV_XAUUSD_M1_{year}.csv"
        part[["dt", "open", "high", "low", "close", "vol"]].to_csv(p, sep=";", header=False, index=False)
        print(f"-> {p} ({len(part):,} bougies)")
