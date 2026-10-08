"""Construit le dataset M15 avec les fonctions EXACTES de app.py (train = serve).
Deux variantes d'alignement HTF :
  - leaky    : comme app.py aujourd'hui (HTF indexees par heure d'OUVERTURE)
  - leakfree : HTF indexees par heure de CLOTURE (seulement bougies fermees)
HistData = EST fixe (UTC-5, sans DST) -> converti en UTC comme Deriv.
"""
import ast, sys, time
from pathlib import Path
import numpy as np, pandas as pd

APP = Path(r"C:\Users\chris\Documents\GitHub\goldstark\app.py")
DATA = Path(r"C:\Users\chris\Desktop\DATA")
OUT = Path(__file__).parent

def load_app_funcs():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    keep = list(ast.parse("import numpy as np\nimport pandas as pd\nimport ta").body)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("add_features", "build_features"):
            keep.append(n)
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "MERGE_TOLERANCES":
            keep.append(n)
    ns = {}
    exec(compile(ast.Module(keep, []), "app.py", "exec"), ns)
    return ns

def load_m1():
    dfs = []
    for p in sorted(DATA.glob("DAT_ASCII_XAUUSD_M1_*.csv")):
        d = pd.read_csv(p, sep=";", header=None, names=["dt", "open", "high", "low", "close", "vol"])
        d["datetime"] = pd.to_datetime(d["dt"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
        dfs.append(d[["datetime", "open", "high", "low", "close"]])
    m1 = pd.concat(dfs, ignore_index=True).dropna().sort_values("datetime").drop_duplicates("datetime")
    m1["datetime"] = m1["datetime"].dt.tz_localize("UTC")
    return m1.reset_index(drop=True)

RULES = {"M15": ("15min", pd.Timedelta(minutes=15)), "M30": ("30min", pd.Timedelta(minutes=30)),
         "H1": ("1h", pd.Timedelta(hours=1)), "H4": ("4h", pd.Timedelta(hours=4)),
         "Daily": ("1D", pd.Timedelta(days=1))}

def resample(m1, rule):
    r = (m1.set_index("datetime").resample(rule)
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index())
    r["tickvol"] = 1   # identique au live (Deriv n'a pas de volume)
    return r[["datetime", "open", "high", "low", "close", "tickvol"]]

if __name__ == "__main__":
    t0 = time.time()
    ns = load_app_funcs()
    m1 = load_m1(); print(f"M1: {len(m1):,} {m1.datetime.min()} -> {m1.datetime.max()}", flush=True)
    raw = {tf: resample(m1, r) for tf, (r, _) in RULES.items()}
    del m1
    for variant in sys.argv[1:] or ["leaky", "leakfree"]:
        dfs = {}
        for tf, d in raw.items():
            d = d.copy()
            if variant == "leakfree" and tf != "M15":
                d["datetime"] = d["datetime"] + RULES[tf][1]
            dfs[tf] = d
        base = ns["build_features"](dfs)
        # label : close dans 4 bougies M15 > close actuel (future_bars=4 du metadata)
        c = base["m15_close"].values
        fut = np.full(len(c), np.nan); fut[:-4] = c[4:]
        # pas de label si trou temporel (week-end) dans les 4 bougies suivantes
        dt = base["datetime"].values
        gap = np.full(len(c), True); gap[:-4] = (dt[4:] - dt[:-4]) == np.timedelta64(60, "m")
        base["fut_ret"] = fut - c
        base.loc[~gap, "fut_ret"] = np.nan
        base.to_parquet(OUT / f"ds_{variant}.parquet")
        print(f"{variant}: {len(base):,} lignes, {base.shape[1]} colonnes ({time.time()-t0:.0f}s)", flush=True)
