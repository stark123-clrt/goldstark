"""Strategies de tendance multi-jours sur XAUUSD (bougies Daily), regles simples et connues.

Pour eviter de "choisir apres coup" le reglage qui a marche, on teste une petite grille de
parametres standards et on regarde si TOUS marchent (plateau) et pas seulement le meilleur.

Execution : signal calcule a la cloture du jour t, position prise a la cloture de t (= ouverture t+1).
Couts : 0.30$ par aller-retour + swap 0.01% par jour de detention (~3.65%/an, prudent).
Comparaison obligatoire avec le buy & hold : l'or a beaucoup monte 2019-2025.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import load_m1

SPREAD = 0.30
SWAP_DAY = 0.0001

def daily_bars():
    m1 = load_m1()
    # journee de trading or : 22:00 UTC -> 22:00 UTC (l'ouverture du dimanche soir rejoint le lundi)
    d = (m1.set_index("datetime").resample("24h", offset="22h")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    cnt = m1.set_index("datetime")["close"].resample("24h", offset="22h").size()
    d = d[cnt.reindex(d.index) > 120]          # on enleve les journees quasi vides (fêtes)
    d.index = d.index + pd.Timedelta(hours=2)  # etiquette = date du jour de trading
    return d

def atr(d, n=20):
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ── Strategies : renvoient une position par jour dans {-1, 0, +1} ─────────────
def donchian(d, n_in, n_out, long_only=False):
    hi_in, lo_in = d.high.rolling(n_in).max().shift(), d.low.rolling(n_in).min().shift()
    hi_out, lo_out = d.high.rolling(n_out).max().shift(), d.low.rolling(n_out).min().shift()
    pos = np.zeros(len(d)); p = 0
    c = d.close.values
    for i in range(len(d)):
        if np.isnan(hi_in.iloc[i]): continue
        if p == 1 and c[i] < lo_out.iloc[i]: p = 0
        elif p == -1 and c[i] > hi_out.iloc[i]: p = 0
        if p == 0:
            if c[i] > hi_in.iloc[i]: p = 1
            elif c[i] < lo_in.iloc[i] and not long_only: p = -1
        pos[i] = p
    return pd.Series(pos, d.index)

def ema_cross(d, fast, slow, long_only=False):
    f, s = d.close.ewm(span=fast).mean(), d.close.ewm(span=slow).mean()
    pos = np.where(f > s, 1, 0 if long_only else -1).astype(float)
    pos[:slow] = 0
    return pd.Series(pos, d.index)

def tsmom(d, n, long_only=False):
    r = d.close / d.close.shift(n) - 1
    pos = np.where(r > 0, 1, 0 if long_only else -1).astype(float)
    pos[:n] = 0
    return pd.Series(pos, d.index)

# ── Backtest ──────────────────────────────────────────────────────────────────
def run(d, pos):
    """Rendement journalier en % du notionnel (taille fixe 1x)."""
    ret = d.close.pct_change().fillna(0)
    held = pos.shift().fillna(0)                       # position decidee hier
    turn = held.diff().abs().fillna(held.abs())       # 1 = entree ou sortie, 2 = retournement
    cost = turn * (SPREAD / 2) / d.close + held.abs() * SWAP_DAY
    return held * ret - cost, held, turn

def stats(r, held, turn):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    n_tr = int((turn > 0).sum() / 2)
    return cagr, sharpe, dd, n_tr, held.abs().mean()

if __name__ == "__main__":
    d = daily_bars()
    print(f"{len(d)} jours {d.index[0].date()} -> {d.index[-1].date()}\n")
    START = "2016-01-01"   # 1 an de chauffe pour les indicateurs
    dd_ = d.loc[START:]
    bh = d.close.pct_change().loc[START:].fillna(0)
    configs = {}
    for lo in (False, True):
        tag = " LONG" if lo else ""
        for n_in, n_out in [(20, 10), (55, 20), (100, 50)]:
            configs[f"Donchian {n_in}/{n_out}{tag}"] = donchian(d, n_in, n_out, lo)
        for f, s in [(20, 100), (50, 200)]:
            configs[f"EMA {f}/{s}{tag}"] = ema_cross(d, f, s, lo)
        for n in (20, 60, 120):
            configs[f"TSMOM {n}j{tag}"] = tsmom(d, n, lo)

    years = sorted(set(dd_.index.year))
    hdr = f"{'strategie':22s} {'CAGR':>6s} {'Sharpe':>6s} {'MaxDD':>6s} {'trades':>6s} {'expo':>5s} | " + " ".join(f"{y:>6d}" for y in years)
    print(hdr); print("-" * len(hdr))
    def line(name, r, held, turn):
        cagr, sh, mdd, n, ex = stats(r, held, turn)
        yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        print(f"{name:22s} {cagr:6.1%} {sh:6.2f} {mdd:6.1%} {n:6d} {ex:5.0%} | " + " ".join(f"{yr.get(y, 0):6.1%}" for y in years))
    line("BUY & HOLD", bh, pd.Series(1.0, bh.index), pd.Series(0.0, bh.index))
    out = {}
    for name, pos in configs.items():
        r, held, turn = run(d, pos)
        out[name] = r.loc[START:]
        line(name, r.loc[START:], held.loc[START:], turn.loc[START:])

    # Partie SHORT seule : est-ce que vendre en tendance baissiere rapporte, ou c'est juste le bull market ?
    print("\nPart SHORT seule (long/short moins long-only) :")
    for name in configs:
        if "LONG" in name: continue
        diff = out[name] - out[name + " LONG"]
        print(f"  {name:20s} contribution short cumulee = {((1 + diff).prod() - 1):+.1%}")
    pd.DataFrame(out).to_parquet(Path(__file__).parent / "trend_daily_returns.parquet")
