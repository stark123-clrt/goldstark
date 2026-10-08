"""Test de l'idee "les formes de bougies se repetent" — recherche des plus proches voisins.

Pour chaque bougie M15 t :
  1. forme = les L dernieres bougies (O,H,L,C) relatives au close de t, divisees par l'ATR14
     -> seule la FORME compte, pas le niveau de prix ni la volatilite
  2. on cherche les K formes les plus ressemblantes dans tout l'historique PASSE
  3. probabilite de hausse = part des voisins dont le prix etait plus haut H bougies apres
Walk-forward annuel : pour tester l'annee Y, la bibliotheque ne contient que ce qui precede Y.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import load_m1

COST = 0.30
LENGTHS = [8, 16, 32]          # nb de bougies dans la forme
HORIZONS = [4, 16]             # 1h et 4h
KS = [50, 200]
YEARS = range(2019, 2026)
KMAX = max(KS)

def m15_bars():
    m1 = load_m1()
    r = (m1.set_index("datetime").resample("15min")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index())
    return r

def backtest(ret, prob, thr, h):
    pnl, last = [], -10**9
    for i in range(len(ret)):
        if i < last + h or np.isnan(ret[i]): continue
        d = 1 if prob[i] >= thr else (-1 if prob[i] <= 1 - thr else 0)
        if d: pnl.append(d * ret[i] - COST); last = i
    a = np.array(pnl)
    if len(a) < 1: return "aucun trade"
    pf = a[a > 0].sum() / max(-a[a < 0].sum(), 1e-9)
    return f"n={len(a):5d} WR={(a>0).mean():.1%} PF={pf:4.2f} PnL={a.sum():+7.0f}$"

if __name__ == "__main__":
    t0 = time.time()
    bars = m15_bars()
    o, h, l, c = (bars[x].values.astype(np.float32) for x in ["open", "high", "low", "close"])
    dt = bars["datetime"]
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().values.astype(np.float32)
    # bougies consecutives sans trou (pas de week-end au milieu)
    step_ok = np.r_[False, np.diff(dt.values) == np.timedelta64(15, "m")]
    run = np.zeros(len(c), int)
    for i in range(1, len(c)): run[i] = run[i - 1] + 1 if step_ok[i] else 0
    futs = {}
    for H in HORIZONS:
        f = np.full(len(c), np.nan, np.float32); f[:-H] = c[H:] - c[:-H]
        ok = np.zeros(len(c), bool); ok[:-H] = run[H:] - run[:-H] == H
        f[~ok] = np.nan; futs[H] = f
    print(f"{len(c):,} bougies M15 ({time.time()-t0:.0f}s)\n", flush=True)

    for L in LENGTHS:
        valid = (run >= L) & (atr > 0) & ~np.isnan(atr)
        idx = np.where(valid)[0]
        win = np.stack([np.stack([o[idx - k], h[idx - k], l[idx - k], c[idx - k]], 1) for k in range(L - 1, -1, -1)], 1)
        X = ((win - c[idx, None, None]) / atr[idx, None, None]).reshape(len(idx), -1).astype(np.float32)
        res = {(H, K): [] for H in HORIZONS for K in KS}
        for Y in YEARS:
            ts = pd.Timestamp(f"{Y}-01-01", tz="UTC")
            t_idx = dt.iloc[idx].values
            # bibliotheque : formes passees dont le resultat (t+Hmax) est connu avant le debut du test
            lib = t_idx < (ts - pd.Timedelta(days=1)).to_datetime64()
            test = (t_idx >= ts.to_datetime64()) & (t_idx < (ts + pd.DateOffset(years=1)).to_datetime64())
            nn = NearestNeighbors(n_neighbors=KMAX, algorithm="brute", n_jobs=-1).fit(X[lib])
            _, nb = nn.kneighbors(X[test])
            lib_rows, test_rows = idx[lib], idx[test]
            for H in HORIZONS:
                up = futs[H][lib_rows] > 0
                known = ~np.isnan(futs[H][lib_rows])
                for K in KS:
                    nk = nb[:, :K]
                    prob = (up[nk] & known[nk]).sum(1) / np.maximum(known[nk].sum(1), 1)
                    res[(H, K)].append((Y, futs[H][test_rows], prob))
            print(f"  L={L} {Y} fait ({time.time()-t0:.0f}s)", flush=True)

        for H in HORIZONS:
            for K in KS:
                print(f"\n=== Forme {L} bougies | horizon {H} bougies ({H*15} min) | {K} voisins ===")
                allr, allp = [], []
                for Y, r, p in res[(H, K)]:
                    m = ~np.isnan(r); allr.append(r[m]); allp.append(p[m])
                    print(f"  {Y}: AUC={roc_auc_score(r[m] > 0, p[m]):.4f} | p>=0.60: {backtest(r, p, 0.60, H)} | p>=0.65: {backtest(r, p, 0.65, H)}")
                R, P = np.concatenate(allr), np.concatenate(allp)
                print(f"  CALIBRATION 2019-2025 (proba affichee -> % hausse reel) base={np.mean(R>0):.1%}")
                for lo, hi in [(0, .35), (.35, .45), (.45, .55), (.55, .65), (.65, 1.01)]:
                    m = (P >= lo) & (P < hi)
                    if m.sum(): print(f"     p in [{lo:.2f},{min(hi,1):.2f}) : {m.sum():7,d} cas -> {np.mean(R[m]>0):.1%} hausse, gain moyen {R[m].mean():+.2f}$")
        sys.stdout.flush()
    print(f"\nTermine en {(time.time()-t0)/60:.1f} min")
