"""Régimes de marché NON SUPERVISÉS multi-timeframes (HMM) + précision du modèle par régime.

1. Description du marché à chaque clôture M15 avec les bougies FERMÉES de M15/H1/H4/Daily
   (comme l'API) : volatilité (ATR/prix), force de tendance (ADX), pente EMA21, largeur Bollinger.
2. GaussianHMM appris sur HistData 2015-2025 SANS aucune étiquette de hausse/baisse.
3. Sur les données Deriv 2026 (jamais vues par le HMM) : précision des modèles régime par régime.
   - ancien modèle (22/02/2026) : période 22/02 -> 09/10/2026, hors échantillon
   - nouveau modèle (11/10/2026) : période 15/08 -> 09/10/2026, hors échantillon
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np, pandas as pd, joblib, json
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import load_app_funcs, load_m1, RULES

REPO = Path(__file__).parent.parent
K = int(os.environ.get("GS_REGIMES", 5))
COST = 0.30
TFS = ["m15", "h1", "h4", "daily"]

def bars(m1, rule):
    r = (m1.set_index("datetime").resample(rule)
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index())
    r["tickvol"] = 1
    return r

def base_closed(ns, m1):
    raw = {tf: bars(m1, r) for tf, (r, _) in RULES.items()}
    for tf in ["M30", "H1", "H4", "Daily"]:
        raw[tf]["datetime"] = raw[tf]["datetime"] + RULES[tf][1]     # bougies fermées seulement
    return ns["build_features"](raw)

def regime_features(b):
    f = {}
    for p in TFS:
        f[f"{p}_vol"] = np.log(b[f"{p}_atr14"] / b[f"{p}_close"])
        f[f"{p}_adx"] = b[f"{p}_adx"]
        f[f"{p}_slope"] = b[f"{p}_ema21_slope"]
        f[f"{p}_bbw"] = np.log(b[f"{p}_bb_width_20_2"].clip(lower=1e-6))
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)

def describe(F, states):
    rows = []
    for s in range(K):
        m = states == s
        d = F[m].mean()
        trend = d["h4_slope"]
        rows.append(dict(regime=s, part=f"{m.mean():.0%}",
                         vol_m15=f"{np.exp(d['m15_vol'])*100:.3f}%", vol_daily=f"{np.exp(d['daily_vol'])*100:.2f}%",
                         adx_h1=round(d["h1_adx"], 1), adx_h4=round(d["h4_adx"], 1),
                         tendance_h4=("hausse" if trend > 0.05 else "baisse" if trend < -0.05 else "plate")))
    return pd.DataFrame(rows).set_index("regime")

def predict(model_dir, base, rows):
    import tensorflow as tf
    meta = json.load(open(model_dir / "metadata_xauusd.json"))
    feats, seq = meta["feature_cols"], meta["seq_length"]
    model = tf.keras.models.load_model(model_dir / "model_xauusd_best.keras")
    sc = joblib.load(model_dir / "scaler_xauusd.joblib")
    X = np.nan_to_num(base[feats].values.astype(np.float64), nan=0, posinf=0, neginf=0)
    X = np.nan_to_num(sc.transform(X), nan=0, posinf=0, neginf=0).astype(np.float32)
    offs = np.arange(-seq + 1, 1)
    out = [model.predict(X[rows[s:s + 2048][:, None] + offs], verbose=0)["direction"].ravel()
           for s in range(0, len(rows), 2048)]
    return np.concatenate(out)

def future_ret(m1, times):
    c = m1.set_index("datetime")["close"]; idx = c.index
    def at(t):
        q = pd.DatetimeIndex(t) - pd.Timedelta(minutes=1)
        pos = idx.searchsorted(q, side="right") - 1; pc = np.maximum(pos, 0)
        ok = (pos >= 0) & ((q - idx[pc]) <= pd.Timedelta(minutes=3))
        return np.where(ok, c.values[pc], np.nan)
    return at(times + pd.Timedelta(minutes=60)) - at(times)

def per_regime(name, states, p, r, thr):
    m = ~np.isnan(r); states, p, r = states[m], p[m], r[m]
    print(f"\n  {name} — seuil BUY >= {thr:.2f} / SELL <= {1-thr:.2f}")
    print("  régime |  barres |  AUC  | signaux | justes | gain moyen/signal")
    for s in range(K):
        k = states == s
        if k.sum() < 100: print(f"    {s}    | {k.sum():6d}  |  (trop peu)"); continue
        auc = roc_auc_score(r[k] > 0, p[k])
        b, sl = k & (p >= thr), k & (p <= 1 - thr)
        g = np.r_[r[b], -r[sl]] - COST
        just = np.mean(np.r_[r[b] > 0, r[sl] < 0]) if len(g) else np.nan
        print(f"    {s}    | {k.sum():6d}  | {auc:.3f} | {len(g):7d} | {just:6.1%} | {g.mean() if len(g) else 0:+.2f}$")

if __name__ == "__main__":
    ns = load_app_funcs()
    print("[1] Description multi-timeframes du marché (HistData 2015-2025)")
    hist = base_closed(ns, load_m1())
    F_hist = regime_features(hist).dropna()
    mu, sd = F_hist.mean(), F_hist.std()
    Z = ((F_hist - mu) / sd).clip(-5, 5).values
    print(f"  {len(Z):,} bougies M15 x {Z.shape[1]} mesures")

    print(f"\n[2] HMM non supervisé : {K} régimes (aucune info de hausse/baisse)")
    hmm = GaussianHMM(n_components=K, covariance_type="full", n_iter=200, random_state=0)
    hmm.fit(Z)
    st_hist = hmm.predict(Z)
    print(describe(F_hist, st_hist).to_string())
    print("  durée moyenne d'un régime :", ", ".join(f"{s}: {1/(1-hmm.transmat_[s, s])*15/60:.1f}h" for s in range(K)))

    print("\n[3] Données Deriv 2026 (jamais vues par le HMM)")
    m1d = load_m1("DAT_DERIV_XAUUSD_M1_*.csv")
    der = base_closed(ns, m1d)
    F_der = regime_features(der)
    ok = F_der.notna().all(axis=1).values
    st = np.full(len(der), -1)
    st[ok] = hmm.predict(((F_der[ok] - mu) / sd).clip(-5, 5).values)
    t_calc = der["datetime"] + pd.Timedelta(minutes=15)
    ret = future_ret(m1d, pd.DatetimeIndex(t_calc))

    for label, mdir, start, thr in [
        ("ANCIEN modèle (22/02/2026), 22/02 -> 09/10/2026", REPO / "models_backup" / "2026-02-22", "2026-02-22", 0.60),
        ("NOUVEAU modèle (11/10/2026), 15/08 -> 09/10/2026", REPO, "2026-08-15", 0.55)]:
        rows = np.where(((der["datetime"] >= pd.Timestamp(start, tz="UTC")).values) & ok)[0]
        rows = rows[rows >= 49]
        p = predict(mdir, der, rows)
        print(f"\n  ===== {label} =====  AUC global = {roc_auc_score(ret[rows][~np.isnan(ret[rows])] > 0, p[~np.isnan(ret[rows])]):.3f}")
        per_regime(label, st[rows], p, ret[rows], thr)
        if "ANCIEN" in label:   # stabilité : le même régime est-il bon sur les 2 moitiés ?
            half = der["datetime"].iloc[rows].values < np.datetime64("2026-06-15")
            per_regime("  1re moitié (22/02 -> 15/06)", st[rows][half], p[half], ret[rows][half], thr)
            per_regime("  2e moitié (15/06 -> 09/10)", st[rows][~half], p[~half], ret[rows][~half], thr)
    joblib.dump({"hmm": hmm, "mu": mu, "sd": sd}, Path(__file__).parent / "regimes_hmm.joblib")
