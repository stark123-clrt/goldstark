"""Precision REELLE du modele deploye sur les donnees Deriv (meme source que l'API), simulee
exactement comme app.py la calcule en live :
  - signal M15 : bougies M15 fermees, HTF = bougies fermees uniquement
  - micro-check : bougie M15 EN COURS incluse (vue a la minute 5 et 10), HTF fermees
Cible : le prix monte-t-il dans l'heure qui suit le moment du calcul ?
Seuils du robot : BUY si p >= 0.60, SELL si p <= 0.40.
"""
import json, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import load_app_funcs, load_m1, RULES

REPO = Path(__file__).parent.parent
MODEL_DIR = Path(os.environ.get("GS_MODEL_DIR", REPO))
meta = json.load(open(MODEL_DIR / "metadata_xauusd.json"))
FEATS, SEQ = meta["feature_cols"], meta["seq_length"]
TRAIN_DATE = pd.Timestamp(os.environ.get("GS_SPLIT", meta["date_trained"]), tz="UTC")
COST = 0.30

def bars(m1, rule):
    r = (m1.set_index("datetime").resample(rule)
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index())
    r["tickvol"] = 1
    return r

def htf_closed(raw):
    """HTF indexees a la cloture -> merge_asof ne voit que les bougies fermees (= live)."""
    out = {}
    for tf, d in raw.items():
        d = d.copy()
        if tf != "M15": d["datetime"] = d["datetime"] + RULES[tf][1]
        out[tf] = d
    return out

def predict(model, scaler, X3):
    out = []
    for s in range(0, len(X3), 1024):
        out.append(model.predict(X3[s:s + 1024], verbose=0)["direction"].ravel())
    return np.concatenate(out)

def windows(base, ends):
    X = np.nan_to_num(base[FEATS].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    return X, np.stack([X[t - SEQ + 1:t + 1] for t in ends])

def report(name, t_eval, prob, fut):
    m = ~np.isnan(fut)
    t_eval = pd.DatetimeIndex(t_eval)
    t_eval, prob, fut = t_eval[m], prob[m], fut[m]
    print(f"\n=== {name} ({len(prob):,} calculs) ===")
    for lab, sel in [("avant la date de coupure", np.asarray(t_eval < TRAIN_DATE)),
                     ("APRES la date de coupure", np.asarray(t_eval >= TRAIN_DATE))]:
        if sel.sum() < 100: continue
        p, r = prob[sel], fut[sel]
        auc = roc_auc_score(r > 0, p)
        buy, sell = p >= 0.60, p <= 0.40
        wb = (r[buy] > 0).mean() if buy.sum() else np.nan
        ws = (r[sell] < 0).mean() if sell.sum() else np.nan
        pnl = np.r_[r[buy], -r[sell]] - COST
        print(f"  {lab:34s} AUC={auc:.4f} | BUY: {buy.sum():5d} signaux, {wb:.1%} justes | "
              f"SELL: {sell.sum():5d} signaux, {ws:.1%} justes | gain moyen/signal {pnl.mean():+.2f}$")

if __name__ == "__main__":
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_DIR / "model_xauusd_best.keras")
    scaler = joblib.load(MODEL_DIR / "scaler_xauusd.joblib")
    ns = load_app_funcs()
    m1 = load_m1("DAT_DERIV_XAUUSD_M1_*.csv")
    print(f"Deriv M1 : {len(m1):,} bougies {m1.datetime.min()} -> {m1.datetime.max()}")
    close_m1 = m1.set_index("datetime")["close"]
    raw = {tf: bars(m1, r) for tf, (r, _) in RULES.items()}
    start_eval = m1.datetime.min() + pd.Timedelta(days=120)   # chauffe Daily (EMA50, etc.)

    def future_ret(times):
        """close 60 min apres - close au moment du calcul (NaN si trou)."""
        idx = close_m1.index
        now = close_m1.reindex(times, method="ffill").values
        later_t = times + pd.Timedelta(minutes=60)
        pos = idx.searchsorted(later_t)
        ok = (pos < len(idx)) & (np.abs((idx[np.minimum(pos, len(idx) - 1)] - later_t).total_seconds()) <= 120)
        later = close_m1.values[np.minimum(pos, len(idx) - 1)]
        return np.where(ok, later - now, np.nan)

    # --- 1) Signal M15 (bougies fermees) ---
    base = ns["build_features"](htf_closed(raw))
    ends = np.where((base["datetime"] >= start_eval).values)[0]
    ends = ends[ends >= SEQ - 1]
    X, W = windows(base, ends)
    Ws = np.nan_to_num(scaler.transform(W.reshape(-1, len(FEATS))), nan=0, posinf=0, neginf=0).reshape(W.shape).astype(np.float32)
    t_calc = base["datetime"].iloc[ends] + pd.Timedelta(minutes=15)   # calcule a la cloture de la bougie
    report("SIGNAL M15 (bougie fermee)", pd.DatetimeIndex(t_calc), predict(model, scaler, Ws), future_ret(pd.DatetimeIndex(t_calc)))

    # --- 2) Micro-checks : bougie M15 en cours, vue a la minute k ---
    for k in (5, 10):
        part = (m1.assign(bar=m1.datetime.dt.floor("15min"))
                  .loc[lambda d: (d.datetime - d.bar) < pd.Timedelta(minutes=k)]
                  .groupby("bar").agg(open=("open", "first"), high=("high", "max"),
                                      low=("low", "min"), close=("close", "last")).reset_index()
                  .rename(columns={"bar": "datetime"}))
        part["tickvol"] = 1
        # remplace chaque bougie M15 par sa version partielle UNIQUEMENT en derniere position de fenetre :
        # on recalcule les features avec la bougie partielle a la place de la vraie, une barre sur deux suffit
        probs, times = [], []
        m15 = raw["M15"].set_index("datetime")
        sample = base["datetime"].iloc[ends].iloc[::2]          # echantillon (1 bougie sur 2) pour aller vite
        sample = sample[sample >= TRAIN_DATE - pd.Timedelta(days=45)]   # micro-checks : autour de la periode test seulement
        pmap = part.set_index("datetime")
        for t in sample:
            if t not in pmap.index: continue
            hist = m15.loc[:t].iloc[-400:].copy()
            hist.iloc[-1] = pmap.loc[t, ["open", "high", "low", "close", "tickvol"]].values
            calc_t = t + pd.Timedelta(minutes=k)
            dfs = {"M15": hist.reset_index()}
            for tf in ["M30", "H1", "H4", "Daily"]:
                d = raw[tf]
                d = d[d["datetime"] + RULES[tf][1] <= calc_t].iloc[-400:].copy()
                d["datetime"] = d["datetime"] + RULES[tf][1]
                dfs[tf] = d
            b = ns["build_features"](dfs)
            if len(b) < SEQ: continue
            Xw = np.nan_to_num(b[FEATS].tail(SEQ).values.astype(np.float32), nan=0, posinf=0, neginf=0)
            probs.append(np.nan_to_num(scaler.transform(Xw), nan=0, posinf=0, neginf=0))
            times.append(calc_t)
        Wm = np.stack(probs).astype(np.float32)
        report(f"MICRO-CHECK minute {k} (bougie en cours)", pd.DatetimeIndex(times),
               predict(model, scaler, Wm), future_ret(pd.DatetimeIndex(times)))
