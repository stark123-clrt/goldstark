"""Evalue honnetement le modele deploye + un LightGBM leak-free, meme periode de test."""
import json, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score

REPO = Path(r"C:\Users\chris\Documents\GitHub\goldstark")
OUT = Path(__file__).parent
COST = 0.30  # aller-retour en $ (spread XAUUSD ~0.2-0.4)
meta = json.load(open(REPO / "metadata_xauusd.json"))
FEATS, SEQ = meta["feature_cols"], meta["seq_length"]

def backtest(close_ret, prob, thr=0.66):
    """Trades non chevauchants de 4 bougies : BUY si p>=thr, SELL si p<=1-thr."""
    pnl, last = [], -10
    for i, (r, p) in enumerate(zip(close_ret, prob)):
        if i < last + 4 or np.isnan(r): continue
        d = 1 if p >= thr else (-1 if p <= 1 - thr else 0)
        if d: pnl.append(d * r - COST); last = i
    a = np.array(pnl)
    if not len(a): return "aucun trade"
    pf = a[a > 0].sum() / max(-a[a < 0].sum(), 1e-9)
    return f"trades={len(a):5d}  WR={(a>0).mean():.1%}  PF={pf:.2f}  PnL={a.sum():+8.1f}$/oz"

def report(name, y_ret, prob):
    m = ~np.isnan(y_ret)
    auc = roc_auc_score(y_ret[m] > 0, prob[m])
    print(f"  {name:38s} AUC={auc:.4f} | {backtest(y_ret, prob)}", flush=True)
    for thr in (0.55, 0.60):
        print(f"  {'':38s} seuil {thr}: {backtest(y_ret, prob, thr)}", flush=True)

def gru_predict(df, rows):
    import tensorflow as tf
    model = tf.keras.models.load_model(REPO / "model_xauusd_best.keras")
    scaler = joblib.load(REPO / "scaler_xauusd.joblib")
    X = np.nan_to_num(df[FEATS].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    Xs = np.nan_to_num(scaler.transform(X), nan=0, posinf=0, neginf=0).astype(np.float32)
    out = []
    for s in range(0, len(rows), 1024):
        idx = rows[s:s + 1024]
        batch = np.stack([Xs[t - SEQ + 1:t + 1] for t in idx])
        out.append(model.predict(batch, verbose=0)["direction"].ravel())
    return np.concatenate(out)

if __name__ == "__main__":
    TEST_START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2025-01-01", tz="UTC")
    res = {}
    for variant in ["leaky", "leakfree"]:
        df = pd.read_parquet(OUT / f"ds_{variant}.parquet")
        rows = np.where((df["datetime"] >= TEST_START).values)[0]
        rows = rows[rows >= SEQ - 1]
        print(f"\n[{variant}] test {df.datetime.iloc[rows[0]]} -> {df.datetime.iloc[rows[-1]]} ({len(rows):,} barres)")
        p = gru_predict(df, rows)
        report(f"GRU deploye ({variant})", df["fut_ret"].values[rows], p)
        res[variant] = (df, rows, p)

    import lightgbm as lgb
    df, rows, _ = res["leakfree"]
    y = df["fut_ret"].values
    dt = df["datetime"]
    VAL_START = TEST_START - pd.Timedelta(days=365)
    tr = ((dt < VAL_START - pd.Timedelta(days=2)) & ~np.isnan(y)).values
    va = ((dt >= VAL_START) & (dt < TEST_START - pd.Timedelta(days=2)) & ~np.isnan(y)).values
    params = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_data_in_leaf=500,
                  feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1)
    dtr = lgb.Dataset(df.loc[tr, FEATS], y[tr] > 0)
    dva = lgb.Dataset(df.loc[va, FEATS], y[va] > 0)
    bst = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    pv = bst.predict(df.loc[va, FEATS])
    print(f"\n[LightGBM leak-free] best_iter={bst.best_iteration} | AUC val={roc_auc_score(y[va] > 0, pv):.4f}")
    report("LightGBM (leakfree)", y[rows], bst.predict(df.iloc[rows][FEATS]))
    imp = pd.Series(bst.feature_importance("gain"), FEATS).sort_values(ascending=False)
    print("  top features:", ", ".join(imp.index[:12]))
    print(f"\n  Base: % hausses test = {(y[rows][~np.isnan(y[rows])] > 0).mean():.1%}")
