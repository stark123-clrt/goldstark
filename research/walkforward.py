"""Walk-forward annuel LightGBM leak-free : entraine sur tout ce qui precede, teste sur l'annee suivante."""
import json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from evaluate import backtest, FEATS

df = pd.read_parquet(Path(__file__).parent / "ds_leakfree.parquet")
y = df["fut_ret"].values; dt = df["datetime"]
params = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_data_in_leaf=500,
              feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1)
for year in range(2019, 2026):
    ts = pd.Timestamp(f"{year}-01-01", tz="UTC"); vs = ts - pd.Timedelta(days=365)
    tr = ((dt < vs - pd.Timedelta(days=2)) & ~np.isnan(y)).values
    va = ((dt >= vs) & (dt < ts - pd.Timedelta(days=2)) & ~np.isnan(y)).values
    te = ((dt >= ts) & (dt < ts + pd.DateOffset(years=1))).values
    bst = lgb.train(params, lgb.Dataset(df.loc[tr, FEATS], y[tr] > 0), 2000,
                    valid_sets=[lgb.Dataset(df.loc[va, FEATS], y[va] > 0)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    p = bst.predict(df.loc[te, FEATS]); yt = y[te]; m = ~np.isnan(yt)
    print(f"{year}: AUC={roc_auc_score(yt[m] > 0, p[m]):.4f} iters={bst.best_iteration:4d} | "
          f"0.55: {backtest(yt, p, 0.55)} | 0.53: {backtest(yt, p, 0.53)}", flush=True)
