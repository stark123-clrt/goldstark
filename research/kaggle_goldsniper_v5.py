"""
GOLDSNIPER XAUUSD v5 — entraînement Kaggle (GPU)
================================================
Même architecture que le modèle déployé (Conv1D x3 + BiLSTM 128/64 + attention, têtes
direction / confidence / magnitude), mêmes 346 features que app.py, mais entraîné EXACTEMENT
comme il est utilisé en direct :

  1. Données : ton dataset HistData M1 2015-2025 + les bougies M1 Deriv (source de l'API) pour 2026.
  2. Pas de fuite : M30/H1/H4/Daily alignées sur leur heure de CLÔTURE -> le modèle ne voit que
     des bougies fermées, comme l'API. (app.py lit "htf_align": "close" dans le metadata.)
  3. Bruit des micro-checks : en plus des bougies M15 fermées, on lui montre des bougies M15
     INCOMPLÈTES (vue à la minute 1..14), exactement ce que l'API calcule chaque minute.
  4. GaussianNoise sur l'entrée : il apprend à rester stable malgré le bruit.
  5. Probabilités calibrées (température) : 0.60 annoncé ≈ 60 % de réussite réelle.
  6. Test honnête sur 2026 (Deriv), jamais vu, puis réentraînement final sur TOUT pour le déploiement.

Kaggle : Accelerator = GPU, ajouter en Input ton dataset HistData + les 2 fichiers
DAT_DERIV_XAUUSD_M1_2025.csv / _2026.csv. Sortie : /kaggle/working/model_v5/
(model_xauusd_best.keras, scaler_xauusd.joblib, metadata_xauusd.json) -> à copier dans le repo goldstark.
"""
import os, sys, json, gc, time, subprocess
from pathlib import Path
from multiprocessing import Pool

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ta"], check=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, ta, joblib
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

# ── RÉGLAGES ──────────────────────────────────────────────────────────────────
SEQ          = 50
HORIZON_MIN  = 60          # cible : le prix monte-t-il dans l'heure qui suit le calcul ?
AUG_FRAC     = float(os.environ.get("AUG_FRAC", 0.20))       # part des bougies M15 qu'on montre aussi en version incomplète
CLIP         = 10.0
CONF_ATR     = 0.5         # "mouvement significatif" = |mouvement| > 0.5 ATR
VAL_START    = pd.Timestamp("2025-07-01", tz="UTC")
TEST_START   = pd.Timestamp("2026-01-01", tz="UTC")   # 2026 = Deriv = vraie source de l'API
EPOCHS_MAX   = int(os.environ.get("EPOCHS_MAX", 30))
HTF_ALIGN    = os.environ.get("GS_HTF_ALIGN", "close")   # "open" = comme le modèle original (bougies HTF non fermées)
CALIBRATE    = os.environ.get("GS_CALIBRATE", "1") == "1"
FINAL_REFIT  = True        # réentraîne sur TOUTES les données pour le modèle déployé
INPUT        = Path(os.environ.get("GS_INPUT", "/kaggle/input"))
OUT          = Path(os.environ.get("GS_OUT", "/kaggle/working/model_v5")); OUT.mkdir(parents=True, exist_ok=True)
RULES        = {"M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "Daily": "1D"}
TF_DUR       = {"M30": pd.Timedelta(minutes=30), "H1": pd.Timedelta(hours=1),
                "H4": pd.Timedelta(hours=4), "Daily": pd.Timedelta(days=1)}
MERGE_TOLERANCES = {"M30": pd.Timedelta(hours=2), "H1": pd.Timedelta(hours=4),
                    "H4": pd.Timedelta(days=2), "Daily": pd.Timedelta(days=5)}
RNG = np.random.default_rng(42)
if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ── 1. DONNÉES ────────────────────────────────────────────────────────────────
def read_csv(p):
    d = pd.read_csv(p, sep=";", header=None, names=["dt", "open", "high", "low", "close", "vol"])
    # HistData (et nos fichiers Deriv) = heure EST fixe (UTC-5) -> UTC comme Deriv/l'API
    d["datetime"] = (pd.to_datetime(d["dt"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
    for c in ["open", "high", "low", "close"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["datetime", "open", "high", "low", "close"]].dropna()

def load_m1():
    years = os.environ.get("GS_YEARS")   # test rapide : ex. "2024,2025,2026"
    files = [p for p in INPUT.rglob("*.csv") if "XAUUSD" in p.name.upper()
             and (not years or any(y in p.name for y in years.split(",")))]
    hist  = [read_csv(p) for p in files if "DERIV" not in p.name.upper()]
    der   = [read_csv(p) for p in files if "DERIV" in p.name.upper()]
    print(f"  {len(hist)} fichiers HistData, {len(der)} fichiers Deriv")
    m1 = pd.concat(hist)
    if der:
        d = pd.concat(der)
        m1 = pd.concat([m1[m1.datetime < d.datetime.min()], d])   # Deriv prioritaire sur la période commune
    m1 = m1.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    print(f"  M1 : {len(m1):,} bougies {m1.datetime.min()} -> {m1.datetime.max()}")
    return m1

def resample(m1, rule):
    r = (m1.set_index("datetime").resample(rule)
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index())
    r["tickvol"] = 1                      # identique au live (Deriv n'a pas de volume)
    return r[["datetime", "open", "high", "low", "close", "tickvol"]]

# ── 2. FEATURES : copie EXACTE de app.py (add_features + build_features) ─────
def add_features(df_raw, tf_name):
    p = tf_name.lower()
    renamed = df_raw.rename(columns={"open":f"{p}_open","high":f"{p}_high","low":f"{p}_low","close":f"{p}_close","tickvol":f"{p}_tickvol"})
    c,h,l,o,v = renamed[f"{p}_close"],renamed[f"{p}_high"],renamed[f"{p}_low"],renamed[f"{p}_open"],renamed[f"{p}_tickvol"]
    cols = {}
    cols[f"{p}_rsi14"]=ta.momentum.RSIIndicator(c,14).rsi()
    cols[f"{p}_rsi7"]=ta.momentum.RSIIndicator(c,7).rsi()
    s5=ta.momentum.StochasticOscillator(h,l,c,5,3); s14=ta.momentum.StochasticOscillator(h,l,c,14,3)
    cols[f"{p}_stoch_k5"]=s5.stoch(); cols[f"{p}_stoch_d5"]=s5.stoch_signal()
    cols[f"{p}_stoch_k14"]=s14.stoch(); cols[f"{p}_stoch_d14"]=s14.stoch_signal()
    macd=ta.trend.MACD(c,17,8,9)
    cols[f"{p}_macd"]=macd.macd(); cols[f"{p}_macd_sig"]=macd.macd_signal(); cols[f"{p}_macd_hist"]=macd.macd_diff()
    cols[f"{p}_roc3"]=ta.momentum.ROCIndicator(c,3).roc(); cols[f"{p}_roc10"]=ta.momentum.ROCIndicator(c,10).roc()
    cols[f"{p}_willr7"]=ta.momentum.WilliamsRIndicator(h,l,c,7).williams_r()
    cols[f"{p}_willr14"]=ta.momentum.WilliamsRIndicator(h,l,c,14).williams_r()
    atr14=ta.volatility.AverageTrueRange(h,l,c,14).average_true_range()
    for pr in [5,8,13,21,34,50]: cols[f"{p}_ema{pr}"]=ta.trend.EMAIndicator(c,pr).ema_indicator()
    for pr in [5,13,21]: cols[f"{p}_dist_ema{pr}"]=(c-cols[f"{p}_ema{pr}"])/atr14.replace(0,np.nan)
    adx=ta.trend.ADXIndicator(h,l,c,14)
    cols[f"{p}_adx"]=adx.adx(); cols[f"{p}_adx_pos"]=adx.adx_pos(); cols[f"{p}_adx_neg"]=adx.adx_neg()
    cols[f"{p}_cci14"]=ta.trend.CCIIndicator(h,l,c,14).cci(); cols[f"{p}_cci7"]=ta.trend.CCIIndicator(h,l,c,7).cci()
    ema21=cols[f"{p}_ema21"]; cols[f"{p}_ema21_slope"]=ema21.diff(3)/ema21.shift(3).replace(0,np.nan)*100
    cols[f"{p}_atr5"]=ta.volatility.AverageTrueRange(h,l,c,5).average_true_range()
    cols[f"{p}_atr7"]=ta.volatility.AverageTrueRange(h,l,c,7).average_true_range()
    cols[f"{p}_atr14"]=atr14
    for bw,bs in [(10,2),(20,2)]:
        bb=ta.volatility.BollingerBands(c,bw,bs); t2=f"{bw}_{bs}"
        cols[f"{p}_bb_upper_{t2}"]=bb.bollinger_hband(); cols[f"{p}_bb_lower_{t2}"]=bb.bollinger_lband()
        cols[f"{p}_bb_mid_{t2}"]=bb.bollinger_mavg(); cols[f"{p}_bb_width_{t2}"]=bb.bollinger_wband()
        cols[f"{p}_bb_pct_{t2}"]=bb.bollinger_pband()
    ret=c.pct_change()
    cols[f"{p}_vol5"]=ret.rolling(5).std(); cols[f"{p}_vol14"]=ret.rolling(14).std()
    cols[f"{p}_vol_ratio"]=cols[f"{p}_vol5"]/cols[f"{p}_vol14"].replace(0,np.nan)
    hl=np.log(h/l.replace(0,np.nan))
    cols[f"{p}_parkinson14"]=(hl**2/(4*np.log(2))).rolling(14).mean().apply(np.sqrt)
    cols[f"{p}_body"]=abs(c-o); cols[f"{p}_body_pct"]=abs(c-o)/o.replace(0,np.nan)*100
    cols[f"{p}_wick_up"]=h-pd.concat([o,c],axis=1).max(axis=1)
    cols[f"{p}_wick_down"]=pd.concat([o,c],axis=1).min(axis=1)-l
    cols[f"{p}_range"]=h-l; cols[f"{p}_direction"]=np.sign(c-o)
    cols[f"{p}_close_pos"]=(c-l)/(h-l).replace(0,np.nan)
    cols[f"{p}_return1"]=c.pct_change(1); cols[f"{p}_return3"]=c.pct_change(3); cols[f"{p}_return5"]=c.pct_change(5)
    cols[f"{p}_mom5"]=cols[f"{p}_direction"].rolling(5).sum(); cols[f"{p}_mom10"]=cols[f"{p}_direction"].rolling(10).sum()
    cols[f"{p}_vol_rel5"]=v/v.rolling(5).mean().replace(0,np.nan)
    cols[f"{p}_vol_rel20"]=v/v.rolling(20).mean().replace(0,np.nan)
    enriched=pd.concat([renamed,pd.DataFrame(cols,index=renamed.index)],axis=1)
    return enriched.dropna().reset_index(drop=True)

def merge_htf(tf_dfs, align=None):
    align = align or HTF_ALIGN
    base = tf_dfs["M15"].copy()
    for tf_name in ["M30", "H1", "H4", "Daily"]:
        d = tf_dfs[tf_name].copy()
        if align == "close":
            d["datetime"] = d["datetime"] + TF_DUR[tf_name]    # ANTI-FUITE : heure de clôture
        base = pd.merge_asof(base, d, on="datetime", direction="backward",
                             tolerance=MERGE_TOLERANCES[tf_name])
    return base

def cross_features(base):
    p="m15"
    base["gold_atr_pct"]=base[f"{p}_atr14"]/base[f"{p}_close"]*100
    base["gold_range_pct"]=base[f"{p}_range"]/base[f"{p}_close"]*100
    base["rsi_align_m15_h1"]=np.sign(base["m15_rsi14"]-50)*np.sign(base["h1_rsi14"]-50)
    base["rsi_align_m15_h4"]=np.sign(base["m15_rsi14"]-50)*np.sign(base["h4_rsi14"]-50)
    base["rsi_align_m15_daily"]=np.sign(base["m15_rsi14"]-50)*np.sign(base["daily_rsi14"]-50)
    base["trend_align_m15_h1"]=np.sign(base["m15_ema21_slope"])*np.sign(base["h1_ema21_slope"])
    base["trend_align_m15_h4"]=np.sign(base["m15_ema21_slope"])*np.sign(base["h4_ema21_slope"])
    base["trend_align_m15_daily"]=np.sign(base["m15_ema21_slope"])*np.sign(base["daily_ema21_slope"])
    base["bull_alignment"]=(
        (base["m15_close"]>base["m15_ema21"]).astype(int)+
        (base["m30_close"]>base["m30_ema21"]).astype(int)+
        (base["h1_close"]>base["h1_ema21"]).astype(int)+
        (base["h4_close"]>base["h4_ema21"]).astype(int)+
        (base["daily_close"]>base["daily_ema21"]).astype(int)
    )
    base["rsi_spread_m15_h1"]=base["m15_rsi14"]-base["h1_rsi14"]
    base["rsi_spread_m15_h4"]=base["m15_rsi14"]-base["h4_rsi14"]
    base["rsi_spread_h1_daily"]=base["h1_rsi14"]-base["daily_rsi14"]
    base["rsi_divergence"]=base[f"{p}_return5"]*(base["m15_rsi14"]-base["m15_rsi14"].shift(5))
    base["atr_ratio_m15_h1"]=base["m15_atr14"]/base["h1_atr14"].replace(0,np.nan)
    base["atr_ratio_m15_h4"]=base["m15_atr14"]/base["h4_atr14"].replace(0,np.nan)
    base["velocity5"]=base[f"{p}_return1"].abs().rolling(5).sum()
    base["velocity10"]=base[f"{p}_return1"].abs().rolling(10).sum()
    base["vol_of_vol"]=base[f"{p}_vol14"].rolling(14).std()
    dt=pd.to_datetime(base["datetime"])
    base["hour_sin"]=np.sin(2*np.pi*(dt.dt.hour+dt.dt.minute/60)/24)
    base["hour_cos"]=np.cos(2*np.pi*(dt.dt.hour+dt.dt.minute/60)/24)
    base["dow_sin"]=np.sin(2*np.pi*dt.dt.dayofweek/7)
    base["dow_cos"]=np.cos(2*np.pi*dt.dt.dayofweek/7)
    hour=dt.dt.hour
    base["session_asian"]=((hour>=0)&(hour<8)).astype(int)
    base["session_london"]=((hour>=8)&(hour<12)).astype(int)
    base["session_newyork"]=((hour>=13)&(hour<17)).astype(int)
    base["session_overlap"]=((hour>=8)&(hour<12)).astype(int)
    return base

def build_features(raw, align=None):
    tfd = {tf: add_features(df, tf) for tf, df in raw.items()}
    return cross_features(merge_htf(tfd, align)).dropna().reset_index(drop=True)

# ── 3. CIBLES ─────────────────────────────────────────────────────────────────
def price_at(close_m1, times):
    """Prix à l'instant T = clôture de la dernière bougie M1 terminée avant T (NaN si trou > 3 min)."""
    idx = close_m1.index
    q = pd.DatetimeIndex(times) - pd.Timedelta(minutes=1)
    pos = idx.searchsorted(q, side="right") - 1
    posc = np.maximum(pos, 0)
    ok = (pos >= 0) & ((q - idx[posc]) <= pd.Timedelta(minutes=3))
    return np.where(ok, close_m1.values[posc], np.nan)

def targets(close_m1, calc_times, atr):
    now = price_at(close_m1, calc_times)
    later = price_at(close_m1, pd.DatetimeIndex(calc_times) + pd.Timedelta(minutes=HORIZON_MIN))
    ret = later - now
    y_dir = (ret > 0).astype(np.float32)
    y_conf = (np.abs(ret) > CONF_ATR * atr).astype(np.float32)
    y_mag = np.clip(np.abs(ret) / np.maximum(atr, 1e-6), 0, 5).astype(np.float32)
    return ret, y_dir, y_conf, y_mag

# ── 4. BOUGIES INCOMPLÈTES (micro-checks) ────────────────────────────────────
# variables globales partagées avec les workers (fork Linux)
G = {}

def partial_row(args):
    """Ligne de features telle que l'API la calcule à la minute k de la bougie M15 en cours."""
    i, k = args
    base, m15raw, m1 = G["base"], G["m15raw"], G["m1"]
    b = base.at[i, "datetime"]
    j = G["m15pos"].get(b)
    if j is None or j < 480: return None
    sl = m15raw.iloc[j - 479:j + 1].copy()                     # ~5 jours d'historique, comme le live
    a, z = m1.index.searchsorted(b), m1.index.searchsorted(b + pd.Timedelta(minutes=k))
    if z <= a: return None
    part = m1.iloc[a:z]
    sl.iloc[-1, sl.columns.get_loc("open")]  = part["open"].iloc[0]
    sl.iloc[-1, sl.columns.get_loc("high")]  = part["high"].max()
    sl.iloc[-1, sl.columns.get_loc("low")]   = part["low"].min()
    sl.iloc[-1, sl.columns.get_loc("close")] = part["close"].iloc[-1]
    f15 = add_features(sl.reset_index(drop=True), "M15")
    if len(f15) == 0 or f15["datetime"].iloc[-1] != b: return None
    frame = base.iloc[max(0, i - 30):i + 1].copy()
    last = f15.iloc[-1]
    for c in f15.columns:
        if c != "datetime": frame.iloc[-1, frame.columns.get_loc(c)] = last[c]
    row = cross_features(frame).iloc[-1]
    return i, k, row[G["feats"]].values.astype(np.float64)

def build_partials(base, m15raw, m1, feats, rows):
    G.update(base=base, m15raw=m15raw.reset_index(drop=True),
             m1=m1.set_index("datetime"), feats=feats,
             m15pos={t: n for n, t in enumerate(m15raw["datetime"])})
    jobs = [(int(i), int(RNG.integers(1, 15))) for i in rows]
    if os.name == "nt":   # Windows (test local) : pas de fork, on calcule en série
        res = [r for r in map(partial_row, jobs) if r is not None]
    else:
        with Pool(os.cpu_count()) as pool:
            res = [r for r in pool.imap(partial_row, jobs, chunksize=64) if r is not None]
    G.clear()
    idx = np.array([r[0] for r in res]); ks = np.array([r[1] for r in res])
    X = np.stack([r[2] for r in res]).astype(np.float32)
    return idx, ks, X

# ── 5. MODÈLE (même architecture que le modèle déployé) ──────────────────────
def build_model(nf):
    inp = layers.Input((SEQ, nf), name="input")
    x = layers.GaussianNoise(0.05)(inp)                         # apprendre avec le bruit
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x); x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x); x = layers.Dropout(0.3)(x)
    att = layers.Dense(1, activation="sigmoid")(x)   # attention par bougie
    x = layers.Multiply()([x, att]); x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    logit = layers.Dense(1, dtype="float32", name="direction_logit")(x)
    d = layers.Activation("sigmoid", dtype="float32", name="direction")(logit)
    cf = layers.Dense(1, activation="sigmoid", dtype="float32", name="confidence")(x)
    mg = layers.Dense(1, activation="softplus", dtype="float32", name="magnitude")(x)
    m = Model(inp, {"direction": d, "confidence": cf, "magnitude": mg})
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=1.0),
              loss={"direction": "binary_crossentropy", "confidence": "binary_crossentropy", "magnitude": "huber"},
              loss_weights={"direction": 1.0, "confidence": 0.3, "magnitude": 0.2},
              metrics={"direction": [tf.keras.metrics.AUC(name="auc")]})
    return m

class Seq(tf.keras.utils.Sequence):
    """Fenêtres de 50 bougies ; pour un échantillon 'incomplet', la dernière ligne est remplacée."""
    def __init__(self, X, Xp, items, Y, batch, shuffle):
        super().__init__()
        self.X, self.Xp, self.items, self.Y, self.b, self.sh = X, Xp, items.copy(), Y, batch, shuffle
        self.offs = np.arange(-SEQ + 1, 1)
        if shuffle: RNG.shuffle(self.items)
    def __len__(self): return int(np.ceil(len(self.items) / self.b))
    def on_epoch_end(self):
        if self.sh: RNG.shuffle(self.items)
    def __getitem__(self, k):
        it = self.items[k * self.b:(k + 1) * self.b]
        Xb = self.X[it[:, 0][:, None] + self.offs]
        aug = it[:, 1] >= 0
        if aug.any(): Xb[aug, -1] = self.Xp[it[aug, 1]]
        y = {n: self.Y[n][it[:, 2]] for n in ("direction", "confidence", "magnitude")}
        return Xb, y

def fit_temperature(logits, y):
    best_t, best = 1.0, 1e9
    for t in np.linspace(0.5, 20, 400):
        p = 1 / (1 + np.exp(-logits / t)); p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if ll < best: best, best_t = ll, t
    return best_t

def calibrated(model, T):
    logit = model.get_layer("direction_logit").output
    d = layers.Activation("sigmoid", dtype="float32", name="direction")(layers.Rescaling(1.0 / T, dtype="float32")(logit))
    return Model(model.input, {"direction": d, "confidence": model.get_layer("confidence").output,
                               "magnitude": model.get_layer("magnitude").output})

def table(name, ret, p):
    m = ~np.isnan(ret); ret, p = ret[m], p[m]
    if len(p) < 50: return
    print(f"\n  {name} : {len(p):,} calculs | AUC = {roc_auc_score(ret > 0, p):.4f}")
    print("    seuil  | BUY : nb  justes | SELL : nb  justes | gain moyen/signal (-0.30$ coût)")
    for thr in (0.52, 0.55, 0.58, 0.60, 0.65):
        b, s = p >= thr, p <= 1 - thr
        if b.sum() + s.sum() == 0: continue
        g = np.r_[ret[b], -ret[s]] - 0.30
        print(f"    {thr:.2f}   | {b.sum():6d} {np.mean(ret[b] > 0) if b.sum() else 0:6.1%} | "
              f"{s.sum():6d} {np.mean(ret[s] < 0) if s.sum() else 0:6.1%} | {g.mean():+.2f}$")
    print("    calibration : proba annoncée -> % hausse réel")
    for lo, hi in [(0, .4), (.4, .45), (.45, .5), (.5, .55), (.55, .6), (.6, 1.01)]:
        s = (p >= lo) & (p < hi)
        if s.sum() >= 50: print(f"      [{lo:.2f},{min(hi,1):.2f}) : {s.sum():6d} -> {np.mean(ret[s] > 0):.1%}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70 + "\n  GOLDSNIPER v5 — leak-free, micro-checks, calibré\n" + "=" * 70)
    print("\n[1] Données"); m1 = load_m1()
    raw = {tf: resample(m1, r) for tf, r in RULES.items()}

    print("\n[2] Features (bougies fermées)"); base = build_features(raw)
    feats = [c for c in base.columns if c != "datetime"]
    assert len(feats) == 346, f"{len(feats)} features au lieu de 346"
    calc_t = base["datetime"] + pd.Timedelta(minutes=15)
    close_m1 = m1.set_index("datetime")["close"]
    ret, yd, yc, ym = targets(close_m1, calc_t, base["m15_atr14"].values)
    print(f"  {len(base):,} bougies M15 | hausse = {np.nanmean(ret > 0):.1%}")

    n = len(base); dt = base["datetime"]
    valid = (~np.isnan(ret)) & (np.arange(n) >= SEQ - 1)
    tr = valid & (dt < VAL_START - pd.Timedelta(days=1)).values
    va = valid & (dt >= VAL_START).values & (dt < TEST_START - pd.Timedelta(days=1)).values
    te = valid & (dt >= TEST_START).values

    print(f"\n[3] Bougies incomplètes (micro-checks) sur {AUG_FRAC:.0%} des bougies ...")
    pick = np.where(valid)[0]; pick = RNG.choice(pick, int(len(pick) * AUG_FRAC), replace=False)
    if len(pick):
        aidx, aks, Xp_raw = build_partials(base, raw["M15"], m1, feats, np.sort(pick))
    else:
        aidx, aks, Xp_raw = np.zeros(0, int), np.zeros(0, int), np.zeros((0, len(feats)))
    a_calc = base["datetime"].iloc[aidx].reset_index(drop=True) + pd.to_timedelta(aks, unit="m")
    aret, ayd, ayc, aym = targets(close_m1, a_calc, base["m15_atr14"].values[aidx])
    print(f"  {len(aidx):,} échantillons incomplets ({(time.time()-t0)/60:.1f} min)")

    # comme app.py : NaN/inf -> 0 AVANT le scaler
    X_raw = np.nan_to_num(base[feats].values.astype(np.float64), nan=0, posinf=0, neginf=0)
    Xp_raw = np.nan_to_num(Xp_raw, nan=0, posinf=0, neginf=0)
    base_live = build_features(raw, "close") if HTF_ALIGN == "open" else None   # pour le score "comme en direct"
    del raw; gc.collect()

    def run(train_mask, aug_train_mask, val_mask, aug_val_mask, epochs=None):
        sc = RobustScaler().fit(X_raw[train_mask])
        tfm = lambda A: np.clip(np.nan_to_num(sc.transform(A), nan=0, posinf=0, neginf=0), -CLIP, CLIP).astype(np.float32)
        X = tfm(X_raw)
        Xp = tfm(Xp_raw) if len(Xp_raw) else np.zeros((1, X.shape[1]), np.float32)
        Y = {"direction": np.r_[yd, ayd], "confidence": np.r_[yc, ayc], "magnitude": np.r_[ym, aym]}
        Y = {k: np.nan_to_num(v).astype(np.float32) for k, v in Y.items()}
        def items(mask, amask):
            full = np.where(mask)[0]; ai = np.where(amask)[0]
            return np.r_[np.c_[full, -np.ones_like(full), full],
                         np.c_[aidx[ai], ai, n + ai]].astype(np.int64)
        model = build_model(len(feats))
        cb = []
        val = None
        if val_mask is not None:
            val = Seq(X, Xp, items(val_mask, aug_val_mask), Y, 1024, False)
            cb = [callbacks.EarlyStopping(monitor="val_direction_auc", mode="max", patience=5, restore_best_weights=True),
                  callbacks.ReduceLROnPlateau(monitor="val_direction_auc", mode="max", factor=0.5, patience=2)]
        model.fit(Seq(X, Xp, items(train_mask, aug_train_mask), Y, 512, True), validation_data=val,
                  epochs=epochs or EPOCHS_MAX, callbacks=cb, verbose=2)
        return model, sc, X, Xp

    def logits_of(model, X, Xp, rows, arows):
        if len(rows) == 0: return np.zeros(0)
        lm = Model(model.input, model.get_layer("direction_logit").output)
        offs = np.arange(-SEQ + 1, 1)
        out = []
        for s in range(0, len(rows), 2048):
            r = rows[s:s + 2048]; W = X[r[:, None] + offs]
            if arows is not None:
                a = arows[s:s + 2048]; W[:, -1] = Xp[a]
            out.append(lm.predict(W, verbose=0).ravel())
        return np.concatenate(out).astype(np.float64)

    amask = lambda m: (m[aidx] & ~np.isnan(aret)) if len(aidx) else np.zeros(0, bool)
    print("\n[4] Entraînement (train 2015 -> juin 2025, validation juil.-déc. 2025)")
    model, sc, X, Xp = run(tr, amask(tr), va, amask(va))
    best_epoch = max(1, int(np.argmax(model.history.history["val_direction_auc"])) + 1) if model.history else 10

    print("\n[5] Calibration des probabilités (validation)")
    vr = np.where(va)[0]; va_a = np.where(amask(va))[0]
    lv = np.r_[logits_of(model, X, Xp, vr, None), logits_of(model, X, Xp, aidx[va_a], va_a)]
    T = fit_temperature(lv, np.r_[yd[vr], ayd[va_a]]) if CALIBRATE else 1.0
    print(f"  température = {T:.2f}")

    print("\n[6] TEST 2026 (Deriv, jamais vu) — modèle déployé actuel : AUC 0.514")
    ter = np.where(te)[0]; te_a = np.where(amask(te))[0]
    sig = lambda z: 1 / (1 + np.exp(-z / T))
    p_m15 = sig(logits_of(model, X, Xp, ter, None))
    p_mic = sig(logits_of(model, X, Xp, aidx[te_a], te_a))
    table("SIGNAL M15 (bougie fermée)", ret[ter], p_m15)
    table("MICRO-CHECKS (bougie en cours)", aret[te_a], p_mic)
    test_auc = roc_auc_score(ret[ter] > 0, p_m15)
    if base_live is not None:
        # même modèle, mêmes dates de test, mais features calculées avec les seules bougies FERMÉES (= l'API en direct)
        bl = base_live[base_live["datetime"] >= TEST_START - pd.Timedelta(days=30)].reset_index(drop=True)
        XL = np.clip(np.nan_to_num(sc.transform(np.nan_to_num(bl[feats].values.astype(np.float64), nan=0, posinf=0, neginf=0)),
                                   nan=0, posinf=0, neginf=0), -CLIP, CLIP).astype(np.float32)
        rows = np.where((bl["datetime"] >= TEST_START).values)[0]; rows = rows[rows >= SEQ - 1]
        rl, _, _, _ = targets(close_m1, bl["datetime"].iloc[rows] + pd.Timedelta(minutes=15), bl["m15_atr14"].values[rows])
        pl = sig(logits_of(model, XL, XL, rows, None))
        table("COMME EN DIRECT (bougies H1/H4/Daily fermées seulement)", rl, pl)

    if FINAL_REFIT:
        print(f"\n[7] Réentraînement final sur TOUTES les données ({best_epoch} époques) pour le déploiement")
        allm = valid
        model, sc, X, Xp = run(allm, amask(allm), None, None, epochs=best_epoch)

    print("\n[8] Sauvegarde")
    final = calibrated(model, T)
    final.save(str(OUT / "model_xauusd_best.keras"))
    joblib.dump(sc, OUT / "scaler_xauusd.joblib")
    json.dump({"symbol": "XAUUSD", "version": "v5", "seq_length": SEQ, "future_bars": 4,
               "horizon_min": HORIZON_MIN, "feature_cols": feats, "n_features": len(feats),
               "htf_align": HTF_ALIGN, "clip": CLIP, "temperature": float(T),
               "best_epoch": best_epoch, "test_auc_2026_m15": float(test_auc),
               "date_trained": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
               "data_end": str(m1.datetime.max())},
              open(OUT / "metadata_xauusd.json", "w"), indent=2)
    print(f"  OK -> {OUT}  ({(time.time()-t0)/60:.1f} min)")
    print(f"\n  >>> AUC test 2026 = {test_auc:.4f} (modèle actuel : 0.514). Déployer seulement si c'est mieux.")

# ── FEATURES MULTI-TIMEFRAMES "À L'INSTANT T" ────────────────────────────────
# À la clôture de chaque bougie M15 (instant T), on calcule M30/H1/H4/Daily EXACTEMENT comme sur le
# graphique à cet instant : bougies fermées + la bougie EN COURS (de son ouverture jusqu'à T).
# Pas de futur, et c'est ce que l'API calculera en direct (même règle, même historique HTF_HIST).
HTF_HIST = 200
FREQ = {"M30": "30min", "H1": "1h", "H4": "4h", "Daily": "1D"}
AS = {}

def _asof_job(args):
    tf_, i = args
    raw_h = AS["raw"][tf_]
    j = AS["opos"][tf_][i]
    if j < 60:                       # tout début des données : pas assez d'historique pour les indicateurs
        return i, None
    hist = raw_h.iloc[max(0, j - HTF_HIST):j]
    o = AS["o"][tf_][i]
    po, ph, pl, pc = AS["part"][tf_][i]
    row = pd.DataFrame({"datetime": [o], "open": [po], "high": [ph], "low": [pl], "close": [pc], "tickvol": [1]})
    try:
        f = add_features(pd.concat([hist, row], ignore_index=True), tf_)
    except Exception:
        return i, None
    if len(f) == 0 or f["datetime"].iloc[-1] != o:
        return i, None
    return i, f.iloc[-1][AS["cols"][tf_]].values.astype(np.float64)

def asof_htf(tf_, m15, raw_h, full_feats):
    """Features de l'unité tf_ telles qu'elles sont à la clôture de chaque bougie M15."""
    T = m15["datetime"] + pd.Timedelta(minutes=15)
    o = (T - pd.Timedelta(minutes=1)).dt.floor(FREQ[tf_])
    g = m15.groupby(o.values)
    part = np.c_[g["open"].transform("first").values, g["high"].cummax().values,
                 g["low"].cummin().values, m15["close"].values]
    cols = [c for c in full_feats.columns if c != "datetime"]
    out = np.full((len(m15), len(cols)), np.nan)
    # bougie HTF qui vient de se fermer pile à T : ce sont ses features "complètes"
    complete = (o + TF_DUR[tf_] == T).values
    lut = full_feats.set_index("datetime")[cols]
    hit = complete & o.isin(lut.index).values
    out[hit] = lut.loc[o[hit]].values
    # bougie HTF en cours : recalcul avec 200 bougies fermées + la bougie partielle
    todo = np.where(~complete)[0]
    AS.update(raw={tf_: raw_h.reset_index(drop=True)}, o={tf_: o.tolist()}, part={tf_: part}, cols={tf_: cols},
              opos={tf_: raw_h["datetime"].values.searchsorted(o.values)})
    jobs = [(tf_, int(i)) for i in todo]
    if os.name == "nt":
        res = map(_asof_job, jobs)
        pool = None
    else:
        pool = Pool(os.cpu_count()); res = pool.imap(_asof_job, jobs, chunksize=256)
    for k, (i, v) in enumerate(res):
        if v is not None: out[i] = v
        if k and k % 100000 == 0: print(f"    {tf_} : {k:,}/{len(jobs):,}", flush=True)
    if pool: pool.close(); pool.join()
    AS.clear()
    return pd.DataFrame(out, columns=cols, index=m15.index)

def build_asof(raw):
    t0 = time.time()
    m15 = raw["M15"].reset_index(drop=True)
    start = os.environ.get("GS_ASOF_FROM")          # test local : ne calcule qu'à partir de cette date
    keep = (m15["datetime"] >= pd.Timestamp(start, tz="UTC")).values if start else np.ones(len(m15), bool)
    base = add_features(raw["M15"], "M15")
    for tf_ in ["M30", "H1", "H4", "Daily"]:
        full = add_features(raw[tf_], tf_)
        sub = m15[keep].reset_index(drop=True)
        h = asof_htf(tf_, sub, raw[tf_], full)
        h["datetime"] = sub["datetime"]
        base = base.merge(h, on="datetime", how="inner")
        print(f"  {tf_} à l'instant T : OK ({(time.time()-t0)/60:.1f} min)", flush=True)
    return cross_features(base).dropna().reset_index(drop=True)

# ── MODE FINETUNE : on reprend TON modèle déployé et on continue son entraînement ──
FT_VAL    = pd.Timestamp(os.environ.get("GS_FT_VAL", "2026-07-01"), tz="UTC")
FT_TEST   = pd.Timestamp(os.environ.get("GS_FT_TEST", "2026-08-15"), tz="UTC")
FT_EPOCHS = int(os.environ.get("GS_FT_EPOCHS", 10))
FT_LR     = float(os.environ.get("GS_FT_LR", 1e-4))
FT_ASOF   = os.environ.get("GS_FEATURES", "asof") == "asof"   # "asof" = multi-timeframes à l'instant T (comme en direct)

def finetune_main():
    t0 = time.time()
    print("=" * 70 + "\n  GOLDSNIPER — réentraînement du modèle existant (fine-tuning)\n" + "=" * 70)
    src = next(p for p in INPUT.rglob("model_xauusd_best.keras"))
    d = src.parent
    model = tf.keras.models.load_model(src, compile=False)
    sc = joblib.load(d / "scaler_xauusd.joblib")          # scaler d'origine : le modèle a appris avec lui
    meta = json.load(open(d / "metadata_xauusd.json"))
    feats = meta["feature_cols"]
    print(f"  modèle chargé : {src} (entraîné le {meta.get('date_trained')})")

    print("\n[1] Données"); m1 = load_m1()
    raw = {tf_: resample(m1, r) for tf_, r in RULES.items()}
    if FT_ASOF:
        print("\n[2] Features MULTI-TIMEFRAMES à l'instant T (bougies M30/H1/H4/Daily en cours incluses)")
        base = build_asof(raw)
    else:
        print("\n[2] Features (même méthode que l'entraînement d'origine)")
        base = build_features(raw, "open")
    base_live = build_features(raw, "close")
    close_m1 = m1.set_index("datetime")["close"]

    def prep(b):
        X = np.nan_to_num(b[feats].values.astype(np.float64), nan=0, posinf=0, neginf=0)   # comme app.py
        X = np.nan_to_num(sc.transform(X), nan=0, posinf=0, neginf=0).astype(np.float32)
        ret, yd, _, _ = targets(close_m1, b["datetime"] + pd.Timedelta(minutes=15), b["m15_atr14"].values)
        return X, ret, yd

    X, ret, yd = prep(base)
    XL, retL, _ = prep(base_live)
    n = len(base); dt = base["datetime"]
    valid = (~np.isnan(ret)) & (np.arange(n) >= SEQ - 1)
    tr = valid & (dt < FT_VAL - pd.Timedelta(days=1)).values
    va = valid & (dt >= FT_VAL).values & (dt < FT_TEST - pd.Timedelta(days=1)).values
    te = valid & (dt >= FT_TEST).values
    teL = np.where(((~np.isnan(retL)) & (base_live["datetime"] >= FT_TEST).values))[0]
    teL = teL[teL >= SEQ - 1]
    print(f"  train {tr.sum():,} | validation {va.sum():,} ({FT_VAL.date()} ->) | test {te.sum():,} ({FT_TEST.date()} ->, jamais vu)")
    offs = np.arange(-SEQ + 1, 1)

    def predict(mdl, Xa, rows):
        out = []
        for s in range(0, len(rows), 2048):
            out.append(mdl.predict(Xa[rows[s:s + 2048][:, None] + offs], verbose=0)["direction"].ravel())
        return np.concatenate(out)

    def evaluate(mdl, label):
        print(f"\n  ===== {label} =====")
        r = np.where(te)[0]
        table("MULTI-TIMEFRAMES À L'INSTANT T (= nouveau direct)" if FT_ASOF else "comme à l'entraînement d'origine",
              ret[r], predict(mdl, X, r))
        table("API ACTUELLE (H1/H4/Daily fermées)", retL[teL], predict(mdl, XL, teL))

    print("\n[3] Score AVANT réentraînement (ton modèle actuel)")
    evaluate(model, "AVANT")

    class W(tf.keras.utils.Sequence):
        def __init__(self, rows, batch, shuffle):
            super().__init__(); self.rows, self.b, self.sh = rows.copy(), batch, shuffle
            if shuffle: RNG.shuffle(self.rows)
        def __len__(self): return int(np.ceil(len(self.rows) / self.b))
        def on_epoch_end(self):
            if self.sh: RNG.shuffle(self.rows)
        def __getitem__(self, k):
            r = self.rows[k * self.b:(k + 1) * self.b]
            z = np.zeros(len(r), np.float32)   # confidence / magnitude : poids 0, on ne réentraîne que la direction
            return X[r[:, None] + offs], {"direction": yd[r], "confidence": z, "magnitude": z}

    print(f"\n[4] Réentraînement (lr={FT_LR}, max {FT_EPOCHS} époques)")
    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR, clipnorm=1.0),
                  loss={"direction": "binary_crossentropy", "confidence": "binary_crossentropy", "magnitude": "mse"},
                  loss_weights={"direction": 1.0, "confidence": 0.0, "magnitude": 0.0},
                  metrics={"direction": [tf.keras.metrics.AUC(name="auc")]})
    model.fit(W(np.where(tr)[0], 256, True), validation_data=W(np.where(va)[0], 1024, False),
              epochs=FT_EPOCHS, verbose=2,
              callbacks=[callbacks.EarlyStopping(monitor="val_direction_auc", mode="max", patience=3,
                                                 restore_best_weights=True)])

    print("\n[5] Score APRÈS réentraînement")
    evaluate(model, "APRÈS")

    print("\n[6] Sauvegarde (même format que l'original : app.py et le robot ne changent pas)")
    model.save(str(OUT / "model_xauusd_best.keras"))
    joblib.dump(sc, OUT / "scaler_xauusd.joblib")
    if FT_ASOF: meta.update(htf_mode="asof", htf_hist=HTF_HIST)
    meta.update(date_trained=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), finetuned_from=meta.get("date_trained"),
                data_end=str(m1.datetime.max()))
    json.dump(meta, open(OUT / "metadata_xauusd.json", "w"), indent=2)
    print(f"  OK -> {OUT}  ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    if os.environ.get("GS_MODE", "train") == "finetune":
        finetune_main()
    else:
        main()
