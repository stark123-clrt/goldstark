"""
GoldSniper XAUUSD API v4.0
- Micro-recalcul toutes les 1 minute (au lieu de 15)
- Signal principal a chaque bougie M15 fermee
- Micro-checks entre les bougies pour detecter retournements rapides
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")
import ta
import asyncio
import websockets
import threading
import traceback
from datetime import datetime, timezone

# =====================================================================
# CHARGEMENT MODELE
# =====================================================================
import tensorflow as tf
model   = tf.keras.models.load_model("model_xauusd_best.keras")
scaler  = joblib.load("scaler_xauusd.joblib")
with open("metadata_xauusd.json") as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]
SEQ_LENGTH   = meta["seq_length"]
N_FEATURES   = meta["n_features"]

app = FastAPI(title="GoldSniper XAUUSD API", version="4.0")

# =====================================================================
# CONFIGURATION
# =====================================================================
DERIV_API_KEY = os.environ.get("DERIV_API_KEY", "rjs4NEQHKlcWSAf")
DERIV_WS_URL  = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

M15_SECONDS        = 900
M3_SECONDS         = 60       # Micro-recalcul toutes les 1 minute
WAIT_AFTER_CLOSE   = 8
MAX_WAIT_FOR_DATA  = 60

MAX_CONNECT_RETRIES = 5
MAX_FETCH_RETRIES   = 3
RETRY_BASE_DELAY    = 2
WS_TIMEOUT          = 20

DERIV_GRANULARITY = {"M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "Daily": 86400}
HISTORY_DAYS = {"M15": 5, "M30": 10, "H1": 15, "H4": 60, "Daily": 365}

MERGE_TOLERANCES = {
    "M30":  pd.Timedelta(hours=2),
    "H1":   pd.Timedelta(hours=4),
    "H4":   pd.Timedelta(days=2),
    "Daily": pd.Timedelta(days=5),
}

# =====================================================================
# MODELES
# =====================================================================
class Candle(BaseModel):
    epoch: int
    open:  float
    high:  float
    low:   float
    close: float

class PredictRequest(BaseModel):
    m15: List[Candle]; m30: List[Candle]; h1: List[Candle]
    h4:  List[Candle]; daily: List[Candle]

# =====================================================================
# SIGNAL CACHE
# =====================================================================
_signal_lock = threading.Lock()
_signal_cache: Dict = {
    "status": "waiting", "signal": None, "probability": None, "confidence": None,
    "timestamp": None, "valid_until": None, "bar_time": None,
    "message": "Demarrage en cours...", "compute_ms": None,
    "check_type": None,   # "M15" = signal principal, "M3" = micro-check
}

def _update_cache(**kw):
    with _signal_lock:
        _signal_cache.update(kw)

def _get_cache() -> Dict:
    with _signal_lock:
        return dict(_signal_cache)

def _log(msg: str):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

# =====================================================================
# UTILITAIRES TEMPORELS
# =====================================================================
def get_expected_closed_bar_epoch() -> int:
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    current_bar_start = now_epoch - (now_epoch % M15_SECONDS)
    return current_bar_start - M15_SECONDS

def seconds_until_next_m15_close() -> float:
    now = datetime.now(timezone.utc).timestamp()
    current_bar_start = now - (now % M15_SECONDS)
    next_close = current_bar_start + M15_SECONDS
    return max(next_close - now, 0)

def seconds_until_next_m3() -> float:
    """Secondes jusqu'au prochain cycle de 1 minute."""
    now = datetime.now(timezone.utc).timestamp()
    current_m3_start = now - (now % M3_SECONDS)
    next_m3 = current_m3_start + M3_SECONDS
    return max(next_m3 - now, 0)

def is_m15_boundary() -> bool:
    """Verifie si on est juste apres une fermeture de bougie M15."""
    now = datetime.now(timezone.utc).timestamp()
    seconds_since_m15 = now % M15_SECONDS
    return seconds_since_m15 < 30  # Dans les 30 premieres secondes apres fermeture M15

def epoch_to_str(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

# =====================================================================
# CONNEXION DERIV
# =====================================================================
async def connect_deriv():
    last_error = None
    for attempt in range(MAX_CONNECT_RETRIES):
        ws = None
        try:
            _log(f"Connexion Deriv ({attempt+1}/{MAX_CONNECT_RETRIES})...")
            ws = await asyncio.wait_for(
                websockets.connect(DERIV_WS_URL, ping_interval=20, ping_timeout=10, close_timeout=5),
                timeout=WS_TIMEOUT
            )
            await ws.send(json.dumps({"ping": 1}))
            await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT)
            await ws.send(json.dumps({"authorize": DERIV_API_KEY}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
            if "error" in auth:
                raise Exception(f"Auth: {auth['error']['message']}")
            _log("Connexion Deriv OK")
            return ws
        except Exception as e:
            last_error = e
            if ws:
                try: await ws.close()
                except: pass
            delay = min(RETRY_BASE_DELAY * (2 ** attempt), 30)
            _log(f"Echec: {e}. Retry dans {delay}s...")
            await asyncio.sleep(delay)
    raise Exception(f"Connexion impossible apres {MAX_CONNECT_RETRIES} tentatives: {last_error}")


async def fetch_candles(ws, tf_name: str, include_current: bool = False) -> List[Candle]:
    """
    Recupere les bougies par date de debut.
    include_current=True : garde la bougie en cours (pour micro-check)
    include_current=False : supprime la derniere bougie (signal M15 classique)
    """
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    start_epoch = now_epoch - (HISTORY_DAYS[tf_name] * 86400)

    req = {
    
        "ticks_history": "frxXAUUSD",
        "adjust_start_time": 1,
        "start": start_epoch,
        "end": "latest",
        "granularity": DERIV_GRANULARITY[tf_name],
        "style": "candles",
    }

    last_error = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            await ws.send(json.dumps(req))
            raw = await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT)
            resp = json.loads(raw)
            if "error" in resp:
                raise Exception(f"Deriv {tf_name}: {resp['error']['message']}")
            candles_raw = resp.get("candles", [])
            if len(candles_raw) < 5:
                raise Exception(f"{tf_name}: seulement {len(candles_raw)} bougies")

            # Pour le micro-check M15, on garde la bougie en cours
            # Pour les autres TF et le signal principal, on la supprime
            if not include_current:
                candles_raw = candles_raw[:-1]

            candles = [
                Candle(epoch=int(c["epoch"]), open=float(c["open"]),
                       high=float(c["high"]), low=float(c["low"]), close=float(c["close"]))
                for c in candles_raw
            ]
            return candles
        except Exception as e:
            last_error = e
            _log(f"  {tf_name} erreur ({attempt+1}): {e}")
            if attempt < MAX_FETCH_RETRIES - 1:
                await asyncio.sleep(2)
    raise Exception(f"{tf_name} echec: {last_error}")


async def fetch_all_candles(is_micro_check: bool = False) -> Dict[str, List[Candle]]:
    """
    Recupere toutes les TF.
    is_micro_check=True : inclut la bougie M15 en cours (pas encore fermee)
    is_micro_check=False : bougies fermees uniquement (signal principal)
    """
    ws = await connect_deriv()
    try:
        all_candles = {}
        for tf in ["M15", "M30", "H1", "H4", "Daily"]:
            # Pour le micro-check, on inclut la bougie M15 en cours
            include = (is_micro_check and tf == "M15")
            all_candles[tf] = await fetch_candles(ws, tf, include_current=include)
            _log(f"  {tf}: {len(all_candles[tf])} bougies"
                 f"{' (avec bougie en cours)' if include else ''}")
        return all_candles
    finally:
        try: await ws.close()
        except: pass


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================
def candles_to_df(candles, tf_name):
    rows = [{"epoch": c.epoch, "open": c.open, "high": c.high, "low": c.low, "close": c.close} for c in candles]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df["tickvol"] = 1
    return df[["datetime","open","high","low","close","tickvol"]].sort_values("datetime").reset_index(drop=True)

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

def build_features(dfs):
    tf_dfs={}
    for tf_name,df_raw in dfs.items():
        tf_dfs[tf_name]=add_features(df_raw,tf_name)

    base=tf_dfs["M15"].copy()

    for tf_name in ["M30","H1","H4","Daily"]:
        base=pd.merge_asof(
            base, tf_dfs[tf_name],
            on="datetime", direction="backward",
            tolerance=MERGE_TOLERANCES[tf_name]
        )

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

    base = base.dropna().reset_index(drop=True)
    return base

# =====================================================================
# PREDICTION
# =====================================================================
def run_prediction(all_candles: Dict[str,List[Candle]]) -> float:
    dfs={tf:candles_to_df(c,tf) for tf,c in all_candles.items()}
    base=build_features(dfs)

    if len(base)<SEQ_LENGTH:
        raise Exception(f"Pas assez: {len(base)} lignes (besoin {SEQ_LENGTH})")

    available=[c for c in feature_cols if c in base.columns]
    X=base[available].tail(SEQ_LENGTH).values.astype(np.float32)
    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    if len(available)<N_FEATURES:
        full=np.zeros((SEQ_LENGTH,N_FEATURES),dtype=np.float32)
        for i,col in enumerate(feature_cols):
            if col in available: full[:,i]=X[:,available.index(col)]
        X=full
    X_scaled=scaler.transform(X)
    X_scaled=np.nan_to_num(X_scaled,nan=0.0,posinf=0.0,neginf=0.0)
    X_input=X_scaled.reshape(1,SEQ_LENGTH,N_FEATURES)
    preds=model.predict(X_input,verbose=0)
    return {
        "direction": float(preds["direction"].flatten()[0]),
        "confidence": float(preds["confidence"].flatten()[0])
    }

# =====================================================================
# PIPELINE — Signal principal (M15) et Micro-check (M3)
# =====================================================================
async def compute_signal(check_type: str = "M15"):
    """
    check_type = "M15" : signal principal apres fermeture bougie M15
    check_type = "M3"  : micro-verification avec bougie M15 en cours
    """
    t_start = time.time()
    is_micro = (check_type == "M3")

    _update_cache(status="computing",
                  message=f"{'Micro-check' if is_micro else 'Signal M15'}...")
    try:
        _log(f"=== {'MICRO-CHECK M3' if is_micro else 'SIGNAL M15'} ===")

        all_candles = await fetch_all_candles(is_micro_check=is_micro)

        result = run_prediction(all_candles)
        prob = result["direction"]
        conf = result["confidence"]
        signal = "BUY" if prob >= 0.66 else ("SELL" if prob <= 0.34 else "NEUTRAL")

        now = datetime.now(timezone.utc)
        compute_ms = int((time.time() - t_start) * 1000)

        # Pour le micro-check, valid_until = prochain micro-check (1 min)
        # Pour le signal M15, valid_until = prochaine bougie M15
        if is_micro:
            valid_epoch = int(now.timestamp()) + M3_SECONDS + 30
        else:
            expected = get_expected_closed_bar_epoch()
            valid_epoch = expected + M15_SECONDS + M15_SECONDS

        valid_until = datetime.fromtimestamp(valid_epoch, tz=timezone.utc)

        _update_cache(
            status="ready", signal=signal, probability=round(prob, 6), confidence=round(conf, 6),
            timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            valid_until=valid_until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            bar_time=epoch_to_str(get_expected_closed_bar_epoch()),
            message=f"{'Micro-check' if is_micro else 'Signal M15'} pret",
            compute_ms=compute_ms,
            check_type=check_type,
        )
        _log(f"=== {'M3' if is_micro else 'M15'}: {signal} | prob={prob:.6f} | conf={conf:.6f} | {compute_ms}ms ===")

    except Exception as e:
        compute_ms = int((time.time() - t_start) * 1000)
        _log(f"=== ERREUR {check_type}: {str(e)[:200]} ===")
        # Pour un micro-check echoue, on ne remplace pas un signal M15 valide
        if not is_micro:
            _update_cache(status="error", message=f"Erreur: {str(e)[:200]}", compute_ms=compute_ms)
        else:
            _log(f"Micro-check echoue, signal M15 precedent conserve")

# =====================================================================
# SCHEDULER — Toutes les 1 minute
# =====================================================================
def _scheduler_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    last_computed_m15 = None

    _log("========== SCHEDULER v4.0 — Micro-recalcul 1min ==========")

    # Premier calcul
    try:
        loop.run_until_complete(compute_signal("M15"))
        last_computed_m15 = get_expected_closed_bar_epoch()
    except Exception as e:
        _log(f"Erreur premier calcul: {e}")

    while True:
        try:
            # Attendre le prochain cycle de 1 minute
            wait_secs = seconds_until_next_m3() + WAIT_AFTER_CLOSE
            next_time = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + wait_secs, tz=timezone.utc
            ).strftime("%H:%M:%S")
            _log(f"Attente {wait_secs:.0f}s -> prochain check ~{next_time} UTC")
            time.sleep(wait_secs)

            # Determiner le type de check
            expected_m15 = get_expected_closed_bar_epoch()

            if expected_m15 != last_computed_m15:
                # Nouvelle bougie M15 fermee → signal principal
                check_type = "M15"
                _log(f"Nouvelle bougie M15: {epoch_to_str(expected_m15)}")
            else:
                # Pas de nouvelle bougie M15 → micro-check
                check_type = "M3"

            # Executer le calcul
            success = False
            for retry in range(3):
                try:
                    loop.run_until_complete(compute_signal(check_type))
                    if _get_cache()["status"] == "ready":
                        success = True
                        if check_type == "M15":
                            last_computed_m15 = expected_m15
                        break
                    raise Exception(_get_cache().get("message", "Calcul echoue"))
                except Exception as e:
                    _log(f"Retry {retry+1}/3: {e}")
                    if retry < 2: time.sleep(5)

            if not success and check_type == "M15":
                _log(f"ECHEC signal M15 {epoch_to_str(expected_m15)}")
                _update_cache(status="error",
                              message=f"Echec M15 {epoch_to_str(expected_m15)}")

        except Exception as e:
            _log(f"ERREUR SCHEDULER: {e}")
            time.sleep(30)

_scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="m3_scheduler")
_scheduler_thread.start()

# =====================================================================
# ENDPOINTS
# =====================================================================
@app.get("/signal")
def get_signal():
    cache = _get_cache()
    if cache["status"] == "computing": cache["retry_in"] = 3
    return cache

@app.get("/signal/force")
def force_refresh():
    def _run():
        l = asyncio.new_event_loop(); asyncio.set_event_loop(l)
        l.run_until_complete(compute_signal("M15"))
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "computing", "message": "Recalcul force M15"}

@app.get("/health")
def health():
    cache = _get_cache(); now = datetime.now(timezone.utc)
    return {
        "api": "GoldSniper v4.0 (Micro-recalcul 1min)",
        "time_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_features": N_FEATURES, "seq_length": SEQ_LENGTH,
        "signal_status": cache["status"],
        "check_type": cache.get("check_type"),
        "last_signal": cache.get("timestamp"),
        "last_bar": cache.get("bar_time"),
        "last_compute_ms": cache.get("compute_ms"),
        "next_m3_in_s": int(seconds_until_next_m3() + WAIT_AFTER_CLOSE),
        "next_m15_in_s": int(seconds_until_next_m15_close() + WAIT_AFTER_CLOSE),
        "scheduler_alive": _scheduler_thread.is_alive(),
    }

@app.get("/")
def root():
    return {"status": "GoldSniper XAUUSD API v4.0 (Micro-recalcul 1min)", "features": N_FEATURES}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        dfs = {"M15": candles_to_df(request.m15, "M15"), "M30": candles_to_df(request.m30, "M30"),
               "H1": candles_to_df(request.h1, "H1"), "H4": candles_to_df(request.h4, "H4"),
               "Daily": candles_to_df(request.daily, "Daily")}
        base = build_features(dfs)
        if len(base) < SEQ_LENGTH: return {"error": f"Pas assez de lignes: {len(base)}"}
        available = [c for c in feature_cols if c in base.columns]
        X = base[available].tail(SEQ_LENGTH).values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if len(available) < N_FEATURES:
            full = np.zeros((SEQ_LENGTH, N_FEATURES), dtype=np.float32)
            for i, col in enumerate(feature_cols):
                if col in available: full[:, i] = X[:, available.index(col)]
            X = full
        X_scaled = scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_input = X_scaled.reshape(1, SEQ_LENGTH, N_FEATURES)
        preds = model.predict(X_input, verbose=0)
        return {
            "probability": round(float(preds["direction"].flatten()[0]), 6),
            "confidence": round(float(preds["confidence"].flatten()[0]), 6)
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


import uvicorn

if __name__ == "__main__":
    # Render définit la variable d'environnement PORT
    port = int(os.environ.get("PORT", 10000))
    # On lance uvicorn sur 0.0.0.0 pour qu'il soit accessible
    uvicorn.run(app, host="0.0.0.0", port=port)