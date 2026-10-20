import os

# --- OUTPUT & WARNING SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import json
import logging
import math
import subprocess
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from github_utils import github_put
from market_utils import fetch_ohlcv_paginated
from asset_utils import get_combined_trading_assets

warnings.filterwarnings('ignore')
logging.getLogger('xgboost').setLevel(logging.CRITICAL)


# --- Mini Webserver voor Render (Voorkomt 'No open ports detected' crash) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<html><body><h2>📊 V20 Quant Worker is online</h2></body></html>".encode("utf-8"))

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()


def push_file_to_github(filename):
    if not os.environ.get('GITHUB_TOKEN') or not os.path.exists(filename):
        return
    with open(filename, 'rb') as f:
        content = f.read()
    if github_put(filename, content, agent='v20-quant-worker'):
        print(f"☁️ Bestand {filename} succesvol gepusht naar GitHub.")


# Uitsluitend 4h tijdsframe (1d is verwijderd omdat de trader alleen 4h leest)
timeframes = {
    '4h': {'interval': '4h', 'limit': 1500, 'lookback': 1200, 'min_interval_seconds': 3 * 3600},
}

# Slanke, niet-overfitted feature set (momentum en volatiliteit)
base_features = [
    'return_1', 'return_3', 'RSI', 'atr_ratio', 'bb_position', 'vol_ratio'
]

MODEL_NAMES = ['xgb', 'cat', 'rf']
LEARNING_RATE = 0.03

LABEL_HORIZON = 3          
EMBARGO = LABEL_HORIZON    
OOS_FRACTION = 0.50        
WF_FOLDS = 4
MIN_FOLD_SIZE = 25
MIN_TRAIN_ROWS = 120
MIN_FEATURE_ROWS = 100
MIN_LOOKBACK = 300

QUANT_MIN_Z = float(os.environ.get('QUANT_MIN_Z', 1.28))       
QUANT_MIN_JOINT_N = int(os.environ.get('QUANT_MIN_JOINT_N', 30))

SL_ATR_MULT = 2.5
TP_ATR_MULT = 5.0

exchange = ccxt.kucoinfutures({
    'enableRateLimit': True,
})

STATE_VERSION = 8  

# Exacte 4h candle timing en buffer
CANDLE_SECONDS = 4 * 3600
TRAIN_DELAY_SECONDS = 120  # Wacht 2 min na sluiting zodat de candle er zeker is


def seconds_until_next_run():
    now = time.time()
    next_close = (int(now // CANDLE_SECONDS) + 1) * CANDLE_SECONDS
    return max(5, next_close + TRAIN_DELAY_SECONDS - now)


def default_state():
    return {
        'state_version': STATE_VERSION,
        'current_regime': 'trending',
        'active_features': base_features.copy(),
        'last_trained_timestamp': 0,
        'model_weights': {m: 1.0 / len(MODEL_NAMES) for m in MODEL_NAMES},
        'model_errors': {m: None for m in MODEL_NAMES},
        'baseline_error': None,
        'last_prediction': None,
        'last_prediction_price': None,
        'predicted_return_pct': 0.0,
        'direction_probability': 50.0,
        'signal': 'none',
        'signal_probability': 50.0,
        'holdout_direction_accuracy': None,
        'holdout_direction_z': None,
        'holdout_classifier_accuracy': None,
        'holdout_samples': 0,
        'joint_signal_accuracy': None,
        'joint_signal_samples': 0,
        'joint_signal_z': None,
        'funding_rate': 0.0,
        'open_interest': 0.0,
        'atr': 0.0,
        'atr_pct': 0.0,
        'last_close': None,
        'last_candle_ts': 0,
        'data_status': 'new',
        'stop_loss': 0.0,
        'take_profit': 0.0,
        'updated_at': 0,
    }


def load_or_init_state(state_file):
    state = default_state()
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)

            if isinstance(loaded, dict) and loaded.get('state_version', 1) == STATE_VERSION:
                state.update(loaded)
            else:
                print(f"🔄 Oude state gedetecteerd voor {state_file}. Reset naar v{STATE_VERSION} defaults.")
        except Exception as e:
            print(f"⚠️ State {state_file} onleesbaar ({e}); start met defaults.")

    if state.get('data_status') in (None, 'new'):
        state['last_trained_timestamp'] = 0
    return state


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_state(state_file, state):
    tmp = state_file + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(state), f, indent=4, allow_nan=False)
    os.replace(tmp, state_file)


def fetch_data_safely(symbol, tf_info):
    try:
        df = fetch_ohlcv_paginated(exchange, symbol, tf_info['interval'], tf_info['limit'])
        time.sleep(1.0)
        if len(df) >= 5:
            return df
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_info['interval'], limit=200)
        time.sleep(1.0)
        if not ohlcv or len(ohlcv) < 5:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df.iloc[:-1].reset_index(drop=True)
    except Exception as e:
        print(f"❌ KuCoin Futures fetch fout voor {symbol}: {e}")
        time.sleep(1.0)
        return pd.DataFrame()


def fetch_futures_market_context(symbol):
    funding_rate = 0.0
    open_interest = 0.0
    try:
        fr_data = exchange.fetch_funding_rate(symbol)
        time.sleep(1.0)
        if fr_data and 'fundingRate' in fr_data:
            funding_rate = float(fr_data['fundingRate'] or 0.0)
    except Exception as e:
        print(f"⚠️ Kon funding rate niet ophalen voor {symbol}: {e}")
        time.sleep(1.0)

    try:
        oi_data = exchange.fetch_open_interest(symbol)
        time.sleep(1.0)
        if oi_data and ('openInterestAmount' in oi_data or 'openInterest' in oi_data):
            open_interest = float(oi_data.get('openInterestAmount') or oi_data.get('openInterest') or 0.0)
    except Exception as e:
        print(f"⚠️ Kon open interest niet ophalen voor {symbol}: {e}")
        time.sleep(1.0)

    return funding_rate, open_interest


def compute_features(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    sma20 = close.rolling(window=20).mean()
    df['return_1'] = close.pct_change()
    df['return_3'] = close.pct_change(3)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    df['atr_ratio'] = (atr / (close + 1e-8)) * 100.0

    bb_std = close.rolling(window=20).std()
    bb_upper = sma20 + (bb_std * 2.0)
    bb_lower = sma20 - (bb_std * 2.0)
    df['bb_position'] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-8)

    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    df['ATR'] = atr

    return df.replace([np.inf, -np.inf], np.nan)


def get_market_regime(df):
    try:
        if 'atr_ratio' in df.columns and not pd.isna(df['atr_ratio'].iloc[-1]):
            return 'trending' if df['atr_ratio'].iloc[-1] > 1.5 else 'sideways'
    except Exception:
        pass
    return 'trending'


def make_regressors():
    lr = LEARNING_RATE
    return {
        'xgb': xgb.XGBRegressor(learning_rate=lr, max_depth=3, n_estimators=40, random_state=42, verbosity=0, n_jobs=1),
        'cat': CatBoostRegressor(learning_rate=lr, depth=3, iterations=40, random_seed=42, verbose=0, thread_count=1),
        'rf': RandomForestRegressor(n_estimators=25, max_depth=3, random_state=42, n_jobs=1),
    }


def make_classifier():
    return xgb.XGBClassifier(learning_rate=LEARNING_RATE, n_estimators=25, max_depth=2, random_state=42,
                             eval_metric='logloss', verbosity=0, n_jobs=1)


def usable_features(df, feats):
    return [f for f in feats if f in df.columns and df[f].notna().sum() >= MIN_FEATURE_ROWS]


def build_dataset(df, feats, lookback):
    d = df.dropna(subset=feats + ['close', 'ATR']).reset_index(drop=True)
    lb = min(int(lookback), len(d) - LABEL_HORIZON - 10)
    if lb < MIN_LOOKBACK:
        return None
    X = d[feats].iloc[:-LABEL_HORIZON]
    y = (d['close'].shift(-LABEL_HORIZON) / d['close'] - 1.0).iloc[:-LABEL_HORIZON]
    X = X.iloc[-lb:]
    y = y.iloc[-lb:]
    latest = d[feats].iloc[[-1]]
    return X, y, latest


def _z_vs_half(acc, n_obs):
    n_eff = n_obs / float(LABEL_HORIZON)
    if acc is None or n_eff <= 0:
        return None
    return float((acc - 0.5) / math.sqrt(0.25 / n_eff))


def walk_forward_eval(X, y):
    n = len(X)
    oos_n = int(n * OOS_FRACTION)
    fold = oos_n // WF_FOLDS
    if fold < MIN_FOLD_SIZE or (n - oos_n - EMBARGO) < MIN_TRAIN_ROWS:
        return None

    starts = [n - oos_n + k * fold for k in range(WF_FOLDS)]
    ends = starts[1:] + [n]

    preds = {name: [] for name in MODEL_NAMES}
    ys, ps = [], []
    for s, e in zip(starts, ends):
        tr_end = s - EMBARGO
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = X.iloc[s:e], y.iloc[s:e]
        for name, model in make_regressors().items():
            model.fit(Xtr, ytr)
            preds[name].append(np.asarray(model.predict(Xte), dtype=float))
        ytr_c = (ytr > 0).astype(int)
        if ytr_c.nunique() == 2:
            clf = make_classifier()
            clf.fit(Xtr, ytr_c)
            ps.append(clf.predict_proba(Xte)[:, 1])
        else:
            ps.append(np.full(len(Xte), np.nan))
        ys.append(np.asarray(yte, dtype=float))

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    maes = {k: float(np.mean(np.abs(v - y_all))) for k, v in preds.items()}

    inv = {k: 1.0 / (v + 1e-9) for k, v in maes.items()}
    tot = sum(inv.values())
    weights = {k: float(v / tot) for k, v in inv.items()}

    ens = sum(weights[k] * preds[k] for k in preds)
    ens_mae = float(np.mean(np.abs(ens - y_all)))
    zero_mae = float(np.mean(np.abs(y_all))) + 1e-12
    dir_acc = float(np.mean(np.sign(ens) == np.sign(y_all)))

    valid_p = ~np.isnan(p_all)
    clf_acc = float(np.mean((p_all[valid_p] > 0.5) == (y_all[valid_p] > 0))) if valid_p.any() else None

    long_m = (ens > 0) & (p_all >= 0.52)
    short_m = (ens < 0) & (p_all <= 0.48)
    sel = long_m | short_m
    joint_n = int(sel.sum())
    joint_acc = float(np.mean(np.sign(ens[sel]) == np.sign(y_all[sel]))) if joint_n else None

    return {
        'weights': weights,
        'maes': maes,
        'skill': ens_mae / zero_mae,
        'dir_acc': dir_acc,
        'dir_z': _z_vs_half(dir_acc, len(y_all)),
        'clf_acc': clf_acc
