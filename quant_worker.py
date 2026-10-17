import os

# --- OUTPUT & WARNING SUPPRESSION (DIRECT BIJ DE START) ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import base64
import copy
import gc
import json
import logging
import math
import time
import urllib.error
import urllib.request
import warnings

import ccxt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
logging.getLogger('xgboost').setLevel(logging.CRITICAL)

assets = ['BTC', 'ETH', 'LINK']

# Alle timeframes voorspellen de volgende candle (horizon = 1).
timeframes = {
    '15m': {'interval': '15m', 'limit': 500, 'min_interval_seconds': 14 * 60},
    '1h': {'interval': '1h', 'limit': 500, 'min_interval_seconds': 55 * 60},
    '1d': {'interval': '1d', 'limit': 365, 'min_interval_seconds': 23 * 3600},
    '1w': {'interval': '1w', 'limit': 100, 'min_interval_seconds': 6 * 86400},
    '1m': {'interval': '1M', 'limit': 60, 'min_interval_seconds': 25 * 86400},
}

base_features = [
    'return_1', 'dist_sma20', 'dist_sma50', 'dist_ema9', 'dist_ema21',
    'RSI', 'atr_ratio', 'bb_position', 'OBV_slope', 'VWAP_dist', 'Supertrend_dist'
]

optional_features = [
    'MACD_hist', 'ADX', 'CMF', 'ATR_percentile', 'regime_score', 'smc_zone_dist',
    'lorentzian_dist', 'funding_rate', 'oi_change', 'oi_ratio',
    'bid_ask_imbalance', 'book_pressure', 'taker_buy_ratio', 'btc_trend_gap'
]

MODEL_NAMES = ['xgb', 'cat', 'rf']

# Validatie-instellingen
HOLDOUT_FRACTION = 0.25      # laatste 25% van het trainingsvenster = out-of-sample test
MIN_HOLDOUT = 15             # minimaal aantal test-candles
MIN_TRAIN_ROWS = 25          # minimaal aantal train-candles (voor de holdout)
MIN_FEATURE_ROWS = 60        # feature telt alleen mee als er genoeg geldige rijen zijn

SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

# --- CENTRALE EXCHANGE INSTANTIE VOOR RATE LIMITING ---
exchange = ccxt.kucoin({'enableRateLimit': True})


# =====================================================================
# HELPERS VOOR DATA & CONTEXT (KUCOIN API OPHALEN)
# =====================================================================
def safe_div(num, den):
    if den is None or pd.isna(den) or abs(float(den)) < 1e-12:
        return 0.0
    return float(num) / float(den)

def as_float(x, default=0.0):
    try:
        return float(x) if x is not None and not pd.isna(x) else default
    except Exception:
        return default

def fetch_market_context_from_exchange(exch, symbol):
    """Haalt live funding rate, open interest en orderboek op via de KuCoin API."""
    funding = 0.0
    oi_data = None
    orderbook = None
    
    # Probeer swap/futures symbool formaat voor api-aanroepen (bijv. BTC/USDT:USDT of BTC/USDT)
    swap_symbol = f"{symbol}:USDT" if not symbol.endswith(":USDT") else symbol

    try:
        fr = exch.fetch_funding_rate(swap_symbol)
        if fr and 'fundingRate' in fr:
            funding = float(fr['fundingRate'])
    except Exception:
        try:
            fr = exch.fetch_funding_rate(symbol)
            if fr and 'fundingRate' in fr:
                funding = float(fr['fundingRate'])
        except Exception:
            pass

    try:
        oi = exch.fetch_open_interest(swap_symbol)
        if oi and 'openInterest' in oi:
            val = float(oi['openInterest'])
            oi_data = {'change': [val]*500, 'ratio': [1.0]*500}
    except Exception:
        pass

    try:
        orderbook = exch.fetch_order_book(symbol, limit=20)
    except Exception:
        pass

    return funding, oi_data, orderbook


# =====================================================================
# STATE
# =====================================================================
def default_state():
    return {
        'learning_rate': 0.05,
        'lookback_window': 150,
        'current_regime': 'sideways',
        'regime_features': {
            'trending': base_features.copy(),
            'sideways': base_features.copy(),
        },
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
        'holdout_classifier_accuracy': None,
        'holdout_samples': 0,
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
            if isinstance(loaded, dict):
                state.update(loaded)
        except Exception:
            pass

    if state.get('data_status') in (None, 'new'):
        state['last_trained_timestamp'] = 0

    if not isinstance(state.get('regime_features'), dict):
        state['regime_features'] = {}
    for reg in ('trending', 'sideways'):
        feats = state['regime_features'].get(reg)
        if not isinstance(feats, list) or not feats:
            state['regime_features'][reg] = base_features.copy()
    if not isinstance(state.get('model_weights'), dict):
        state['model_weights'] = {m: 1.0 / len(MODEL_NAMES) for m in MODEL_NAMES}
    for m in MODEL_NAMES:
        state['model_weights'].setdefault(m, 1.0 / len(MODEL_NAMES))
    if not isinstance(state.get('model_errors'), dict):
        state['model_errors'] = {}
    for m in MODEL_NAMES:
        v = state['model_errors'].get(m)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            state['model_errors'][m] = None
    for legacy in ('search_direction', 'max_depth', 'n_estimators',
                   'previous_learning_rate', 'previous_max_depth'):
        state.pop(legacy, None)
    if not isinstance(state.get('baseline_error'), (int, float)) or not math.isfinite(state['baseline_error']) \
            or state.get('data_status') in (None, 'new'):
        state['baseline_error'] = None
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


# =====================================================================
# DATA & FEATURES
# =====================================================================
def fetch_data_safely(symbol, tf_info):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_info['interval'], limit=tf_info['limit'])
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"❌ KuCoin fetch fout voor {symbol}: {e}")
        return pd.DataFrame()


def compute_features(df):
    df = df.copy()
    close = df['close']

    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['EMA_9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()

    df['return_1'] = close.pct_change()
    df['dist_sma20'] = (close - df['SMA_20']) / df['SMA_20']
    df['dist_sma50'] = (close - df['SMA_50']) / df['SMA_50']
    df['dist_ema9'] = (close - df['EMA_9']) / df['EMA_9']
    df['dist_ema21'] = (close - df['EMA_21']) / df['EMA_21']

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - close.shift())
    low_close = np.abs(df['low'] - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    df['atr_ratio'] = df['ATR'] / close

    bb_std = close.rolling(window=20).std()
    bb_upper = df['SMA_20'] + (bb_std * 2)
    bb_lower = df['SMA_20'] - (bb_std * 2)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)

    df['OBV'] = (np.sign(close.diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_slope'] = df['OBV'].diff() / (df['volume'].rolling(20).mean() + 1e-8)

    typical_price = (df['high'] + df['low'] + close) / 3
    vwap = (typical_price * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)
    df['VWAP_dist'] = (close - vwap) / (vwap + 1e-8)

    # Supertrend
    hl2 = (df['high'] + df['low']) / 2
    basic_upper = (hl2 + 3.0 * df['ATR']).to_numpy(copy=True)
    basic_lower = (hl2 - 3.0 * df['ATR']).to_numpy(copy=True)
    close_val = close.to_numpy(copy=True)
    f_upper = basic_upper.copy()
    f_lower = basic_lower.copy()
    trend = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if np.isnan(basic_upper[i]) or np.isnan(f_upper[i - 1]):
            continue
        if basic_upper[i] < f_upper[i - 1] or close_val[i - 1] > f_upper[i - 1]:
            f_upper[i] = basic_upper[i]
        else:
            f_upper[i] = f_upper[i - 1]
        if basic_lower[i] > f_lower[i - 1] or close_val[i - 1] < f_lower[i - 1]:
            f_lower[i] = basic_lower[i]
        else:
            f_lower[i] = f_lower[i - 1]
        if close_val[i] > f_upper[i - 1]:
            trend[i] = 1
        elif close_val[i] < f_lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    supertrend = np.where(trend == 1, f_lower, f_upper)
    df['Supertrend_dist'] = (close - supertrend) / (close + 1e-8)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd_line - macd_signal

    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr14 = true_range.rolling(window=14).sum()
    plus_di = 100 * (plus_dm.rolling(window=14).sum() / (tr14 + 1e-8))
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / (tr14 + 1e-8))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['ADX'] = dx.rolling(window=14).mean()

    mf_multiplier = ((close - df['low']) - (df['high'] - close)) / ((df['high'] - df['low']) + 1e-8)
    df['CMF'] = (mf_multiplier * df['volume']).rolling(window=20).sum() / (df['volume'].rolling(20).sum() + 1e-8)

    df['ATR_percentile'] = df['ATR'].rolling(window=100).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    df['regime_score'] = (df['ADX'] / 100.0) * (1.0 - df['atr_ratio'] / (df['atr_ratio'].rolling(20).mean() + 1e-8))

    body_size = (close - df['open']).abs()
    is_order_block = body_size > (2.0 * body_size.rolling(20).mean())
    recent_ob_price = pd.Series(np.where(is_order_block, (df['high'] + df['low']) / 2, np.nan), index=df.index).ffill()
    df['smc_zone_dist'] = ((close - recent_ob_price) / (close + 1e-8)).fillna(0)

    rolling_max = df['high'].rolling(10).max()
    rolling_min = df['low'].rolling(10).min()
    mid_range = (rolling_max + rolling_min) / 2
    df['lorentzian_dist'] = (close - mid_range) / (rolling_max - rolling_min + 1e-8)

    return df.replace([np.inf, -np.inf], np.nan)


def enrich_market_context(df, funding=None, oi=None, orderbook=None, btc_ref=None):
    out = df.copy()
    out['funding_rate'] = pd.Series(funding, index=out.index) if funding is not None else 0.0

    if oi is not None:
        out['oi_change'] = pd.Series(oi.get('change', [0.0] * len(out)), index=out.index)
        out['oi_ratio'] = pd.Series(oi.get('ratio', [0.0] * len(out)), index=out.index)
    else:
        out['oi_change'] = 0.0
        out['oi_ratio'] = 0.0

    if orderbook is not None:
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            total_bid = sum(float(x[1]) for x in bids[:10])
            total_ask = sum(float(x[1]) for x in asks[:10])
            out['bid_ask_imbalance'] = (best_bid - best_ask) / (best_ask + 1e-8)
            out['book_pressure'] = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
        else:
            out['bid_ask_imbalance'] = 0.0
            out['book_pressure'] = 0.0
    else:
        out['bid_ask_imbalance'] = 0.0
        out['book_pressure'] = 0.0

    out['taker_buy_ratio'] = 0.0
    if btc_ref is not None and 'close' in btc_ref.columns:
        out['btc_trend_gap'] = out['close'].pct_change().fillna(0.0) - btc_ref['close'].pct_change().fillna(0.0)
    else:
        out['btc_trend_gap'] = 0.0

    return out


def get_market_regime(df):
    try:
        if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]):
            return 'trending' if df['ADX'].iloc[-1] > 25 else 'sideways'
    except Exception:
        pass
    return 'sideways'


# =====================================================================
# MODELLEN & VALIDATIE
# =====================================================================
def make_regressors(lr):
    return {
        'xgb': xgb.XGBRegressor(learning_rate=lr, max_depth=4, n_estimators=50, random_state=42, verbosity=0, n_jobs=1),
        'cat': CatBoostRegressor(learning_rate=lr, depth=4, iterations=50, random_seed=42, verbose=0, thread_count=1),
        'rf': RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42, n_jobs=1),
    }


def make_classifier(lr):
    return xgb.XGBClassifier(learning_rate=lr, n_estimators=30, max_depth=3, random_state=42,
                             eval_metric='logloss', verbosity=0, n_jobs=1)


def usable_features(df, feats):
    return [f for f in feats if f in df.columns and df[f].notna().sum() >= MIN_FEATURE_ROWS]


def build_dataset(df, feats, lookback):
    """
    Target = reëel rendement van de VOLGENDE candle (close[t+1]/close[t] - 1).
    De laatste rij heeft nog geen label en wordt gebruikt als voorspel-input.
    """
    d = df.dropna(subset=feats + ['close']).reset_index(drop=True)
    if len(d) < lookback + 5:
        return None
    # Maak X (features) en y (reëel rendement van de volgende candle)
    X = d[feats].iloc[:-1]
    y = (d['close'].shift(-1) / d['close'] - 1.0).iloc[:-1]
    
    # Neem alleen het gekozen lookback-venster voor de training
    X = X.iloc[-lookback:]
    y = y.iloc[-lookback:]
    
    # De allerlaatste rij is de input om de volgende voorspelling mee te doen
    latest = d[feats].iloc[[-1]]
    
    return X, y, latest

    
def evaluate_config(X, y, lr):
    n = len(X)
    h = max(MIN_HOLDOUT, int(n * HOLDOUT_FRACTION))
    if n - h < MIN_TRAIN_ROWS:
        return None
    Xtr, Xte, ytr, yte = X.iloc[:-h] if hasattr(X, 'iloc') else X[:-h], X.iloc[-h:] if hasattr(X, 'iloc') else X[-h:], y[:-h], y[-h:]

    preds, maes = {}, {}
    for name, model in make_regressors(lr).items():
        model.fit(Xtr, ytr)
        p = np.asarray(model.predict(Xte), dtype=float)
        preds[name] = p
        maes[name] = float(np.mean(np.abs(p - yte)))

    inv = {k: 1.0 / (v + 1e-9) for k, v in maes.items()}
    tot = sum(inv.values())
    weights = {k: float(v / tot) for k, v in inv.items()}

    ens = sum(weights[k] * preds[k] for k in preds)
    y_arr = np.asarray(yte)
    ens_mae = float(np.mean(np.abs(ens - y_arr)))
    zero_mae = float(np.mean(np.abs(y_arr))) + 1e-12
    dir_acc = float(np.mean(np.sign(ens) == np.sign(y_arr)))

    clf_acc = None
    ytr_c = (ytr > 0).astype(int)
    if len(np.unique(ytr_c)) == 2:
        try:
            clf = make_classifier(lr)
            clf.fit(Xtr, ytr_c)
            p_up = clf.predict_proba(Xte)[:, 1]
            clf_acc = float(np.mean((p_up > 0.5) == (y_arr > 0)))
        except Exception:
            pass

    return {
        'weights': weights,
        'maes': maes,
        'skill': ens_mae / zero_mae,
        'dir_acc': dir_acc,
        'clf_acc': clf_acc,
        'holdout_n': int(h),
    }


def mutate_config(cfg, regime):
    new = copy.deepcopy(cfg)
    new['learning_rate'] = float(np.clip(cfg['learning_rate'] + np.random.choice([-0.01, 0.01]), 0.01, 0.15))
    new['lookback_window'] = int(np.clip(cfg['lookback_window'] + np.random.choice([-20, 20]), 60, 300))

    feats = list(cfg['features'])
    if np.random.rand() < 0.5:
        addable = [f for f in optional_features if f not in feats]
        removable = [f for f in feats if f in optional_features]
        if addable and (not removable or np.random.rand() < 0.5):
            feats.append(str(np.random.choice(addable)))
        elif removable:
            feats.remove(str(np.random.choice(removable)))
    new['features'] = feats
    return new


def score_config(df, cfg):
    feats = usable_features(df, cfg['features'])
    if not feats:
        return None
    ds = build_dataset(df, feats, cfg['lookback_window'])
    if ds is None:
        return None
    X, y, _ = ds
    res = evaluate_config(X, y, cfg['learning_rate'])
    if res is not None:
        res['features_used'] = feats
    return res


# =====================================================================
# EEN RUN VOOR 1 ASSET/TIMEFRAME (INCL. KUCOIN API OPHAALDATA)
# =====================================================================
def _insufficient(state, now):
    state.update({'data_status': 'insufficient_data', 'signal': 'none', 'signal_probability': 50.0,
                  'last_prediction': None, 'last_prediction_price': None,
                  'last_trained_timestamp': now, 'updated_at': int(now)})
    return state


def process(df_raw, state, now, symbol):
    df_raw = df_raw.iloc[:-1].reset_index(drop=True)
    if len(df_raw) < 60:
        return _insufficient(state, now), False

    df = compute_features(df_raw)
    
    # --- HIER WORDEN LIVE MARKTDATA EN API-CONTEXT OPGEHAALD ---
    funding, oi, ob = fetch_market_context_from_exchange(exchange, symbol)
    df = enrich_market_context(df, funding=funding, oi=oi, orderbook=ob)
    # ------------------------------------------------------------

    regime = get_market_regime(df)
    state['current_regime'] = regime

    best_cfg = {
        'learning_rate': float(state['learning_rate']),
        'lookback_window': int(state['lookback_window']),
        'features': list(state['regime_features'].get(regime, base_features)),
    }
    best_res = score_config(df, best_cfg)

    cand_cfg = mutate_config(best_cfg, regime)
    cand_res = score_config(df, cand_cfg)

    chosen_cfg, chosen_res = best_cfg, best_res
    if cand_res is not None and (best_res is None or cand_res['skill'] < best_res['skill']):
        chosen_cfg, chosen_res = cand_cfg, cand_res
        print(f"   ↗ nieuwe config geaccepteerd (skill {cand_res['skill']:.3f}"
              + (f" < {best_res['skill']:.3f})" if best_res else ")"))

    if chosen_res is None:
        return _insufficient(state, now), False

    feats = chosen_res['features_used']
    ds = build_dataset(df, feats, chosen_cfg['lookback_window'])
    if ds is None:
        return _insufficient(state, now), False

    X, y, latest = ds
    lr = chosen_cfg['learning_rate']
    weights = chosen_res['weights']

    pred_ret = {}
    # Maak er een 2D NumPy array van zodat .reshape() wel werkt
    latest_arr = latest.to_numpy() if hasattr(latest, 'to_numpy') else np.asarray(latest)
    if latest_arr.ndim == 1:
        latest_arr = latest_arr.reshape(1, -1)

    for name, model in make_regressors(lr).items():
        model.fit(X, y)
        pred_ret[name] = float(model.predict(latest_arr)[0])
    ens_ret = float(sum(weights[k] * pred_ret[k] for k in pred_ret))

    prob_up = 50.0
    y_cls = (y > 0).astype(int)
    if len(np.unique(y_cls)) == 2:
        try:
            clf = make_classifier(lr)
            clf.fit(X, y_cls)
            prob_up = float(clf.predict_proba(latest_arr)[0][1] * 100)
        except Exception:
            prob_up = 50.0
    last_close = float(df['close'].iloc[-1])
    atr = df['ATR'].iloc[-1]
    atr = float(atr) if pd.notna(atr) and atr > 0 else last_close * 0.02

    reg_up = ens_ret > 0
    clf_up = prob_up >= 50.0
    if reg_up and clf_up:
        signal, sig_prob = 'long', prob_up
    elif (not reg_up) and (not clf_up):
        signal, sig_prob = 'short', 100.0 - prob_up
    else:
        signal, sig_prob = 'none', 50.0

    sign = 1 if reg_up else -1

    maes_returns = chosen_res.get('maes', {}) if chosen_res else {}
    model_errors_returns = {}
    model_errors_pct = {}
    model_errors_usd = {}
    for k, v in maes_returns.items():
        if v is None:
            model_errors_returns[k] = None
            model_errors_pct[k] = None
            model_errors_usd[k] = None
        else:
            model_errors_returns[k] = float(v)
            model_errors_pct[k] = float(v * 100.0)
            model_errors_usd[k] = float(v * last_close)

    model_errors_readable = {
        k: f"{model_errors_usd.get(k):.2f} USD ({model_errors_pct.get(k):.2f}%)"
        if model_errors_usd.get(k) is not None else None
        for k in model_errors_returns.keys()
    }

    state.update({
        'learning_rate': lr,
        'lookback_window': chosen_cfg['lookback_window'],
        'active_features': feats,
        'model_weights': weights,
        'model_errors': model_errors_returns,
        'model_errors_pct': model_errors_pct,
        'model_errors_usd': model_errors_usd,
        'model_errors_readable': model_errors_readable,
        'baseline_error': chosen_res['skill'],
        'holdout_direction_accuracy': chosen_res['dir_acc'],
        'holdout_classifier_accuracy': chosen_res['clf_acc'],
        'holdout_samples': chosen_res['holdout_n'],
        'last_close': last_close,
        'last_prediction': last_close * (1.0 + ens_ret),             # <--- OPGELOST: ens_ret is al decimaal (bijv -0.0025 voor -0.25%)
        'last_prediction_price': last_close * (1.0 + ens_ret),
        'predicted_return_pct': ens_ret * 100.0,                    # <--- Dit wordt netjes het reële percentage (bijv -0.25%)
        'direction_probability': prob_up,
        'signal': signal,
        'signal_probability': sig_prob,
        'atr': atr,
        'atr_pct': atr / last_close * 100.0,
        'stop_loss': (last_close - sign * SL_ATR_MULT * atr) if signal != 'none' else 0.0,
        'take_profit': (last_close + sign * TP_ATR_MULT * atr) if signal != 'none' else 0.0,
        'last_candle_ts': int(pd.Timestamp(df['Date'].iloc[-1]).timestamp()) if 'Date' in df else 0,
        'data_status': 'ok',
        'last_trained_timestamp': now,
        'updated_at': int(now),
    })
    state['regime_features'][regime] = chosen_cfg['features']
    return state, True


# =====================================================================
# GITHUB SYNC
# =====================================================================
def _gh_request(method, path, token, payload=None):
    url = f'https://api.github.com/repos/{GITHUB_REPO}/{path}'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'quant-worker',
    }
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode('utf-8'))


def push_to_github(files):
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("❌ Geen GITHUB_TOKEN gevonden in environment variables!")
        return
    if not files:
        print("ℹ️ Geen gewijzigde state-bestanden om te pushen.")
        return

    pushed = 0
    for fname in files:
        try:
            with open(fname, 'rb') as f:
                local = f.read()
            for attempt in range(2):
                sha = None
                try:
                    remote = _gh_request('GET', f'contents/{fname}?ref={GITHUB_BRANCH}', token)
                    sha = remote.get('sha')
                    if base64.b64decode(remote.get('content', '')) == local:
                        break
                except urllib.error.HTTPError as e:
                    if e.code != 404:
                        raise
                payload = {
                    'message': f'Auto-update {fname} [skip ci]',
                    'content': base64.b64encode(local).decode('ascii'),
                    'branch': GITHUB_BRANCH,
                }
                if sha:
                    payload['sha'] = sha
                try:
                    _gh_request('PUT', f'contents/{fname}', token, payload)
                    pushed += 1
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (409, 422) and attempt == 0:
                        continue
                    raise
        except Exception as e:
            print(f"❌ Push van {fname} mislukt: {e}")
    print(f"✅ {pushed} bestand(en) naar GitHub gepusht.")


# =====================================================================
# MAIN
# =====================================================================
def main():
    written = []
    for asset in assets:
        symbol = f'{asset}/USDT'
        for tf_name, tf_info in timeframes.items():
            state_file = f'model_state_{asset.lower()}_{tf_name}.json'
            try:
                now = time.time()
                state = load_or_init_state(state_file)
                if now - state.get('last_trained_timestamp', 0) < tf_info['min_interval_seconds']:
                    continue

                df_raw = fetch_data_safely(symbol, tf_info)
                if df_raw is None or df_raw.empty:
                    time.sleep(1.0)
                    continue

                state, ok = process(df_raw, state, now, symbol)
                save_state(state_file, state)
                written.append(state_file)
                if ok:
                    print(f"✔ {asset} {tf_name}: signaal={state['signal']} "
                          f"P(up)={state['direction_probability']:.1f}% "
                          f"holdout-richting={state['holdout_direction_accuracy']:.2f} skill={state['baseline_error']:.3f}")
                else:
                    print(f"⚠️ {asset} {tf_name}: onvoldoende data ({state.get('data_status')})")
                gc.collect()
                time.sleep(4.0)
            except Exception as e:
                print(f"❌ Fout bij {asset} ({tf_name}): {e}")
                gc.collect()
                time.sleep(4.0)

    push_to_github(written)


if __name__ == '__main__':
    main()
