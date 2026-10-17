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
    'MACD_hist', 'ADX', 'CMF', 'ATR_percentile', 'regime_score', 'smc_zone_dist', 'lorentzian_dist'
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

# --- CENTRALE EXCHANGE INSTANTIE VOOR RATE LIMITING (spot-data voor candles) ---
exchange = ccxt.kucoin({'enableRateLimit': True})


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
        'baseline_error': None,   # holdout-skill van de huidige config (lager = beter, <1 = beter dan "geen beweging")
        'last_prediction': None,
        'last_prediction_price': None,
        'predicted_return_pct': 0.0,
        'direction_probability': 50.0,   # kans op STIJGING in %
        'signal': 'none',                # 'long' | 'short' | 'none'
        'signal_probability': 50.0,      # kans in % dat de richting van het signaal klopt
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

    # State uit een oude versie (geen data_status): direct opnieuw trainen i.p.v. wachten op min_interval
    if state.get('data_status') in (None, 'new'):
        state['last_trained_timestamp'] = 0

    # Oude/ontbrekende velden herstellen
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
    # Verouderde velden uit eerdere versies (foutmaat was niet vergelijkbaar met nieuwe validatie)
    for legacy in ('search_direction', 'max_depth', 'n_estimators',
                   'previous_learning_rate', 'previous_max_depth'):
        state.pop(legacy, None)
    if not isinstance(state.get('baseline_error'), (int, float)) or not math.isfinite(state['baseline_error']) \
            or state.get('data_status') in (None, 'new'):
        state['baseline_error'] = None
    return state


def _json_safe(obj):
    """Vervangt NaN/inf door None zodat het bestand geldige JSON is."""
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
    """Berekent alle indicatoren. Verwacht kolommen open/high/low/close/volume."""
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

    # Rolling VWAP (20 candles) i.p.v. cumulatief over het hele venster
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

    # ADX (correcte +DM / -DM definitie)
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
    Target = rendement van de VOLGENDE candle (close[t+1]/close[t] - 1), niet de absolute prijs.
    De laatste rij heeft nog geen label en wordt gebruikt als voorspel-input.
    """
    d = df.dropna(subset=feats + ['close']).reset_index(drop=True)
    if len(d) < 3:
        return None
    latest = d[feats].iloc[[-1]]
    ret = (d['close'].shift(-1) / d['close'] - 1.0).iloc[:-1]
    X = d[feats].iloc[:-1]
    X, ret = X.iloc[-lookback:], ret.iloc[-lookback:]
    return X, ret, latest


def evaluate_config(X, y, lr):
    """Out-of-sample test: train op het begin, test op de laatste candles. Geeft metrics en modelgewichten."""
    n = len(X)
    h = max(MIN_HOLDOUT, int(n * HOLDOUT_FRACTION))
    if n - h < MIN_TRAIN_ROWS:
        return None
    Xtr, Xte, ytr, yte = X.iloc[:-h], X.iloc[-h:], y.iloc[:-h], y.iloc[-h:]

    preds, maes = {}, {}
    for name, model in make_regressors(lr).items():
        model.fit(Xtr, ytr)
        p = np.asarray(model.predict(Xte), dtype=float)
        preds[name] = p
        maes[name] = float(np.mean(np.abs(p - yte.to_numpy())))

    inv = {k: 1.0 / (v + 1e-9) for k, v in maes.items()}
    tot = sum(inv.values())
    weights = {k: float(v / tot) for k, v in inv.items()}

    ens = sum(weights[k] * preds[k] for k in preds)
    y_arr = yte.to_numpy()
    ens_mae = float(np.mean(np.abs(ens - y_arr)))
    zero_mae = float(np.mean(np.abs(y_arr))) + 1e-12
    dir_acc = float(np.mean(np.sign(ens) == np.sign(y_arr)))

    clf_acc = None
    ytr_c = (ytr > 0).astype(int)
    if ytr_c.nunique() == 2:
        clf = make_classifier(lr)
        clf.fit(Xtr, ytr_c)
        p_up = clf.predict_proba(Xte)[:, 1]
        clf_acc = float(np.mean((p_up > 0.5) == (y_arr > 0)))

    return {
        'weights': weights,
        'maes': maes,
        'skill': ens_mae / zero_mae,     # <1 = beter dan "prijs blijft gelijk"
        'dir_acc': dir_acc,
        'clf_acc': clf_acc,
        'holdout_n': int(h),
    }


def mutate_config(cfg, regime):
    """Stelt een kleine, willekeurige wijziging voor. Wordt alleen overgenomen als die out-of-sample beter scoort."""
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
# EEN RUN VOOR 1 ASSET/TIMEFRAME
# =====================================================================
def _insufficient(state, now):
    state.update({'data_status': 'insufficient_data', 'signal': 'none', 'signal_probability': 50.0,
                  'last_prediction': None, 'last_prediction_price': None,
                  'last_trained_timestamp': now, 'updated_at': int(now)})
    return state


def process(df_raw, state, now):
    """
    Neemt ruwe candles (incl. de nog lopende candle), traint en voorspelt de volgende candle.
    Retourneert (state, ok).
    """
    # De laatste candle is nog niet afgesloten -> weglaten (anders is de laatste label/feature onbetrouwbaar)
    df_raw = df_raw.iloc[:-1].reset_index(drop=True)
    if len(df_raw) < 60:
        return _insufficient(state, now), False

    df = compute_features(df_raw)
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

    # Een mutatie wordt alleen overgenomen als hij op DEZELFDE holdout beter presteert
    chosen_cfg, chosen_res = best_cfg, best_res
    if cand_res is not None and (best_res is None or cand_res['skill'] < best_res['skill']):
        chosen_cfg, chosen_res = cand_cfg, cand_res
        print(f"   ↗ nieuwe config geaccepteerd (skill {cand_res['skill']:.3f}"
              + (f" < {best_res['skill']:.3f})" if best_res else ")"))

    if chosen_res is None:
        return _insufficient(state, now), False

    feats = chosen_res['features_used']
    X, y, latest = build_dataset(df, feats, chosen_cfg['lookback_window'])
    lr = chosen_cfg['learning_rate']
    weights = chosen_res['weights']

    # Finale fit op alle gelabelde data met de gekozen config
    pred_ret = {}
    for name, model in make_regressors(lr).items():
        model.fit(X, y)
        pred_ret[name] = float(model.predict(latest)[0])
    ens_ret = float(sum(weights[k] * pred_ret[k] for k in pred_ret))

    prob_up = 50.0
    y_cls = (y > 0).astype(int)
    if y_cls.nunique() == 2:
        try:
            clf = make_classifier(lr)
            clf.fit(X, y_cls)
            prob_up = float(clf.predict_proba(latest)[0][1] * 100)
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
        signal, sig_prob = 'none', 50.0     # regressie en classifier spreken elkaar tegen

    sign = 1 if reg_up else -1

    # Convert the per-model MAEs (which are in return units, e.g. 0.0045 = 0.45%)
    # to percent and USD so the UI can show both.
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
            # raw return MAE
            model_errors_returns[k] = float(v)
            # percent (0.45 -> 0.45%)
            model_errors_pct[k] = float(v * 100.0)
            # approximate USD error at the last close price
            model_errors_usd[k] = float(v * last_close)

    # Optionally keep a human-readable combined dict if your UI prefers strings:
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
        # keep the raw returns MAE for backwards compatibility
        'model_errors': model_errors_returns,
        # new keys for the UI
        'model_errors_pct': model_errors_pct,
        'model_errors_usd': model_errors_usd,
        'model_errors_readable': model_errors_readable,
        'baseline_error': chosen_res['skill'],
        'holdout_direction_accuracy': chosen_res['dir_acc'],
        'holdout_classifier_accuracy': chosen_res['clf_acc'],
        'holdout_samples': chosen_res['holdout_n'],
        'last_close': last_close,
        'last_prediction': last_close * (1.0 + ens_ret),
        'last_prediction_price': last_close * (1.0 + ens_ret),
        'predicted_return_pct': ens_ret * 100.0,
        'direction_probability': prob_up,
        'signal': signal,
        'signal_probability': sig_prob,
        'atr': atr,
        'atr_pct': atr / last_close * 100.0,
        # Alleen een SL/TP als er een eenduidig signaal is; anders suggereert het een richting die er niet is
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
# GITHUB SYNC (via de contents-API; geen git, geen force-push, token alleen in de header)
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
                        break   # ongewijzigd
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
                        continue   # sha verouderd -> opnieuw ophalen
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

                state, ok = process(df_raw, state, now)
                save_state(state_file, state)
                written.append(state_file)
                if ok:
                    print(f"✔ {asset} {tf_name}: signaal={state['signal']} "
                          f"P(up)={state['direction_probability']:.1f}% "
                          f"holdout-richting={state['holdout_direction_accuracy']:.2f} skill={state['baseline_error']:.3f}")
                else:
                    print(f"⚠️ {asset} {tf_name}: onvoldoende data ({state.get('data_status')})")
                gc.collect()
                time.sleep(4.0)   # rate limits
            except Exception as e:
                print(f"❌ Fout bij {asset} ({tf_name}): {e}")
                gc.collect()
                time.sleep(4.0)

    push_to_github(written)


if __name__ == '__main__':
    main()
