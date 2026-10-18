import os

# --- OUTPUT & WARNING SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import base64
import copy
import gc
import json
import logging
import math
import subprocess
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

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

def push_file_to_github(filename):
    token = os.environ.get('GITHUB_TOKEN')
    if not token or not os.path.exists(filename):
        return
    try:
        with open(filename, 'rb') as f:
            local = f.read()
        sha = None
        try:
            remote = json.loads(urllib.request.urlopen(urllib.request.Request(
                f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}',
                headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'}
            )).read().decode('utf-8'))
            sha = remote.get('sha')
        except Exception:
            pass
        payload = {
            'message': f'Auto-update {filename} [skip ci]',
            'content': base64.b64encode(local).decode('ascii'),
            'branch': GITHUB_BRANCH,
        }
        if sha:
            payload['sha'] = sha
        urllib.request.urlopen(urllib.request.Request(
            f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
            method='PUT'
        ))
        print(f"☁️ Bestand {filename} succesvol gepusht naar GitHub.")
    except Exception as e:
        print(f"❌ Push van {filename} mislukt: {e}")

# --- STAP 1: DYNAMISCHE ASSET SELECTIE (Kern + Scanner Top 5) ---
def get_combined_trading_symbols(top_n=5):
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'LINK/USDT:USDT']
    ranking_file = 'kucoin_best_coins_ranking.csv'
    dynamic_symbols = []
    
    if os.path.exists(ranking_file):
        try:
            df = pd.read_csv(ranking_file)
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].head(top_n).tolist()
                for s in symbols:
                    if s not in dynamic_symbols and ':USDT' in s:
                        dynamic_symbols.append(s)
            print(f"📈 Dynamische top {top_n} geladen uit scanner: {dynamic_symbols}")
        except Exception as e:
            print(f"⚠️ Fout bij lezen van ranking CSV: {e}")
            
    combined = list(dict.fromkeys(core_symbols + dynamic_symbols))
    print(f"🎯 Definitieve actieve trading scope ({len(combined)} symbols): {combined}")
    return combined

timeframes = {
    '4h': {'interval': '4h', 'limit': 500, 'min_interval_seconds': 3 * 3600},
    '1d': {'interval': '1d', 'limit': 365, 'min_interval_seconds': 23 * 3600},
}

base_features = [
    'return_1', 'dist_sma20', 'dist_sma50', 'RSI', 'atr_ratio', 
    'bb_position', 'ADX', 'MACD_hist', 'OBV_slope', 'VWAP_dist',
    'dist_support', 'dist_resistance'
]

MODEL_NAMES = ['xgb', 'cat', 'rf']

HOLDOUT_FRACTION = 0.25
MIN_HOLDOUT = 15
MIN_TRAIN_ROWS = 25
MIN_FEATURE_ROWS = 60

SL_ATR_MULT = 2.5
TP_ATR_MULT = 5.0

# Strenge initialisatie als lineaire swap exchange
exchange = ccxt.kucoin({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

def safe_div(num, den):
    if den is None or pd.isna(den) or abs(float(den)) < 1e-12:
        return 0.0
    return float(num) / float(den)


def fetch_market_context_from_exchange(exch, symbol):
    funding = 0.0
    oi_data = None
    orderbook = None
    swap_symbol = f"{symbol}:USDT" if not symbol.endswith(":USDT") else symbol

    try:
        fr = exch.fetch_funding_rate(swap_symbol)
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


def default_state():
    return {
        'learning_rate': 0.05,
        'lookback_window': 120,
        'current_regime': 'trending',
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
    high = df['high']
    low = df['low']

    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()

    df['return_1'] = close.pct_change()
    df['dist_sma20'] = (close - df['SMA_20']) / (df['SMA_20'] + 1e-8)
    df['dist_sma50'] = (close - df['SMA_50']) / (df['SMA_50'] + 1e-8)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    df['atr_ratio'] = df['ATR'] / (close + 1e-8)

    bb_std = close.rolling(window=20).std()
    bb_upper = df['SMA_20'] + (bb_std * 2.0)
    bb_lower = df['SMA_20'] - (bb_std * 2.0)
    df['bb_position'] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-8)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = true_range.rolling(window=14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14).sum() / (tr14 + 1e-8)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14).sum() / (tr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
    df['ADX'] = dx.rolling(window=14).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd_line - macd_signal

    df['OBV'] = (np.sign(close.diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_slope'] = df['OBV'].diff() / (df['volume'].rolling(20).mean() + 1e-8)

    typical_price = (high + low + close) / 3
    vwap = (typical_price * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)
    df['VWAP_dist'] = (close - vwap) / (vwap + 1e-8)

    rolling_min = low.rolling(window=50, min_periods=10).min()
    rolling_max = high.rolling(window=50, min_periods=10).max()
    df['dist_support'] = (close - rolling_min) / (df['ATR'] + 1e-8)
    df['dist_resistance'] = (rolling_max - close) / (df['ATR'] + 1e-8)

    return df.replace([np.inf, -np.inf], np.nan)


def enrich_market_context(df, funding=None, oi=None, orderbook=None):
    out = df.copy()
    out['funding_rate'] = pd.Series(funding, index=out.index) if funding is not None else 0.0
    out['oi_change'] = 0.0
    out['oi_ratio'] = 0.0
    out['bid_ask_imbalance'] = 0.0
    out['book_pressure'] = 0.0
    out['taker_buy_ratio'] = 0.0
    out['btc_trend_gap'] = 0.0
    return out


def get_market_regime(df):
    try:
        if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]):
            return 'trending' if df['ADX'].iloc[-1] > 16 else 'sideways'
    except Exception:
        pass
    return 'trending'


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
    d = df.dropna(subset=feats + ['close']).reset_index(drop=True)
    if len(d) < lookback + 5:
        return None
    X = d[feats].iloc[:-1]
    y = (d['close'].shift(-1) / d['close'] - 1.0).iloc[:-1]
    X = X.iloc[-lookback:]
    y = y.iloc[-lookback:]
    latest = d[feats].iloc[[-1]]
    return X, y, latest


def evaluate_config(X, y, lr):
    n = len(X)
    h = max(MIN_HOLDOUT, int(n * HOLDOUT_FRACTION))
    if n - h < MIN_TRAIN_ROWS:
        return None
    Xtr, Xte, ytr, yte = X.iloc[:-h], X.iloc[-h:], y[:-h], y[-h:]

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
    funding, oi, ob = fetch_market_context_from_exchange(exchange, symbol)
    df = enrich_market_context(df, funding=funding, oi=oi, orderbook=ob)

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
    
    adx_val = df['ADX'].iloc[-1] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 20.0
    
    if adx_val >= 16.0 and reg_up and clf_up:
        signal, sig_prob = 'long', prob_up
    elif adx_val >= 16.0 and (not reg_up) and (not clf_up):
        signal, sig_prob = 'short', 100.0 - prob_up
    else:
        signal, sig_prob = 'none', 50.0

    sign = 1 if reg_up else -1

    state.update({
        'learning_rate': lr,
        'lookback_window': chosen_cfg['lookback_window'],
        'active_features': feats,
        'model_weights': weights,
        'baseline_error': chosen_res['skill'],
        'holdout_direction_accuracy': chosen_res['dir_acc'],
        'last_close': last_close,
        'last_prediction': last_close * (1.0 + ens_ret),
        'last_prediction_price': last_close * (1.0 + ens_ret),
        'predicted_return_pct': ens_ret * 100.0,
        'direction_probability': prob_up,
        'signal': signal,
        'signal_probability': sig_prob,
        'atr': atr,
        'atr_pct': atr / last_close * 100.0,
        'stop_loss': (last_close - sign * SL_ATR_MULT * atr) if signal != 'none' else 0.0,
        'take_profit': (last_close + sign * TP_ATR_MULT * atr) if signal != 'none' else 0.0,
        'data_status': 'ok',
        'last_trained_timestamp': now,
        'updated_at': int(now),
    })
    return state, True


def push_to_github(files):
    token = os.environ.get('GITHUB_TOKEN')
    if not token or not files:
        return
    for fname in files:
        try:
            with open(fname, 'rb') as f:
                local = f.read()
            sha = None
            try:
                remote = json.loads(urllib.request.urlopen(urllib.request.Request(
                    f'https://api.github.com/repos/{GITHUB_REPO}/contents/{fname}?ref={GITHUB_BRANCH}',
                    headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'}
                )).read().decode('utf-8'))
                sha = remote.get('sha')
            except Exception:
                pass
            payload = {
                'message': f'Auto-update {fname} [skip ci]',
                'content': base64.b64encode(local).decode('ascii'),
                'branch': GITHUB_BRANCH,
            }
            if sha:
                payload['sha'] = sha
            urllib.request.urlopen(urllib.request.Request(
                f'https://api.github.com/repos/{GITHUB_REPO}/contents/{fname}',
                data=json.dumps(payload).encode('utf-8'),
                headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'},
                method='PUT'
            ))
        except Exception as e:
            print(f"❌ Push van {fname} mislukt: {e}")


def main():
    # 1. Voer eerst automatisch de marktscanner uit
    try:
        print("🔄 Start automatische marktscan via getcoins.py...")
        subprocess.run(["python", "getcoins.py"], check=True)
        print("✅ Marktscan succesvol voltooid. CSV is bijgewerkt.")
        push_file_to_github('kucoin_best_coins_ranking.csv')
    except Exception as e:
        print(f"⚠️ Kon getcoins.py niet uitvoeren of pushen: {e}. We gebruiken de bestaande CSV indien aanwezig.")

    # 2. Haal de actieve symbolen op uit de verse ranking
    symbols_scope = get_combined_trading_symbols(top_n=5)

    written = []
    for symbol in symbols_scope:
        asset = symbol.split('/')[0].split(':')[0]
        
        for tf_name, tf_info in timeframes.items():
            state_file = f'model_state_{asset.lower()}_{tf_name}.json'
            try:
                now = time.time()
                state = load_or_init_state(state_file)
                if now - state.get('last_trained_timestamp', 0) < tf_info['min_interval_seconds']:
                    continue
                df_raw = fetch_data_safely(symbol, tf_info)
                if df_raw.empty:
                    continue
                state, ok = process(df_raw, state, now, symbol)
                save_state(state_file, state)
                written.append(state_file)
                gc.collect()
                time.sleep(4.0)
            except Exception as e:
                print(f"❌ Fout bij {asset} ({tf_name}): {e}")
                gc.collect()
                time.sleep(4.0)
    push_to_github(written)


if __name__ == '__main__':
    main()
