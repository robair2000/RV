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

# --- STAP 1: DYNAMISCHE ASSET SELECTIE ---
def get_combined_trading_symbols(top_n=5):
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
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
    '4h': {'interval': '4h', 'limit': 600, 'min_interval_seconds': 3 * 3600},
    '1d': {'interval': '1d', 'limit': 400, 'min_interval_seconds': 23 * 3600},
}

# Orthogonale, stationaire features voor de Quant Worker (onafhankelijk van Agent S&R)
base_features = [
    'return_1', 'RSI', 'atr_ratio', 'bb_position', 'MACD_hist', 
    'OBV_slope', 'VWAP_dist', 'funding_rate', 'orderbook_imbalance'
]

MODEL_NAMES = ['xgb', 'cat', 'rf']

HOLDOUT_FRACTION = 0.25
MIN_HOLDOUT = 30
MIN_TRAIN_ROWS = 60
MIN_FEATURE_ROWS = 100

SL_ATR_MULT = 2.5
TP_ATR_MULT = 5.0

exchange = ccxt.kucoinfutures({
    'enableRateLimit': True,
})

def safe_div(num, den):
    if den is None or pd.isna(den) or abs(float(den)) < 1e-12:
        return 0.0
    return float(num) / float(den)


def fetch_market_context_from_exchange(exch, symbol):
    funding = 0.0
    orderbook_imbalance = 0.0
    swap_symbol = f"{symbol}:USDT" if not symbol.endswith(":USDT") else symbol

    try:
        fr = exch.fetch_funding_rate(swap_symbol)
        if fr and 'fundingRate' in fr:
            funding = float(fr['fundingRate'])
    except Exception:
        try:
            time.sleep(0.3)
            fr = exch.fetch_funding_rate(symbol)
            if fr and 'fundingRate' in fr:
                funding = float(fr['fundingRate'])
        except Exception:
            pass

    time.sleep(0.3)
    try:
        orderbook = exch.fetch_order_book(symbol, limit=20)
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            bids_vol = sum([b[1] for b in orderbook['bids'][:10]])
            asks_vol = sum([a[1] for a in orderbook['asks'][:10]])
            orderbook_imbalance = (bids_vol - asks_vol) / (bids_vol + asks_vol + 1e-8)
    except Exception:
        pass

    return funding, orderbook_imbalance


STATE_VERSION = 2  # Verhoog dit getal bij ingrijpende logica-updates om oude states automatisch te resetten

def default_state():
    return {
        'state_version': STATE_VERSION,
        'learning_rate': 0.03,
        'lookback_window': 350,
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
            
            # Controleer de versie: als de opgeslagen state verouderd is, negeer hem en forceer defaults!
            if isinstance(loaded, dict) and loaded.get('state_version', 1) == STATE_VERSION:
                state.update(loaded)
            else:
                print(f"🔄 Oude of incompatibele state gedetecteerd voor {state_file}. Reset naar v{STATE_VERSION} defaults.")
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
        if not ohlcv or len(ohlcv) < 5:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # KRITISCH: Verwijder altijd de incomplete lopende bar om repainting te voorkomen
        df = df.iloc[:-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"❌ KuCoin Futures fetch fout voor {symbol}: {e}")
        return pd.DataFrame()


def compute_features(df, funding=0.0, ob_imbalance=0.0):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    sma20 = close.rolling(window=20).mean()
    df['return_1'] = close.pct_change()

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

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd_line - macd_signal

    obv = (np.sign(close.diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_slope'] = obv.diff(3) / (df['volume'].rolling(20).mean() + 1e-8)

    typical_price = (high + low + close) / 3
    vwap = (typical_price * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)
    df['VWAP_dist'] = (close - vwap) / (vwap + 1e-8)

    df['funding_rate'] = float(funding)
    df['orderbook_imbalance'] = float(ob_imbalance)
    df['ATR'] = atr

    return df.replace([np.inf, -np.inf], np.nan)


def get_market_regime(df):
    try:
        if 'atr_ratio' in df.columns and not pd.isna(df['atr_ratio'].iloc[-1]):
            return 'trending' if df['atr_ratio'].iloc[-1] > 1.5 else 'sideways'
    except Exception:
        pass
    return 'trending'


def make_regressors(lr):
    return {
        'xgb': xgb.XGBRegressor(learning_rate=lr, max_depth=3, n_estimators=40, random_state=42, verbosity=0, n_jobs=1),
        'cat': CatBoostRegressor(learning_rate=lr, depth=3, iterations=40, random_seed=42, verbose=0, thread_count=1),
        'rf': RandomForestRegressor(n_estimators=25, max_depth=3, random_state=42, n_jobs=1),
    }


def make_classifier(lr):
    return xgb.XGBClassifier(learning_rate=lr, n_estimators=25, max_depth=2, random_state=42,
                             eval_metric='logloss', verbosity=0, n_jobs=1)


def usable_features(df, feats):
    return [f for f in feats if f in df.columns and df[f].notna().sum() >= MIN_FEATURE_ROWS]


def build_dataset(df, feats, lookback):
    d = df.dropna(subset=feats + ['close', 'ATR']).reset_index(drop=True)
    if len(d) < lookback + 10:
        return None
    # Snijd de laatste 3 rijen af omdat shift(-3) daar NaN oplevert
    X = d[feats].iloc[:-3]
    y = (d['close'].shift(-3) / d['close'] - 1.0).iloc[:-3]
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
    if len(df_raw) < MIN_FEATURE_ROWS:
        return _insufficient(state, now), False

    funding, ob_imbalance = fetch_market_context_from_exchange(exchange, symbol)
    df = compute_features(df_raw, funding=funding, ob_imbalance=ob_imbalance)

    regime = get_market_regime(df)
    state['current_regime'] = regime

    best_cfg = {
        'learning_rate': float(state['learning_rate']),
        'lookback_window': int(state['lookback_window']),
        'features': list(state['regime_features'].get(regime, base_features)),
    }
    best_res = score_config(df, best_cfg)

    if best_res is None or best_res['skill'] >= 1.0 or best_res['dir_acc'] < 0.50:
        return _insufficient(state, now), False

    chosen_cfg, chosen_res = best_cfg, best_res
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
    clf_up = prob_up >= 52.0 # Strikte drempel
    clf_down = prob_up <= 48.0

    # GEBRUIK HIER DE HARDE PERFORMANCE GATE: Alleen signaal bij echte out-of-sample edge
    has_edge = chosen_res['dir_acc'] >= 0.52 and chosen_res['skill'] < 0.98

    if has_edge and reg_up and clf_up:
        signal, sig_prob = 'long', prob_up
    elif has_edge and (not reg_up) and clf_down:
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
    try:
        print("🔄 Start automatische marktscan via getcoins.py...")
        subprocess.run(["python", "getcoins.py"], check=True)
        print("✅ Marktscan succesvol voltooid. CSV is bijgewerkt.")
        push_file_to_github('kucoin_best_coins_ranking.csv')
    except Exception as e:
        print(f"⚠️ Kon getcoins.py niet uitvoeren of pushen: {e}. We gebruiken de bestaande CSV indien aanwezig.")

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
                time.sleep(3.0)
            except Exception as e:
                print(f"❌ Fout bij {asset} ({tf_name}): {e}")
                gc.collect()
                time.sleep(3.0)
    push_to_github(written)


if __name__ == '__main__':
    main()
