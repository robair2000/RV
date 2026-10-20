import os

# --- OUTPUT & WARNING SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from github_utils import github_pull, github_put
from market_utils import TF_MS, fetch_ohlcv_paginated, fetch_cached_1d_shared, fetch_cached_4h_shared
from asset_utils import get_combined_trading_assets


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, format, *args):
        pass


def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"🌐 Web service health check server gestart op poort {port}")
    server.serve_forever()


# --- CONFIG ---
MIN_VOL_RATIO = float(os.environ.get('MIN_VOL_RATIO', 1.3))
MIN_BOSS_SCORE = float(os.environ.get('MIN_BOSS_SCORE', 75.0))

TRAIL_SL_ATR = 1.5
TP_ATR_BASE = 3.0
META_SL_ATR = float(os.environ.get('META_SL_ATR', TRAIL_SL_ATR))
META_TP_ATR = float(os.environ.get('META_TP_ATR', TP_ATR_BASE))
META_HORIZON_BARS = int(os.environ.get('META_HORIZON_BARS', 12))

META_COST_PCT = float(os.environ.get('META_COST_PCT', 0.0012))
META_MARGIN = float(os.environ.get('META_MARGIN', 0.03))
META_MIN_EVENTS = int(os.environ.get('META_MIN_EVENTS', 60))
META_MIN_POS = int(os.environ.get('META_MIN_POS', 12))
META_MIN_Z = float(os.environ.get('META_MIN_Z', 1.05))
META_SE_INFLATION = float(os.environ.get('META_SE_INFLATION', 1.2))
META_DECLUSTER_BARS = int(os.environ.get('META_DECLUSTER_BARS', 4))
META_HISTORY_4H = int(os.environ.get('META_HISTORY_4H', 2000))
META_HISTORY_1D = int(os.environ.get('META_HISTORY_1D', 450))
META_RETRAIN_SECONDS = int(os.environ.get('META_RETRAIN_SECONDS', 4 * 3600))
META_FOLD_CUTS = (0.40, 0.55, 0.70, 0.85)

PUSH_HEARTBEAT_SECONDS = int(os.environ.get('PUSH_HEARTBEAT_SECONDS', 3 * 3600))

META_FEATURES = [
    'bb_width_pctl', 'atr_ratio', 'vol_ratio', 'vol_ratio_3', 
    'dist_high20', 'rsi_c', 'pct_b_c', 'adx4', 'adx1d', 'ret_6'
]

CHECK_INTERVAL_SECONDS = 1800
exchange = ccxt.kucoinfutures({'enableRateLimit': True})


def load_markov_4h_state(asset):
    filename = f"markov_4h_state_{asset.lower()}.json"
    if not os.path.exists(filename):
        github_pull(filename, agent='squeeze-agent')
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'probabilities_next_4h': {'green': 50.0, 'red': 50.0}}


def compute_indicator_frame(df):
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / close) * 100.0
    atr_ratio = atr / atr.rolling(50).mean().fillna(1.0)

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    
    def _wilder(s, n=14):
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    tr_s = _wilder(tr)
    plus_di = 100.0 * _wilder(plus_dm) / (tr_s + 1e-8)
    minus_di = 100.0 * _wilder(minus_dm) / (tr_s + 1e-8)
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
    adx = _wilder(dx)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + 2.0 * std20
    lower_bb = sma20 - 2.0 * std20
    bb_width = (upper_bb - lower_bb) / (sma20 + 1e-8)
    bb_width_pctl = bb_width.rolling(120).rank(pct=True).fillna(0.5)
    
    pct_b = (close - lower_bb) / ((upper_bb - lower_bb) + 1e-8)

    vol_sma = volume.rolling(20).mean()
    vol_ratio = (volume / vol_sma.where(vol_sma > 0)).fillna(1.0)
    vol_ratio_3 = volume.rolling(3).mean() / vol_sma.where(vol_sma > 0).fillna(1.0)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))

    high20 = high.rolling(20).max().shift(1)
    dist_high20 = (close - high20) / (atr + 1e-8)

    return pd.DataFrame({
        'close': close, 'sma20': sma20, 'atr': atr, 'atr_pct': atr_pct, 'atr_ratio': atr_ratio,
        'adx': adx, 'pct_b': pct_b, 'rsi': rsi, 'vol_ratio': vol_ratio, 'vol_ratio_3': vol_ratio_3,
        'bb_width': bb_width, 'bb_width_pctl': bb_width_pctl, 'dist_high20': dist_high20,
    }, index=df.index)


def compute_primary_setups(df_4h, df_1d, markov_data=None):
    f = compute_indicator_frame(df_4h)
    f['timestamp'] = df_4h['timestamp'].values
    f['ret_6'] = f['close'].pct_change(6) * 100.0

    f1 = compute_indicator_frame(df_1d)
    daily = pd.DataFrame({
        'avail_ts': df_1d['timestamp'].values + TF_MS['1d'] - TF_MS['4h'],
        'adx1d': f1['adx'].values,
    }).dropna().sort_values('avail_ts')
    left = pd.DataFrame({'timestamp': f['timestamp'].values}).sort_values('timestamp')
    merged = pd.merge_asof(left, daily, left_on='timestamp', right_on='avail_ts', direction='backward')
    f['adx1d'] = merged['adx1d'].values

    m_probs = markov_data.get('probabilities_next_4h', {'green': 50.0, 'red': 50.0}) if markov_data else {'green': 50.0, 'red': 50.0}
    p_green = m_probs.get('green', 50.0)
    p_red = m_probs.get('red', 50.0)

    # Squeeze & Volume filters
    is_squeeze = f['bb_width_pctl'] <= 0.35
    volume_ok = f['vol_ratio'] >= MIN_VOL_RATIO

    # Uitbraak met 4h Markov kansen als richtingsfilter
    long_breakout = is_squeeze & (f['pct_b'] > 0.92) & volume_ok & (f['close'] > f['sma20']) & (p_green >= 52.0)
    short_breakout = is_squeeze & (f['pct_b'] < 0.08) & volume_ok & (f['close'] < f['sma20']) & (p_red >= 52.0)

    f['setup_long'] = long_breakout.fillna(False).astype(bool)
    f['setup_short'] = short_breakout.fillna(False).astype(bool)

    ext = ((f['pct_b'] - 0.5).abs() - 0.5).clip(lower=0.0)
    vol_pts = ((f['vol_ratio'] - MIN_VOL_RATIO) / 1.0).clip(0, 1) * 15.0
    f['twm_score'] = np.where(f['setup_long'] | f['setup_short'], (70.0 + vol_pts + ext * 10.0).round(2), 50.0)
    return f


def event_features(f, idx, sgn):
    sub = f.iloc[list(idx)]
    out = pd.DataFrame({
        'bb_width_pctl': sub['bb_width_pctl'].values,
        'atr_ratio': sub['atr_ratio'].values,
        'vol_ratio': sub['vol_ratio'].values,
        'vol_ratio_3': sub['vol_ratio_3'].values,
        'dist_high20': sgn * sub['dist_high20'].values,
        'rsi_c': sgn * (sub['rsi'].values - 50.0),
        'pct_b_c': sgn * (sub['pct_b'].values - 0.5),
        'adx4': sub['adx'].values,
        'adx1d': sub['adx1d'].values,
        'ret_6': sgn * sub['ret_6'].values,
    })
    return out.replace([np.inf, -np.inf], np.nan)


def barrier_outcome(hi, lo, cl, t, sgn, atr):
    n = len(cl)
    if not (atr > 0):
        return None
    entry = cl[t]
    sl = entry - sgn * META_SL_ATR * atr
    tp = entry + sgn * META_TP_ATR * atr
    cost_r = META_COST_PCT * entry / (META_SL_ATR * atr)
    end = t + META_HORIZON_BARS
    for j in range(t + 1, min(end, n - 1) + 1):
        if sgn == 1:
            hit_sl, hit_tp = lo[j] <= sl, hi[j] >= tp
        else:
            hit_sl, hit_tp = hi[j] >= sl, lo[j] <= tp
        if hit_sl:
            return 0, -1.0 - cost_r
        if hit_tp:
            return 1, META_TP_ATR / META_SL_ATR - cost_r
    if end > n - 1:
        return None
    r = sgn * (cl[end] - entry) / (META_SL_ATR * atr)
    return 0, r - cost_r


def build_meta_events(histories, markov_data_dict):
    parts = []
    horizon_ms = META_HORIZON_BARS * TF_MS['4h']
    for asset, (d4, d1) in histories.items():
        if d4 is None or d1 is None or len(d4) < 250 or len(d1) < 60:
            continue
        m_data = markov_data_dict.get(asset, {})
        f = compute_primary_setups(d4, d1, markov_data=m_data)
        hi, lo, cl = d4['high'].values, d4['low'].values, d4['close'].values
        ts = d4['timestamp'].values
        atr_arr = f['atr'].values
        for direction, sgn, col in (('long', 1, 'setup_long'), ('short', -1, 'setup_short')):
            idxs = np.where(f[col].values & ~np.isnan(atr_arr))[0]
            kept, outcomes, last = [], [], -10 ** 9
            for t in idxs:
                if t - last < META_DECLUSTER_BARS:
                    continue
                res = barrier_outcome(hi, lo, cl, int(t), sgn, float(atr_arr[t]))
                if res is None:
                    continue
                last = t
                kept.append(int(t))
                outcomes.append(res)
            if not kept:
                continue
            feats = event_features(f, kept, sgn)
            feats['asset'] = asset
            feats['direction'] = direction
            feats['ts'] = ts[kept].astype('int64')
            feats['ts_end'] = feats['ts'] + horizon_ms
            feats['label'] = [o[0] for o in outcomes]
            feats['net_r'] = [o[1] for o in outcomes]
            parts.append(feats)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def make_meta_classifier():
    return xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=3.0, random_state=42, verbosity=0, n_jobs=1, eval_metric='logloss',
    )


def train_meta_model(events):
    pack = {'ok': False, 'reason': 'geen events', 'model': None, 'oos': pd.DataFrame(),
            'n_events': 0, 'n_pos': 0, 'auc': None, 'z': None, 'threshold': 0.52, 'trained_at': time.time()}
    if events is None or events.empty:
        return pack

    ev = events.sort_values('ts').reset_index(drop=True)
    n = len(ev)
    pack['n_events'] = n
    pack['n_pos'] = int(ev['label'].sum())

    if n < META_MIN_EVENTS or pack['n_pos'] < META_MIN_POS:
        pack['reason'] = f"te weinig events ({n} events, {pack['n_pos']} winnaars)"
        return pack

    final = make_meta_classifier()
    final.fit(ev[META_FEATURES], ev['label'])
    pack.update({'ok': True, 'reason': 'ok', 'model': final})
    return pack


def analyze_advanced_patterns(df_4h, df_1d, asset='BTC', meta_pack=None, markov_data=None):
    if df_4h is None or len(df_4h) < 60 or df_1d is None or len(df_1d) < 60:
        return {'pattern_signal': 'none', 'pattern_score': 50.0, 'meta_status': 'insufficient_data'}

    f = compute_primary_setups(df_4h, df_1d, markov_data=markov_data)
    i = len(f) - 1
    last = f.iloc[i]

    signal = 'long' if bool(last['setup_long']) else ('short' if bool(last['setup_short']) else 'none')
    score = 50.0
    meta_status = 'no_setup'
    meta_prob = None

    pack = meta_pack or {}
    if signal != 'none':
        if not pack.get('ok') or pack.get('model') is None:
            meta_status = 'meta-gate gesloten'
            signal = 'none'
        else:
            sgn = 1 if signal == 'long' else -1
            feats = event_features(f, [i], sgn)[META_FEATURES]
            meta_prob = float(pack['model'].predict_proba(feats)[0][1])
            thr = float(pack['threshold'])
            if meta_prob < thr:
                meta_status = 'meta_prob_below_threshold'
                signal = 'none'
            else:
                meta_status = 'ok'
                score = round(min(100.0, 75.0 + 200.0 * (meta_prob - thr)), 2)

    leverage = 3 if score >= 85.0 else (2 if score >= 75.0 else 1)
    raw_ts = float(df_4h['timestamp'].iloc[-1])

    return {
        'pattern_score': float(score),
        'pattern_signal': signal,
        'twm_signal': signal,
        'twm_score': float(last['twm_score']),
        'recommended_leverage': int(leverage),
        'strictness_mode': 'squeeze_breakout_markov_4h',
        'atr_pct': float(last['atr_pct']),
        'adx_4h': float(last['adx']),
        'volume_expansion_ratio': float(last['vol_ratio']),
        'sma20': float(last['sma20']),
        'meta_status': meta_status,
        'meta_prob': meta_prob,
        'last_candle_ts': int(raw_ts / 1000.0 if raw_ts > 1e10 else raw_ts),
        'updated_at': time.time(),
    }


def push_state_to_github(filename, data):
    content_bytes = json.dumps(data, indent=2).encode('utf-8')
    with open(filename, 'wb') as f:
        f.write(content_bytes)
    github_put(filename, content_bytes, message=f"Auto-update state {filename} [skip ci]", agent='squeeze-agent')


def run_final_boss_scan():
    print("--- [Squeeze Breakout + 4h Markov Agent] Start marktscan ---")
    symbols = get_combined_trading_assets(top_n=5)

    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    live, markov_dict = {}, {}
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        markov_dict[asset] = load_markov_4h_state(asset)
        d4 = fetch_ohlcv_paginated(exchange, symbol, '4h', total_bars=250)
        time.sleep(1.0)
        d1 = fetch_cached_1d_shared(exchange, symbol, asset, total_bars=250)
        live[asset] = (d4, d1)
        time.sleep(1.5)

    histories = {asset: (d4, d1) for asset, (d4, d1) in live.items() if not d4.empty and not d1.empty}
    meta_pack = train_meta_model(build_meta_events(histories, markov_dict))

    for asset, (d4, d1) in live.items():
        if not d4.empty and not d1.empty:
            m_data = markov_dict.get(asset, {})
            p_g = m_data.get('probabilities_next_4h', {}).get('green', 50.0)
            analysis = analyze_advanced_patterns(d4, d1, asset=asset, meta_pack=meta_pack, markov_data=m_data)
            push_state_to_github(f'pattern_state_{asset.lower()}_4h.json', analysis)
            print(f"🚀 {asset} [P(Green 4h)={p_g}%] -> Signal={analysis['pattern_signal'].upper()} | Score={analysis['pattern_score']}")
        time.sleep(1.0)


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Squeeze Breakout + 4h Markov Agent gestart.")
    while True:
        try:
            run_final_boss_scan()
        except Exception as e:
            print(f"Fout in scan lus: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)
