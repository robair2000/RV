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
from market_utils import TF_MS, fetch_ohlcv_paginated, fetch_cached_1d_shared


# --- RENDER WEB SERVER ---
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


def pull_ranking_from_github():
    if github_pull('kucoin_best_coins_ranking.csv', agent='v20-smart-agent'):
        print("📥 Ranking CSV is up-to-date (GitHub).")
    else:
        print("⚠️ Kon ranking CSV niet ophalen van GitHub, we gebruiken lokale versie indien aanwezig.")


# --- CONFIG (alles overschrijfbaar via env) ---
MIN_ADX_TREND = float(os.environ.get('MIN_ADX_TREND', 18.0))
MIN_BOSS_SCORE = float(os.environ.get('MIN_BOSS_SCORE', 78.0))
ZONE_MAX_AGE_BARS = int(os.environ.get('ZONE_MAX_AGE_BARS', 60))  # supply/demand-zone verloopt na 60 4h-candles (10 dagen)

# Trader-parameters (moeten gelijk blijven aan trader_worker.py; worden ook naar de state geschreven)
TRAIL_SL_ATR = 2.5
MIN_RRR = 3.0
TP_ATR_BASE = 5.0
EFFECTIVE_TP_ATR = max(TP_ATR_BASE, TRAIL_SL_ATR * (MIN_RRR + 0.2))  # = 8.0

# Meta-labeling (triple barrier, gelijk aan het echte trade-profiel van de V20 trader)
META_SL_ATR = float(os.environ.get('META_SL_ATR', TRAIL_SL_ATR))
META_TP_ATR = float(os.environ.get('META_TP_ATR', EFFECTIVE_TP_ATR))
META_HORIZON_BARS = int(os.environ.get('META_HORIZON_BARS', 30))      # 30 x 4h = 5 dagen max
META_COST_PCT = float(os.environ.get('META_COST_PCT', 0.0012))        # round-trip fees + slippage (fractie)
META_MARGIN = float(os.environ.get('META_MARGIN', 0.05))              # marge boven break-even kans
META_MIN_EVENTS = int(os.environ.get('META_MIN_EVENTS', 150))
META_MIN_POS = int(os.environ.get('META_MIN_POS', 20))
META_MIN_Z = float(os.environ.get('META_MIN_Z', 1.28))                # one-sided ~90%
META_SE_INFLATION = float(os.environ.get('META_SE_INFLATION', 1.3))   # gecorreleerde assets/events
META_REQUIRE_OOS_EV = os.environ.get('META_REQUIRE_OOS_EV', 'true').lower() in ('1', 'true', 'yes')
META_MIN_SELECTED = int(os.environ.get('META_MIN_SELECTED', 15))
META_DECLUSTER_BARS = int(os.environ.get('META_DECLUSTER_BARS', 6))
META_HISTORY_4H = int(os.environ.get('META_HISTORY_4H', 2000))
META_HISTORY_1D = int(os.environ.get('META_HISTORY_1D', 450))
META_RETRAIN_SECONDS = int(os.environ.get('META_RETRAIN_SECONDS', 4 * 3600))
META_FOLD_CUTS = (0.40, 0.55, 0.70, 0.85)

PUSH_HEARTBEAT_SECONDS = int(os.environ.get('PUSH_HEARTBEAT_SECONDS', 3 * 3600))

META_FEATURES = [
    'sma_dist', 'ema9_dist', 'macd_n', 'rsi_c', 'pct_b_c', 'atr_pct',
    'adx4', 'adx1d', 'vol_ratio', 'lvl_dist_atr', 'in_zone', 'ret_6',
]


def get_combined_trading_assets(top_n=5):
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
    ranking_file = 'kucoin_best_coins_ranking.csv'
    dynamic_symbols = []

    if os.path.exists(ranking_file):
        try:
            df = pd.read_csv(ranking_file)
            if 'Symbol' in df.columns:
                for s in df['Symbol'].head(top_n).tolist():
                    if s not in dynamic_symbols and ':USDT' in s:
                        dynamic_symbols.append(s)
            print(f"📈 Dynamische top {top_n} ingeladen voor agent: {dynamic_symbols}")
        except Exception as e:
            print(f"⚠️ Fout bij lezen van ranking CSV in agent: {e}")

    combined = list(dict.fromkeys(core_symbols + dynamic_symbols))
    print(f"🎯 Actieve agent trading scope ({len(combined)} symbols): {combined}")
    return combined


CHECK_INTERVAL_SECONDS = 1800

exchange = ccxt.kucoinfutures({'enableRateLimit': True})


def fetch_historical_candles(symbol, timeframe, limit=250):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 5:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.iloc[:-1].reset_index(drop=True)  # laatste (lopende) candle weg
        return df
    except Exception as e:
        print(f"⚠️ Fout bij ophalen candles voor {symbol} ({timeframe}): {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# INDICATOREN (volledig vectorized: dezelfde code voor live en voor historie)
# ---------------------------------------------------------------------------
def _wilder(s, n=14):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_indicator_frame(df):
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / close) * 100.0

    # ADX met Wilder-smoothing en correct berekende +DM/-DM
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr_s = _wilder(tr)
    plus_di = 100.0 * _wilder(plus_dm) / (tr_s + 1e-8)
    minus_di = 100.0 * _wilder(minus_dm) / (tr_s + 1e-8)
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
    adx = _wilder(dx)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + 2.0 * std20
    lower_bb = sma20 - 2.0 * std20
    bb_width = (upper_bb - lower_bb) / (sma20 + 1e-8)
    prior_min_width = bb_width.shift(1).rolling(20).min()
    bb_squeeze = bb_width <= prior_min_width * 1.1
    pct_b = (close - lower_bb) / ((upper_bb - lower_bb) + 1e-8)
    ema9 = close.ewm(span=9, adjust=False).mean()

    vol_sma = volume.rolling(20).mean()
    vol_ratio = (volume / vol_sma.where(vol_sma > 0)).fillna(1.0)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))

    return pd.DataFrame({
        'close': close, 'ema9': ema9, 'sma20': sma20, 'atr': atr, 'atr_pct': atr_pct,
        'adx': adx, 'macd_hist': macd_hist, 'pct_b': pct_b, 'rsi': rsi,
        'vol_ratio': vol_ratio, 'bb_squeeze': bb_squeeze,
    }, index=df.index)


def compute_smart_indicators(df):
    """Laatste rij als dict (compat-wrapper)."""
    f = compute_indicator_frame(df)
    last = f.iloc[-1]
    price = float(last['close'])

    def num(x, default):
        return float(x) if pd.notna(x) else default

    return {
        'close': price,
        'ema9': float(last['ema9']),
        'atr': num(last['atr'], price * 0.02),
        'atr_pct': num(last['atr_pct'], 2.0),
        'adx': num(last['adx'], 0.0),
        'macd_hist': float(last['macd_hist']),
        'pct_b': float(last['pct_b']),
        'rsi': num(last['rsi'], 50.0),
        'volume_expansion_ratio': round(float(last['vol_ratio']), 2),
        'is_volume_expansion': bool(last['vol_ratio'] > 1.35),
        'is_bollinger_squeeze': bool(last['bb_squeeze']),
    }


def sr_sd_series(df):
    """Causale S/R + supply/demand per bar. Pivot i is pas bevestigd op bar i+5 (geen lookahead).
    Zones verlopen na ZONE_MAX_AGE_BARS bars."""
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    volumes = df['volume'].values
    n = len(df)
    vol_sma = pd.Series(volumes).rolling(20).mean().values

    supports, resistances, demand, supply = [], [], [], []
    support = np.empty(n)
    resistance = np.empty(n)
    in_dem = np.zeros(n, dtype=bool)
    in_sup = np.zeros(n, dtype=bool)

    for t in range(n):
        i = t - 5
        if i >= 5:
            if lows[i] == lows[i - 5:i + 6].min():
                supports.append(lows[i])
            if highs[i] == highs[i - 5:i + 6].max():
                resistances.append(highs[i])
            vs = vol_sma[i]
            if vs == vs and volumes[i] > vs * 1.5:
                if closes[i] > closes[i - 1]:
                    if closes[i - 1] < opens[i - 1]:
                        demand.append((lows[i - 1], highs[i - 1], i))
                elif closes[i] < closes[i - 1]:
                    if closes[i - 1] > opens[i - 1]:
                        supply.append((lows[i - 1], highs[i - 1], i))

        price = closes[t]
        below = [s for s in supports if s <= price]
        above = [r for r in resistances if r >= price]
        support[t] = max(below) if below else price * 0.90
        resistance[t] = min(above) if above else price * 1.10
        in_dem[t] = any(z[0] <= price <= z[1] and (t - z[2]) <= ZONE_MAX_AGE_BARS for z in demand[-5:])
        in_sup[t] = any(z[0] <= price <= z[1] and (t - z[2]) <= ZONE_MAX_AGE_BARS for z in supply[-5:])

    return {'support': support, 'resistance': resistance, 'in_demand': in_dem, 'in_supply': in_sup}


def detect_support_resistance_and_sd(df):
    """Laatste waarden (compat-wrapper)."""
    s = sr_sd_series(df)
    return {
        'nearest_support': float(s['support'][-1]),
        'nearest_resistance': float(s['resistance'][-1]),
        'in_demand_zone': bool(s['in_demand'][-1]),
        'in_supply_zone': bool(s['in_supply'][-1]),
    }


def compute_primary_setups(df_4h, df_1d):
    """Per 4h-bar: V20 trend-pullback setups en TWM exhaustion-signalen (causaal, geschikt voor historie)."""
    f = compute_indicator_frame(df_4h)
    sr = sr_sd_series(df_4h)
    f['support'] = sr['support']
    f['resistance'] = sr['resistance']
    f['in_demand'] = sr['in_demand']
    f['in_supply'] = sr['in_supply']
    f['timestamp'] = df_4h['timestamp'].values
    f['ret_6'] = f['close'].pct_change(6) * 100.0

    # Dagelijkse ADX, uitgelijnd zonder lookahead: een daily bar is pas bruikbaar als hij gesloten is
    f1 = compute_indicator_frame(df_1d)
    daily = pd.DataFrame({
        'avail_ts': df_1d['timestamp'].values + TF_MS['1d'] - TF_MS['4h'],
        'adx1d': f1['adx'].values,
    }).dropna().sort_values('avail_ts')
    left = pd.DataFrame({'timestamp': f['timestamp'].values}).sort_values('timestamp')
    merged = pd.merge_asof(left, daily, left_on='timestamp', right_on='avail_ts', direction='backward')
    f['adx1d'] = merged['adx1d'].values

    close, atr = f['close'], f['atr']
    has_trend = (f['adx'] >= MIN_ADX_TREND) & (f['adx1d'] > 14.0)
    close_to_support = (close - f['support']) <= 2.0 * atr
    close_to_resistance = (f['resistance'] - close) <= 2.0 * atr

    long_s = (
        has_trend & (f['macd_hist'] > 0)
        & (close >= f['ema9'] * 0.992) & (close <= f['ema9'] * 1.015)
        & (close_to_support | f['in_demand'])
    )
    short_s = (
        has_trend & (f['macd_hist'] < 0)
        & (close <= f['ema9'] * 1.008) & (close >= f['ema9'] * 0.985)
        & (close_to_resistance | f['in_supply'])
    )
    f['setup_long'] = (long_s & ~short_s).fillna(False).astype(bool)
    f['setup_short'] = (short_s & ~long_s).fillna(False).astype(bool)

    # TWM: uitputting buiten de Bollinger-band met volume
    f['twm_short'] = ((f['pct_b'] > 1.02) & (f['rsi'] > 75.0) & (f['vol_ratio'] > 1.4)).fillna(False).astype(bool)
    f['twm_long'] = ((f['pct_b'] < -0.02) & (f['rsi'] < 25.0) & (f['vol_ratio'] > 1.4)).fillna(False).astype(bool)

    # Gegradeerde TWM-score (65..100): hoe extremer volume, RSI en band-overschrijding, hoe hoger
    ext = ((f['pct_b'] - 0.5).abs() - 0.5).clip(lower=0.0)
    rsi_ext = np.where(f['twm_short'], f['rsi'] - 75.0, 25.0 - f['rsi'])
    vol_pts = ((f['vol_ratio'] - 1.4) / 1.1).clip(0, 1) * 12.0
    rsi_pts = (pd.Series(rsi_ext, index=f.index) / 15.0).clip(0, 1) * 12.0
    band_pts = ((ext - 0.02) / 0.20).clip(0, 1) * 11.0
    twm_active = f['twm_short'] | f['twm_long']
    f['twm_score'] = np.where(twm_active, (65.0 + vol_pts + rsi_pts + band_pts).round(2), 50.0)
    return f


# ---------------------------------------------------------------------------
# META-LABELING (gepoold over alle assets, triple barrier, purged walk-forward)
# ---------------------------------------------------------------------------
def event_features(f, idx, sgn):
    """Richting-genormaliseerde features: 1 pooled model voor long én short."""
    sub = f.iloc[list(idx)]
    close = sub['close'].values
    atr = sub['atr'].values
    if sgn == 1:
        lvl = (close - sub['support'].values) / atr
        zone = sub['in_demand'].values
    else:
        lvl = (sub['resistance'].values - close) / atr
        zone = sub['in_supply'].values
    out = pd.DataFrame({
        'sma_dist': sgn * (close - sub['sma20'].values) / sub['sma20'].values,
        'ema9_dist': sgn * (close / sub['ema9'].values - 1.0),
        'macd_n': sgn * sub['macd_hist'].values / close * 100.0,
        'rsi_c': sgn * (sub['rsi'].values - 50.0),
        'pct_b_c': sgn * (sub['pct_b'].values - 0.5),
        'atr_pct': sub['atr_pct'].values,
        'adx4': sub['adx'].values,
        'adx1d': sub['adx1d'].values,
        'vol_ratio': sub['vol_ratio'].values,
        'lvl_dist_atr': np.clip(lvl, 0.0, 10.0),
        'in_zone': zone.astype(float),
        'ret_6': sgn * sub['ret_6'].values,
    })
    return out.replace([np.inf, -np.inf], np.nan)


def barrier_outcome(hi, lo, cl, t, sgn, atr):
    """Triple barrier vanaf bar t. Return (label, net_R) of None als de trade nog niet is afgelopen.
    SL wordt eerst gerekend als beide barriers in dezelfde candle geraakt worden (conservatief)."""
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
        return None  # nog open: niet meenemen (voorkomt label-lek)
    r = sgn * (cl[end] - entry) / (META_SL_ATR * atr)
    return 0, r - cost_r


def build_meta_events(histories):
    """histories: {asset: (df_4h, df_1d)} -> DataFrame met één rij per (gedeclusterd) setup-event."""
    parts = []
    horizon_ms = META_HORIZON_BARS * TF_MS['4h']
    for asset, (d4, d1) in histories.items():
        if d4 is None or d1 is None or len(d4) < 250 or len(d1) < 60:
            continue
        f = compute_primary_setups(d4, d1)
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
        n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=5.0, random_state=42, verbosity=0, n_jobs=1, eval_metric='logloss',
    )


def auc_standard_error(auc, n_pos, n_neg):
    """Hanley & McNeil (1982)."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc ** 2 / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 1e-12))


def breakeven_probability(median_atr_pct):
    cost_r = META_COST_PCT / (META_SL_ATR * max(median_atr_pct, 0.1) / 100.0)
    return (1.0 + cost_r) / (META_TP_ATR / META_SL_ATR + 1.0)


def train_meta_model(events):
    """Purged expanding-window walk-forward over gepoolde events + eindmodel.
    Fail-closed: pack['ok'] is alleen True bij statistisch aantoonbare edge (zie META_* drempels)."""
    pack = {'ok': False, 'reason': 'geen events', 'model': None, 'oos': pd.DataFrame(),
            'n_events': 0, 'n_pos': 0, 'auc': None, 'z': None, 'p_be': None, 'threshold': None,
            'sel_n': 0, 'sel_mean_r': None, 'base_rate': None, 'trained_at': time.time()}
    if events is None or events.empty:
        return pack

    ev = events.sort_values('ts').reset_index(drop=True)
    n = len(ev)
    pack['n_events'] = n
    pack['n_pos'] = int(ev['label'].sum())
    pack['base_rate'] = float(ev['label'].mean())
    p_be = breakeven_probability(float(ev['atr_pct'].median()))
    threshold = p_be + META_MARGIN
    pack['p_be'], pack['threshold'] = float(p_be), float(threshold)

    if n < META_MIN_EVENTS or pack['n_pos'] < META_MIN_POS:
        pack['reason'] = f"te weinig events ({n} events, {pack['n_pos']} winnaars; min {META_MIN_EVENTS}/{META_MIN_POS})"
        return pack

    cuts = [int(n * q) for q in META_FOLD_CUTS] + [n]
    parts = []
    for k in range(len(cuts) - 1):
        test = ev.iloc[cuts[k]:cuts[k + 1]]
        if test.empty:
            continue
        cut_ts = int(test['ts'].iloc[0])
        train = ev[ev['ts_end'] < cut_ts]  # purge: geen trainlabel dat in de testperiode doorloopt
        if len(train) < 60 or train['label'].nunique() < 2:
            continue
        m = make_meta_classifier()
        m.fit(train[META_FEATURES], train['label'])
        part = test[['asset', 'direction', 'ts', 'label', 'net_r']].copy()
        part['prob'] = m.predict_proba(test[META_FEATURES])[:, 1]
        parts.append(part)

    if not parts:
        pack['reason'] = 'geen geldige walk-forward folds'
        return pack

    oos = pd.concat(parts, ignore_index=True)
    pack['oos'] = oos
    n_pos = int(oos['label'].sum())
    n_neg = len(oos) - n_pos
    if n_pos < 5 or n_neg < 5 or oos['label'].nunique() < 2:
        pack['reason'] = 'OOS-set te klein voor AUC'
        return pack

    auc = float(roc_auc_score(oos['label'], oos['prob']))
    se = auc_standard_error(auc, n_pos, n_neg) * META_SE_INFLATION
    z = (auc - 0.5) / se
    sel = oos[oos['prob'] >= threshold]
    sel_mean = float(sel['net_r'].mean()) if len(sel) else None
    pack.update({'auc': auc, 'z': float(z), 'sel_n': int(len(sel)), 'sel_mean_r': sel_mean})

    if z < META_MIN_Z:
        pack['reason'] = f"AUC {auc:.3f} niet significant (z={z:.2f} < {META_MIN_Z})"
        return pack
    if META_REQUIRE_OOS_EV and (len(sel) < META_MIN_SELECTED or sel_mean is None or sel_mean <= 0):
        pack['reason'] = f"OOS-EV niet positief boven drempel (n={len(sel)}, gem. R={sel_mean})"
        return pack

    final = make_meta_classifier()
    final.fit(ev[META_FEATURES], ev['label'])
    pack.update({'ok': True, 'reason': 'ok', 'model': final})
    return pack


_META_CACHE = {'pack': None, 'ts': 0.0}


def get_meta_pack(symbols, force=False):
    now = time.time()
    if not force and _META_CACHE['pack'] is not None and now - _META_CACHE['ts'] < META_RETRAIN_SECONDS:
        return _META_CACHE['pack']

    print(f"🧠 Meta-model hertrainen op {len(symbols)} assets (gepoold, purged walk-forward)...")
    histories = {}
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        d4 = fetch_ohlcv_paginated(exchange, symbol, '4h', META_HISTORY_4H)
        time.sleep(2.0)
        d1 = fetch_cached_1d_shared(exchange, symbol, asset, total_bars=META_HISTORY_1D)
        time.sleep(2.0)
        if len(d4) >= 250 and len(d1) >= 60:
            histories[asset] = (d4, d1)
        else:
            print(f"⚠️ {asset}: te weinig historie ({len(d4)} x 4h, {len(d1)} x 1d), niet in meta-set.")
    try:
        pack = train_meta_model(build_meta_events(histories))
    except Exception as e:
        print(f"❌ Meta-model training mislukt: {e}")
        pack = {'ok': False, 'reason': f'training-fout: {e}', 'model': None, 'oos': pd.DataFrame()}

    _META_CACHE.update({'pack': pack, 'ts': now})
    print(f"🧠 Meta-model: ok={pack['ok']} | reden={pack['reason']} | events={pack.get('n_events')} | "
          f"AUC={pack.get('auc')} | z={pack.get('z')} | drempel={pack.get('threshold')}")
    return pack


# ---------------------------------------------------------------------------
# LIVE ANALYSE
# ---------------------------------------------------------------------------
def _num(x, nd=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def _empty_analysis():
    return {
        'pattern_score': 50.0, 'pattern_signal': 'none', 'twm_signal': 'none', 'twm_score': 50.0,
        'recommended_leverage': 1, 'recommended_trailing_mult': TRAIL_SL_ATR,
        'effective_sl_mult': TRAIL_SL_ATR, 'effective_tp_mult': EFFECTIVE_TP_ATR,
        'min_score_required': MIN_BOSS_SCORE, 'strictness_mode': 'trend_pullback_meta_labeled',
        'sma20': None, 'meta_status': 'insufficient_data', 'meta_prob': None, 'meta_threshold': None,
        'meta_auc': None, 'meta_z': None, 'meta_events': 0, 'breakeven_prob': None,
        'last_candle_ts': 0, 'updated_at': time.time(),
    }


def analyze_advanced_patterns(df_4h, df_1d, asset='BTC', meta_pack=None):
    if df_4h is None or len(df_4h) < 60 or df_1d is None or len(df_1d) < 60:
        return _empty_analysis()

    f = compute_primary_setups(df_4h, df_1d)
    i = len(f) - 1
    last = f.iloc[i]

    signal = 'long' if bool(last['setup_long']) else ('short' if bool(last['setup_short']) else 'none')
    score = 50.0

    twm_signal = 'long' if bool(last['twm_long']) else ('short' if bool(last['twm_short']) else 'none')
    twm_score = float(last['twm_score']) if twm_signal != 'none' else 50.0

    meta_status = 'no_setup'
    meta_prob = None
    pack = meta_pack or {}
    if signal != 'none':
        if not pack.get('ok') or pack.get('model') is None:
            meta_status = f"meta-gate gesloten: {pack.get('reason', 'geen model')}"
            signal = 'none'
        else:
            sgn = 1 if signal == 'long' else -1
            feats = event_features(f, [i], sgn)[META_FEATURES]
            meta_prob = float(pack['model'].predict_proba(feats)[0][1])
            thr = float(pack['threshold'])
            print(f"🔍 {asset} - Setup {signal.upper()} | meta P(TP)={meta_prob:.3f} | drempel={thr:.3f} | AUC={pack.get('auc'):.3f}")
            if meta_prob < thr:
                meta_status = 'meta_prob_below_threshold'
                signal = 'none'
            else:
                meta_status = 'ok'
                score = round(min(100.0, MIN_BOSS_SCORE + 200.0 * (meta_prob - thr)), 2)

    leverage = 3 if score >= 88.0 else (2 if score >= 78.0 else 1)

    raw_ts = float(df_4h['timestamp'].iloc[-1])
    last_candle_timestamp = int(raw_ts / 1000.0 if raw_ts > 1e10 else raw_ts)

    return {
        'pattern_score': float(score),
        'pattern_signal': signal,
        'twm_signal': twm_signal,
        'twm_score': float(twm_score),
        'recommended_leverage': int(leverage),
        'recommended_trailing_mult': TRAIL_SL_ATR,
        'effective_sl_mult': TRAIL_SL_ATR,
        'effective_tp_mult': EFFECTIVE_TP_ATR,
        'min_score_required': MIN_BOSS_SCORE,
        'strictness_mode': 'trend_pullback_meta_labeled',
        'atr_pct': _num(last['atr_pct'], 2),
        'adx_4h': _num(last['adx'], 2),
        'volume_expansion_ratio': _num(last['vol_ratio'], 2),
        'sma20': _num(last['sma20']),
        'meta_status': meta_status,
        'meta_prob': _num(meta_prob, 4),
        'meta_threshold': _num(pack.get('threshold'), 4),
        'meta_auc': _num(pack.get('auc'), 4),
        'meta_z': _num(pack.get('z'), 2),
        'meta_events': int(pack.get('n_events') or 0),
        'breakeven_prob': _num(pack.get('p_be'), 4),
        'last_candle_ts': last_candle_timestamp,
        'updated_at': time.time(),
    }


# ---------------------------------------------------------------------------
# STATE PUSH (zonder commit-spam)
# ---------------------------------------------------------------------------
def _without_ts(d):
    return {k: v for k, v in d.items() if k != 'updated_at'}


def push_state_to_github(filename, data):
    """Schrijft/pusht alleen als de inhoud (zonder updated_at) veranderd is, of als de
    heartbeat verlopen is (zodat traders de state niet als 'te oud' afkeuren)."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                old = json.load(f)
            same = _without_ts(old) == _without_ts(json.loads(json.dumps(data)))
            age = time.time() - float(old.get('updated_at', 0) or 0)
            if same and age < PUSH_HEARTBEAT_SECONDS:
                return
        except Exception as e:
            print(f"⚠️ Kon bestaande state {filename} niet vergelijken: {e}")

    content_bytes = json.dumps(data, indent=2).encode('utf-8')
    with open(filename, 'wb') as f:
        f.write(content_bytes)
    if not github_put(filename, content_bytes, message=f"Auto-update V20 state {filename} [skip ci]", agent='v20-smart-agent'):
        if os.environ.get('GITHUB_TOKEN'):
            print(f"⚠️ State {filename} lokaal bijgewerkt maar NIET naar GitHub gepusht.")


def run_final_boss_scan():
    print("--- [Trend-Pullback Agent] Start marktscan ---")
    pull_ranking_from_github()
    symbols = get_combined_trading_assets(top_n=5)

    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    live = {}
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        d4 = fetch_historical_candles(symbol, '4h', limit=250)
        time.sleep(1.5)
        d1 = fetch_cached_1d_shared(exchange, symbol, asset, total_bars=250)
        live[asset] = (d4, d1)
        time.sleep(2.5)

    meta_pack = get_meta_pack(symbols)

    for asset, (d4, d1) in live.items():
        if not d4.empty and not d1.empty:
            analysis = analyze_advanced_patterns(d4, d1, asset=asset, meta_pack=meta_pack)
            push_state_to_github(f'pattern_state_{asset.lower()}_4h.json', analysis)
            print(f"👑 {asset} Agent: V20={analysis['pattern_signal'].upper()} | TWM={analysis['twm_signal'].upper()} "
                  f"| Score={analysis['pattern_score']} | Meta={analysis['meta_status']}")
        else:
            print(f"⚠️ Geen voldoende candles voor {asset}, overgeslagen.")
        time.sleep(2.0)


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Trend-Pullback Agent gestart.")

    while True:
        try:
            run_final_boss_scan()
        except Exception as e:
            print(f"Fout in scan lus: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)
