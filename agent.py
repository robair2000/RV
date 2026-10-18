import os

# --- OUTPUT & WARNING SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"

import base64
import json
import math
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt
import numpy as np
import pandas as pd
import xgboost as xgb


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


# --- CONFIGURATIE V20 S&R & SUPPLY/DEMAND AGENT ---
def get_combined_trading_assets(top_n=5):
    """
    Combineert altijd de vaste kern (BTC, ETH, LINK) met de 
    dynamische Top N uit de V20 ranking scanner (kucoin_best_coins_ranking.csv).
    """
    core_assets = ['BTC', 'ETH', 'LINK']
    ranking_file = 'kucoin_best_coins_ranking.csv'
    dynamic_assets = []
    
    if os.path.exists(ranking_file):
        try:
            df = pd.read_csv(ranking_file)
            symbols = df['Symbol'].head(top_n).tolist()
            for s in symbols:
                asset_name = s.split('/')[0].split(':')[0]
                if asset_name not in dynamic_assets:
                    dynamic_assets.append(asset_name)
            print(f"📈 Dynamische top {top_n} ingeladen voor agent: {dynamic_assets}")
        except Exception as e:
            print(f"⚠️ Fout bij lezen van ranking CSV in agent: {e}")
            
    combined = list(dict.fromkeys(core_assets + dynamic_assets))
    print(f"🎯 Actieve agent trading scope ({len(combined)} assets): {combined}")
    return combined

# Haal de actieve assets op bij start
ASSETS = get_combined_trading_assets(top_n=5)

TIMEFRAMES = {'4h': '4h', '1d': '1d'}
CHECK_INTERVAL_SECONDS = 1800
EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'kucoin')

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})


def resolve_symbol(asset):
    primary = f"{asset}/USDT:USDT"
    if primary in exchange.markets:
        return primary
    return f"{asset}/USDT"


def fetch_historical_candles(symbol, timeframe, limit=250):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Fout bij ophalen candles voor {symbol} ({timeframe}): {e}")
        return pd.DataFrame()


def compute_smart_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / close) * 100.0

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr_smooth = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / (tr_smooth + 1e-8)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / (tr_smooth + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
    adx = dx.rolling(14).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal_line

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + (std20 * 2.0)
    lower_bb = sma20 - (std20 * 2.0)
    pct_b = (close - lower_bb) / ((upper_bb - lower_bb) + 1e-8)
    ema9 = close.ewm(span=9, adjust=False).mean()

    vol_sma = volume.rolling(20).mean()
    is_vol_expansion = volume.iloc[-1] > (vol_sma.iloc[-1] * 1.15)

    return {
        'close': float(close.iloc[-1]),
        'ema9': float(ema9.iloc[-1]),
        'atr': float(atr.iloc[-1]) if not math.isnan(atr.iloc[-1]) else close.iloc[-1] * 0.02,
        'atr_pct': float(atr_pct.iloc[-1]) if not math.isnan(atr_pct.iloc[-1]) else 2.0,
        'adx': float(adx.iloc[-1]) if not math.isnan(adx.iloc[-1]) else 0.0,
        'macd_hist': float(macd_hist.iloc[-1]),
        'pct_b': float(pct_b.iloc[-1]),
        'is_volume_expansion': bool(is_vol_expansion)
    }


def detect_support_resistance_and_sd(df):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    vol_sma = pd.Series(volumes).rolling(20).mean().values

    support_levels = []
    resistance_levels = []
    demand_zones = []
    supply_zones = []

    for i in range(5, len(df) - 5):
        if lows[i] == min(lows[i-5:i+6]):
            support_levels.append(lows[i])
        if highs[i] == max(highs[i-5:i+6]):
            resistance_levels.append(highs[i])

        if volumes[i] > vol_sma[i] * 1.5:
            if closes[i] > closes[i-1]:
                if i > 0 and closes[i-1] < df['open'].iat[i-1]:
                    demand_zones.append((lows[i-1], highs[i-1]))
            elif closes[i] < closes[i-1]:
                if i > 0 and closes[i-1] > df['open'].iat[i-1]:
                    supply_zones.append((lows[i-1], highs[i-1]))

    current_price = closes[-1]
    valid_supports = [s for s in support_levels if s <= current_price]
    valid_resistances = [r for r in resistance_levels if r >= current_price]

    nearest_support = max(valid_supports) if valid_supports else current_price * 0.90
    nearest_resistance = min(valid_resistances) if valid_resistances else current_price * 1.10

    in_demand_zone = any(zone[0] <= current_price <= zone[1] for zone in demand_zones[-5:]) if demand_zones else False
    in_supply_zone = any(zone[0] <= current_price <= zone[1] for zone in supply_zones[-5:]) if supply_zones else False

    return {
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'in_demand_zone': in_demand_zone,
        'in_supply_zone': in_supply_zone
    }


def train_meta_classifier(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change(3).shift(-3)
    df['target'] = (df['returns'] > 0.005).astype(int)
    
    df['sma20'] = df['close'].rolling(20).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['adx_proxy'] = (df['close'] - df['sma20']).abs() / (df['atr'] + 1e-8)
    
    feats = ['sma20', 'atr', 'adx_proxy', 'volume']
    df_clean = df.dropna(subset=feats + ['target'])
    
    if len(df_clean) < 40:
        return 0.5
    
    X = df_clean[feats]
    y = df_clean['target']
    
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X, y)
    
    latest_feat = pd.DataFrame([{
        'sma20': df['sma20'].iloc[-1],
        'atr': df['atr'].iloc[-1],
        'adx_proxy': df['adx_proxy'].iloc[-1],
        'volume': df['volume'].iloc[-1]
    }])
    
    try:
        prob = float(model.predict_proba(latest_feat)[0][1])
    except Exception:
        prob = 0.5
        
    return prob


def analyze_advanced_patterns(df_4h, df_1d, asset='BTC'):
    if df_4h is None or len(df_4h) < 60 or df_1d is None or len(df_1d) < 60:
        return {
            'pattern_score': 50.0, 
            'pattern_signal': 'none', 
            'recommended_leverage': 1, 
            'recommended_trailing_mult': 2.5, 
            'min_score_required': 84.0,
            'strictness_mode': 'v20_sr_sd_confluence',
            'confidence': 0.0
        }

    ind_4h = compute_smart_indicators(df_4h)
    ind_1d = compute_smart_indicators(df_1d)
    sr_sd = detect_support_resistance_and_sd(df_4h)

    signal = 'none'
    score = 50.0

    has_trend = ind_4h['adx'] > 18.0 and ind_1d['adx'] > 14.0

    close_to_support = (ind_4h['close'] - sr_sd['nearest_support']) <= (2.0 * ind_4h['atr'])
    close_to_resistance = (sr_sd['nearest_resistance'] - ind_4h['close']) <= (2.0 * ind_4h['atr'])

    is_long_setup = (
        has_trend and
        ind_4h['macd_hist'] > 0 and
        ind_4h['close'] >= ind_4h['ema9'] * 0.992 and
        ind_4h['close'] <= ind_4h['ema9'] * 1.015 and
        (close_to_support or sr_sd['in_demand_zone'])
    )

    is_short_setup = (
        has_trend and
        ind_4h['macd_hist'] < 0 and
        ind_4h['close'] <= ind_4h['ema9'] * 1.008 and
        ind_4h['close'] >= ind_4h['ema9'] * 0.985 and
        (close_to_resistance or sr_sd['in_supply_zone'])
    )

    if is_long_setup and not is_short_setup:
        signal = 'long'
        score = 85.0
    elif is_short_setup and not is_long_setup:
        signal = 'short'
        score = 85.0

    if signal != 'none':
        meta_prob = train_meta_classifier(df_4h)
        score += (meta_prob * 15.0)
        
        threshold_map = {'BTC': 0.46, 'ETH': 0.46, 'LINK': 0.40, 'SOL': 0.40, 'AVAX': 0.40}
        min_prob = threshold_map.get(asset.upper(), 0.40)

        if meta_prob < min_prob:
            signal = 'none'
            score = 50.0

    leverage = 3 if score >= 90.0 else 2
    trailing_mult = 2.5

    return {
        'pattern_score': float(score),
        'pattern_signal': signal,
        'recommended_leverage': int(leverage),
        'recommended_trailing_mult': float(trailing_mult),
        'min_score_required': 84.0,
        'strictness_mode': 'v20_sr_sd_confluence',
        'atr_pct': round(ind_4h['atr_pct'], 2),
        'adx_4h': round(ind_4h['adx'], 2),
        'updated_at': time.time()
    }


def push_state_to_github(filename, data):
    token = os.environ.get('GITHUB_TOKEN')
    content_bytes = json.dumps(data, indent=2).encode('utf-8')
    
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                if f.read() == content_bytes:
                    return
        except Exception:
            pass

    with open(filename, 'wb') as f:
        f.write(content_bytes)

    if not token:
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "v20-smart-agent"
    }

    sha = None
    try:
        req = urllib.request.Request(url + f"?ref={GITHUB_BRANCH}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            sha = json.loads(resp.read().decode('utf-8')).get('sha')
    except Exception:
        pass

    payload = {
        "message": f"Auto-update V20 state {filename} [skip ci]",
        "content": base64.b64encode(content_bytes).decode('ascii'),
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    try:
        req_data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, headers=headers, method='PUT')
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception:
        pass


def run_final_boss_scan():
    print("--- [V20 S&R & S&D Agent] Start marktscan ---")
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    for asset in ASSETS:
        symbol = resolve_symbol(asset)
        if not symbol:
            continue
        
        df_4h = fetch_historical_candles(symbol, '4h', limit=250)
        df_1d = fetch_historical_candles(symbol, '1d', limit=250)
        
        if not df_4h.empty and not df_1d.empty:
            analysis = analyze_advanced_patterns(df_4h, df_1d, asset=asset)
            
            pattern_filename = f'pattern_state_{asset.lower()}_4h.json'
            push_state_to_github(pattern_filename, analysis)

            model_data = {
                'data_status': 'ok',
                'signal': analysis['pattern_signal'],
                'atr_pct': analysis.get('atr_pct', 2.0),
                'updated_at': analysis['updated_at']
            }
            model_filename = f'model_state_{asset.lower()}_4h.json'
            push_state_to_github(model_filename, model_data)

            print(f"👑 {asset} V20: Signaal={analysis['pattern_signal'].upper()} | Score={analysis['pattern_score']}")
        time.sleep(5.0)


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 V20 S&R & Supply/Demand Agent gestart.")

    while True:
        try:
            run_final_boss_scan()
        except Exception as e:
            print(f"Fout in V20 scan lus: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)
