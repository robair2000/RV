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


# --- RENDER WEB SERVER (VOOR POORT-BINDING & HEALTH CHECKS) ---
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


# --- CONFIGURATIE FINAL BOSS AGENT (SWING SNIPER MODE: 4h & 1d) ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = {'4h': '4h', '1d': '1d'}
CHECK_INTERVAL_SECONDS = 1800  # Elke 30 minuten scannen (voldoende voor macro candles)
EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'kucoin')

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})


def resolve_symbol(asset):
    primary = f"{asset}/USDT:USDT"
    if primary in exchange.markets:
        return primary
    return f"{asset}/USDT"


def fetch_historical_candles(symbol, timeframe, limit=120):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Fout bij ophalen candles voor {symbol} ({timeframe}): {e}")
        return pd.DataFrame()


def analyze_advanced_patterns(df_4h, df_1d):
    """
    Final Boss Swing Mode: Analyseert 4h en 1d candles voor macro-trends 
    die 2 tot 3 dagen (of langer) aanhouden.
    """
    if df_4h is None or len(df_4h) < 30 or df_1d is None or len(df_1d) < 30:
        return {
            'pattern_score': 50.0, 
            'pattern_signal': 'none', 
            'recommended_leverage': 1, 
            'recommended_trailing_mult': 2.5,  # Ruimere trailing stop voor swing trades
            'min_score_required': 85.0,
            'strictness_mode': 'strict',
            'confidence': 0.0
        }

    def compute_macro_indicators(df):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 1. RSI (14) op macro niveau
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])

        # 2. Moving Average Trend (50 en 200 equivalent op 4h/1d)
        sma20 = close.rolling(window=20).mean()
        sma50 = close.rolling(window=50).mean() if len(close) >= 50 else sma20
        is_uptrend = close.iloc[-1] > sma20.iloc[-1] and sma20.iloc[-1] > sma50.iloc[-1]
        is_downtrend = close.iloc[-1] < sma20.iloc[-1] and sma20.iloc[-1] < sma50.iloc[-1]

        # 3. Macro Volume & Volatiliteit
        vol_sma = volume.rolling(window=20).mean()
        is_volume_expansion = float(volume.iloc[-1]) > float(vol_sma.iloc[-1]) * 1.5

        # 4. Macro Price Action (Structure Breakouts over meerdere candles)
        recent = df.tail(20)
        r_highs = recent['high'].values
        r_lows = recent['low'].values
        r_closes = recent['close'].values

        macro_breakout_bullish = r_closes[-1] > r_highs[:-1].max() * 0.995
        macro_breakout_bearish = r_closes[-1] < r_lows[:-1].min() * 1.005

        return {
            'rsi': current_rsi,
            'is_uptrend': bool(is_uptrend),
            'is_downtrend': bool(is_downtrend),
            'is_volume_expansion': bool(is_volume_expansion),
            'macro_breakout_bullish': bool(macro_breakout_bullish),
            'macro_breakout_bearish': bool(macro_breakout_bearish),
            'trend_slope': float((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]),
            'volatility': float(close.rolling(window=20).std().iloc[-1] / close.iloc[-1])
        }

    ind_4h = compute_macro_indicators(df_4h)
    ind_1d = compute_macro_indicators(df_1d)

    score = 50.0
    signal = 'none'

    # --- MACRO SWING LOGICA (2-3 DAGEN TRENDS) ---
    is_long_setup = ind_4h['is_uptrend'] and ind_1d['is_uptrend'] and (ind_4h['macro_breakout_bullish'] or ind_4h['rsi'] < 45)
    is_short_setup = ind_4h['is_downtrend'] and ind_1d['is_downtrend'] and (ind_4h['macro_breakout_bearish'] or ind_4h['rsi'] > 55)

    if is_long_setup and not is_short_setup:
        signal = 'long'
        score = 78.0
        if ind_4h['macro_breakout_bullish']: score += 10.0
        if ind_1d['is_uptrend']: score += 8.0
        if ind_4h['is_volume_expansion']: score += 4.0

    elif is_short_setup and not is_long_setup:
        signal = 'short'
        score = 78.0
        if ind_4h['macro_breakout_bearish']: score += 10.0
        if ind_1d['is_downtrend']: score += 8.0
        if ind_4h['is_volume_expansion']: score += 4.0

    score = min(float(score), 98.0)
    min_score_required = 85.0

    # Veilige leverage voor swing trading (lager om liquidatie op lange termijn te vermijden)
    leverage = 3 if score >= 90 else 2
    trailing_mult = 2.5  # Grotere ademruimte voor multi-day swings

    if signal == 'none' or score < min_score_required:
        signal = 'none'
        score = 50.0
        leverage = 1

    return {
        'pattern_score': float(score),
        'pattern_signal': signal,
        'recommended_leverage': int(leverage),
        'recommended_trailing_mult': float(trailing_mult),
        'min_score_required': float(min_score_required),
        'strictness_mode': 'macro_swing',
        'rsi_4h': round(ind_4h['rsi'], 2),
        'rsi_1d': round(ind_1d['rsi'], 2),
        'volume_expansion': ind_4h['is_volume_expansion'],
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
        "User-Agent": "final-boss-agent"
    }

    sha = None
    try:
        req = urllib.request.Request(url + f"?ref={GITHUB_BRANCH}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            sha = json.loads(resp.read().decode('utf-8')).get('sha')
    except Exception:
        pass

    payload = {
        "message": f"Auto-update Swing Boss state {filename} [skip ci]",
        "content": base64.b64encode(content_bytes).decode('ascii'),
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    try:
        req_data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, headers=headers, method='PUT')
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"☁️ Swing Boss state {filename} gepusht naar GitHub.")
    except Exception as e:
        print(f"⛔ Fout bij pushen {filename}: {e}")


def run_final_boss_scan():
    print("--- [Final Boss Macro Swing Agent] Start marktscan ---")
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    for asset in ASSETS:
        symbol = resolve_symbol(asset)
        if not symbol:
            continue
        
        df_4h = fetch_historical_candles(symbol, '4h', limit=500)
        df_1d = fetch_historical_candles(symbol, '1d', limit=200)
        
        if not df_4h.empty and not df_1d.empty:
            analysis = analyze_advanced_patterns(df_4h, df_1d)
            filename = f'pattern_state_{asset.lower()}_4h.json'
            push_state_to_github(filename, analysis)
            print(f"👑 {asset} Swing: Signaal={analysis['pattern_signal'].upper()} | Score={analysis['pattern_score']}")
        time.sleep(5.0)


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Final Boss Macro Swing Agent gestart.")

    while True:
        try:
            run_final_boss_scan()
        except Exception as e:
            print(f"Fout in Final Boss scan lus: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)
