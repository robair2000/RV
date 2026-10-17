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
        pass  # Houd de logs schoon van standaard HTTP access logs


def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"🌐 Web service health check server gestart op poort {port}")
    server.serve_forever()


# --- CONFIGURATIE FINAL BOSS AGENT ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = {'15m': '15m', '1h': '1h'}
CHECK_INTERVAL_SECONDS = 300  # Elke 5 minuten markten analyseren
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


def analyze_advanced_patterns(df_15m, df_1h):
    """
    De Final Boss analyse: combineert candlestick patronen, Bollinger Band squeezes,
    RSI momentum, multi-timeframe convergentie en bepaalt dynamisch leverage + trailing stops.
    """
    if df_15m is None or len(df_15m) < 40 or df_1h is None or len(df_1h) < 40:
        return {
            'pattern_score': 50.0, 
            'pattern_signal': 'none', 
            'recommended_leverage': 1, 
            'recommended_trailing_mult': 1.5,
            'confidence': 0.0
        }

    def compute_indicators(df):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 1. RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])

        # 2. Bollinger Bands & Squeeze detectie (Volatiliteitsknijp)
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        bb_width = (bb_upper - bb_lower) / sma20
        is_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=50).quantile(0.2).iloc[-1]

        # 3. Volume Trend
        vol_sma = volume.rolling(window=20).mean()
        is_volume_spike = float(volume.iloc[-1]) > float(vol_sma.iloc[-1]) * 1.4

        # 4. Candlestick patronen (Engulfing)
        is_bull_eng = (close.iloc[-1] > df['open'].iloc[-1]) and (close.iloc[-2] < df['open'].iloc[-2]) and (close.iloc[-1] > df['open'].iloc[-2])
        is_bear_eng = (close.iloc[-1] < df['open'].iloc[-1]) and (close.iloc[-2] > df['open'].iloc[-2]) and (close.iloc[-1] < df['open'].iloc[-2])
        
        return {
            'rsi': current_rsi,
            'is_squeeze': bool(is_squeeze),
            'is_volume_spike': bool(is_volume_spike),
            'is_bull_eng': bool(is_bull_eng),
            'is_bear_eng': bool(is_bear_eng),
            'trend': float((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5])
        }

    ind_15m = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)

    score = 50.0
    signal = 'none'

    # Logica voor LONG
    if (ind_15m['is_bull_eng'] or ind_15m['rsi'] < 35) and ind_1h['trend'] > -0.01:
        signal = 'long'
        score = 75.0
        if ind_15m['is_volume_spike']: score += 10.0
        if ind_15m['is_squeeze']: score += 10.0  # Breakout potentie
        if ind_1h['trend'] > 0: score += 5.0     # Multi-timeframe steun

    # Logica voor SHORT
    elif (ind_15m['is_bear_eng'] or ind_15m['rsi'] > 65) and ind_1h['trend'] < 0.01:
        signal = 'short'
        score = 75.0
        if ind_15m['is_volume_spike']: score += 10.0
        if ind_15m['is_squeeze']: score += 10.0
        if ind_1h['trend'] < 0: score += 5.0

    score = min(float(score), 98.0)

    # --- DYNAMISCHE LEVERAGE & TRAILING STOP DOOR DE FINAL BOSS ---
    leverage = 1
    trailing_mult = 1.5

    if signal != 'none':
        if score >= 85 and ind_15m['is_squeeze']:
            leverage = 15
        elif score >= 75:
            leverage = 10
        else:
            leverage = 5

        if ind_15m['is_squeeze']:
            trailing_mult = 2.0  # Meer ademruimte tijdens breakout
        elif score >= 90:
            trailing_mult = 1.2  # Zeer sterke overtuiging, strakke bescherming
        else:
            trailing_mult = 1.5  # Standaard ATR multiplier
    else:
        score = 50.0

    return {
        'pattern_score': float(score),
        'pattern_signal': signal,
        'recommended_leverage': int(leverage),
        'recommended_trailing_mult': float(trailing_mult),
        'rsi_15m': round(ind_15m['rsi'], 2),
        'rsi_1h': round(ind_1h['rsi'], 2),
        'volume_spike': ind_15m['is_volume_spike'],
        'bollinger_squeeze': ind_15m['is_squeeze'],
        'updated_at': time.time()
    }


def push_state_to_github(filename, data):
    token = os.environ.get('GITHUB_TOKEN')
    content_bytes = json.dumps(data, indent=2).encode('utf-8')
    with open(filename, 'wb') as f:
        f.write(content_bytes)

    if not token:
        print(f"⚠️ Geen GITHUB_TOKEN gevonden; state {filename} lokaal bijgewerkt.")
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
        "message": f"Auto-update Final Boss state {filename} [skip ci]",
        "content": base64.b64encode(content_bytes).decode('ascii'),
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    try:
        req_data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, headers=headers, method='PUT')
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"☁️ Final Boss state {filename} gepusht naar GitHub.")
    except Exception as e:
        print(f"⛔ Fout bij pushen {filename}: {e}")


def run_final_boss_scan():
    print("--- [Final Boss Agent] Start evaluatie ---")
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    for asset in ASSETS:
        symbol = resolve_symbol(asset)
        if not symbol:
            continue
        
        df_15m = fetch_historical_candles(symbol, '15m', limit=120)
        df_1h = fetch_historical_candles(symbol, '1h', limit=120)
        
        if not df_15m.empty and not df_1h.empty:
            analysis = analyze_advanced_patterns(df_15m, df_1h)
            filename = f'pattern_state_{asset.lower()}_15m.json'
            push_state_to_github(filename, analysis)
            print(f"👑 {asset} Final Boss: Signaal={analysis['pattern_signal'].upper()} | Score={analysis['pattern_score']} | Lev={analysis['recommended_leverage']}x | Trail={analysis['recommended_trailing_mult']}x")
        time.sleep(2.0)


if __name__ == '__main__':
    # Start de webserver op een achtergrondthread voor Render port-binding & health checks
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Final Boss Agent Web Service gestart.")

    while True:
        try:
            run_final_boss_scan()
        except Exception as e:
            print(f"Fout in Final Boss scan lus: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)
