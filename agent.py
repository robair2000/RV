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


# --- CONFIGURATIE FINAL BOSS AGENT (ULTIMATE SNIPER MODE) ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = {'15m': '15m', '1h': '1h'}
CHECK_INTERVAL_SECONDS = 300  
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
    De Final Boss Ultimate Sniper Mode: Maximale kwaliteit, 
    minimale frequentie en strikte symmetrische filters.
    """
    if df_15m is None or len(df_15m) < 40 or df_1h is None or len(df_1h) < 40:
        return {
            'pattern_score': 50.0, 
            'pattern_signal': 'none', 
            'recommended_leverage': 1, 
            'recommended_trailing_mult': 1.5,
            'min_score_required': 88.0,
            'strictness_mode': 'strict',
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

        # 2. Bollinger Bands & Squeeze detectie
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        bb_width = (bb_upper - bb_lower) / sma20
        is_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=50).quantile(0.12).iloc[-1]

        # 3. Volume Trend (Extra streng: 2.0x)
        vol_sma = volume.rolling(window=20).mean()
        is_volume_spike = float(volume.iloc[-1]) > float(vol_sma.iloc[-1]) * 2.0

        # 4. Institutionele Price Action (SMC Sniper)
        recent = df.tail(30)
        r_highs = recent['high'].values
        r_lows = recent['low'].values
        r_closes = recent['close'].values
        r_opens = recent['open'].values

        # Fair Value Gaps (FVG) - Strikte disbalans check
        fvg_bullish = False
        fvg_bearish = False
        for i in range(len(recent) - 2):
            if r_lows[i+2] > r_highs[i] and (r_closes[i+1] - r_opens[i+1]) > (r_highs[i+1] - r_lows[i+1]) * 0.5:
                fvg_bullish = True
            if r_highs[i+2] < r_lows[i] and (r_opens[i+1] - r_closes[i+1]) > (r_highs[i+1] - r_lows[i+1]) * 0.5:
                fvg_bearish = True

        # Liquidity Sweep / Stop Hunt (SFP)
        prev_high = r_highs[:-1].max()
        prev_low = r_lows[:-1].min()
        sfp_bullish = r_lows[-1] < prev_low and r_closes[-1] > prev_low
        sfp_bearish = r_highs[-1] > prev_high and r_closes[-1] < prev_high

        # Market Structure Shift / BOS (Harde vereiste)
        bos_bullish = r_closes[-1] > r_highs[-2] and r_closes[-2] > r_closes[-3]
        bos_bearish = r_closes[-1] < r_lows[-2] and r_closes[-2] < r_closes[-3]

        return {
            'rsi': current_rsi,
            'is_squeeze': bool(is_squeeze),
            'is_volume_spike': bool(is_volume_spike),
            'fvg_bullish': bool(fvg_bullish),
            'fvg_bearish': bool(fvg_bearish),
            'sfp_bullish': bool(sfp_bullish),
            'sfp_bearish': bool(sfp_bearish),
            'bos_bullish': bool(bos_bullish),
            'bos_bearish': bool(bos_bearish),
            'trend': float((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]),
            'volatility': float(std20.iloc[-1] / close.iloc[-1])
        }

    ind_15m = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)

    score = 50.0
    signal = 'none'

    # --- ULTIEME SNIPER LOGICA (Symmetrisch & Streng) ---
    is_long_setup = (ind_15m['sfp_bullish'] or ind_15m['fvg_bullish'] or ind_15m['rsi'] < 28) and ind_15m['bos_bullish'] and ind_1h['trend'] > -0.005
    is_short_setup = (ind_15m['sfp_bearish'] or ind_15m['fvg_bearish'] or ind_15m['rsi'] > 72) and ind_15m['bos_bearish'] and ind_1h['trend'] < 0.005

    if is_long_setup and not is_short_setup:
        signal = 'long'
        score = 75.0
        if ind_15m['sfp_bullish']: score += 7.0  
        if ind_15m['fvg_bullish']: score += 6.0  
        if ind_15m['bos_bullish']: score += 6.0  
        if ind_15m['is_volume_spike']: score += 9.0
        if ind_15m['is_squeeze']: score += 5.0

    elif is_short_setup and not is_long_setup:
        signal = 'short'
        score = 75.0
        if ind_15m['sfp_bearish']: score += 7.0
        if ind_15m['fvg_bearish']: score += 6.0
        if ind_15m['bos_bearish']: score += 6.0
        if ind_15m['is_volume_spike']: score += 9.0
        if ind_15m['is_squeeze']: score += 5.0

    score = min(float(score), 98.0)

    # --- VERHOOGDE DREMPELS ---
    strictness_mode = 'strict'
    min_score_required = 88.0  
    
    if ind_15m['volatility'] > 0.025:
        strictness_mode = 'flexible'
        min_score_required = 83.0
    else:
        strictness_mode = 'strict'
        min_score_required = 88.0

    leverage = 1
    trailing_mult = 1.5

    if signal != 'none' and score >= min_score_required:
        leverage = 10 if (score >= 92 and ind_15m['is_squeeze']) else 5
        
        if ind_15m['is_squeeze']:
            trailing_mult = 2.0  
        elif score >= 94:
            trailing_mult = 1.2  
        else:
            trailing_mult = 1.5
    else:
        signal = 'none'
        score = 50.0

    return {
        'pattern_score': float(score),
        'pattern_signal': signal,
        'recommended_leverage': int(leverage),
        'recommended_trailing_mult': float(trailing_mult),
        'min_score_required': float(min_score_required),
        'strictness_mode': strictness_mode,
        'rsi_15m': round(ind_15m['rsi'], 2),
        'rsi_1h': round(ind_1h['rsi'], 2),
        'volume_spike': ind_15m['is_volume_spike'],
        'bollinger_squeeze': ind_15m['is_squeeze'],
        'smart_money': {
            'fvg_bullish': ind_15m['fvg_bullish'],
            'fvg_bearish': ind_15m['fvg_bearish'],
            'liquidity_sweep_bullish': ind_15m['sfp_bullish'],
            'liquidity_sweep_bearish': ind_15m['sfp_bearish'],
            'bos_bullish': ind_15m['bos_bullish'],
            'bos_bearish': ind_15m['bos_bearish']
        },
        'updated_at': time.time()
    }


def push_state_to_github(filename, data):
    token = os.environ.get('GITHUB_TOKEN')
    content_bytes = json.dumps(data, indent=2).encode('utf-8')
    
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                existing_bytes = f.read()
            if existing_bytes == content_bytes:
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
        "message": f"Auto-update Final Boss sniper state {filename} [skip ci]",
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
    print("--- [Final Boss Ultimate Sniper] Start marktscan ---")
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    for asset in ASSETS:
        symbol = resolve_symbol(asset)
        if not symbol:
            continue
        
        df_15m = fetch_historical_candles(symbol, '15m', limit=1000)
        df_1h = fetch_historical_candles(symbol, '1h', limit=1000)
        
        if not df_15m.empty and not df_1h.empty:
            analysis = analyze_advanced_patterns(df_15m, df_1h)
            filename = f'pattern_state_{asset.lower()}_15m.json'
            push_state_to_github(filename, analysis)
            print(f"👑 {asset} Sniper: Signaal={analysis['pattern_signal'].upper()} | Score={analysis['pattern_score']} (Min: {analysis['min_score_required']})")
        time.sleep(5.0)


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Final Boss Ultimate Sniper Agent gestart.")

    while True:
        try:
            run_final_boss_scan()
        except Exception as e:
            print(f"Fout in Final Boss scan lus: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)
