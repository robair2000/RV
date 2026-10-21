import os
import json
import time
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pandas as pd
import ccxt

from github_utils import github_put
from market_utils import fetch_ohlcv_paginated
from asset_utils import get_combined_trading_assets

# --- Configuraties & Constanten ---
META_HISTORY_4H = 2000
MIN_VOL_RATIO = 1.3
SQUEEZE_PCTL = 0.35
MIN_BOSS_SCORE = 75.0

exchange = ccxt.kucoinfutures({'enableRateLimit': True})


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<html><body><h2>🚀 Squeeze Agent is online</h2></body></html>".encode("utf-8"))

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()


def push_file_to_github(filename):
    if not os.environ.get('GITHUB_TOKEN') or not os.path.exists(filename):
        return
    with open(filename, 'rb') as f:
        content = f.read()
    if github_put(filename, content, agent='squeeze-agent'):
        print(f"☁️ Bestand {filename} gepusht naar GitHub.")


def load_markov_state(asset):
    # Gecorrigeerd naar het juiste bestandsformaat van markov.py
    path = f'markov_4h_state_{asset.lower()}.json'
    if not os.path.exists(path):
        return {'green': 50.0, 'red': 50.0}, 1.0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            probs = data.get('probabilities_next_4h', {'green': 50.0, 'red': 50.0})
            risk_mult = float(data.get('risk_multiplier', 1.0))
            return probs, risk_mult
    except Exception:
        return {'green': 50.0, 'red': 50.0}, 1.0


def compute_indicators(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    df['sma20'] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    
    bb_upper = df['sma20'] + (std20 * 2)
    bb_lower = df['sma20'] - (std20 * 2)
    
    bb_width = (bb_upper - bb_lower) / df['sma20']
    df['bb_width_pctl'] = bb_width.rolling(100).rank(pct=True)
    df['percent_b'] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-8)
    
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = (df['atr'] / close) * 100.0

    return df.dropna().reset_index(drop=True)

def run_agent_scan():
    try:
        subprocess.run(["python", "getcoins.py"], check=True)
        push_file_to_github('kucoin_best_coins_ranking.csv')
    except Exception as e:
        print(f"⚠️ getcoins.py overgeslagen: {e}")

    symbols = get_combined_trading_assets()
    written = []

    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        try:
            print(f"Bezig met ophalen en scannen van {symbol}...")
            d4 = fetch_ohlcv_paginated(exchange, symbol, '4h', total_bars=META_HISTORY_4H)
            if d4 is None or len(d4) < 100:
                continue
            
            # Verwijder de actieve lopende candle zodat we alleen op gesloten candles werken
            d4 = d4.iloc[:-1].reset_index(drop=True)
            if d4.empty:
                continue

            df = compute_indicators(d4)
            if df.empty:
                continue

            markov_probs, risk_mult = load_markov_state(asset)

            latest_signal = 'none'
            score = 50.0
            leverage = 2

            # Pak strikt de allergrootste/laatste gesloten bar van ons gefilterde dataframe
            # Pak strikt de allergrootste/laatste gesloten bar van ons gefilterde dataframe
            last_row = df.iloc[-1]
            last_candle_ts = int(last_row['timestamp'])

            is_squeeze = last_row['bb_width_pctl'] <= SQUEEZE_PCTL
            has_volume = last_row['vol_ratio'] >= MIN_VOL_RATIO

            if is_squeeze and has_volume:
                pct_b = last_row['percent_b']
                
                # Echte uitbraak eisen via de Bollinger %B band-randen in plaats van de middelste lijn
                if pct_b >= 0.90:
                    latest_signal = 'long'
                    score = 85.0
                elif pct_b <= 0.10:
                    latest_signal = 'short'
                    score = 85.0
                else:
                    latest_signal = 'none'
                    score = 50.0

                if latest_signal != 'none':
                    base_lev = 3 if score >= 85 else 2
                    leverage = int(np.clip(round(base_lev * risk_mult), 1, 5))

            atr_pct_val = float(last_row['atr_pct']) if 'atr_pct' in last_row else 2.0

            state = {
                'updated_at': time.time(),
                'last_candle_ts': last_candle_ts,
                'pattern_signal': latest_signal,
                'pattern_score': score,
                'recommended_leverage': leverage,
                'risk_multiplier': risk_mult,
                'atr_pct': atr_pct_val,
                'min_score_required': MIN_BOSS_SCORE
            }

            filename = f'pattern_state_{asset.lower()}_4h.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            push_file_to_github(filename)
            written.append(filename)
            print(f"🚀 Squeeze Agent geüpdatet voor {asset}: signal={latest_signal}, lev={leverage} (risk_mult={risk_mult})")

            time.sleep(3.0)

        except Exception as e:
            print(f"❌ Fout bij agent scan voor {asset}: {e}")

    print(f"✅ Agent scan voltooid. {len(written)} bestanden bijgewerkt.")


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Squeeze Agent Web Service gestart met ingebouwde candle-sync.")

    CANDLE_SECONDS = 4 * 3600
    SCAN_DELAY_SECONDS = 60  # Wacht 1 minuut na candle sluiting

    def seconds_until_next_scan():
        now = time.time()
        next_close = (int(now // CANDLE_SECONDS) + 1) * CANDLE_SECONDS
        return max(5, next_close + SCAN_DELAY_SECONDS - now)

    while True:
        try:
            print("🔍 Start gesynchroniseerde Squeeze Agent scan na candle close...")
            run_agent_scan()
        except Exception as e:
            print(f"⚠️ Fout in agent hoofdloop: {e}")
        
        sleep_seconds = seconds_until_next_scan()
        print(f"💤 Squeeze Agent wacht tot de volgende 4h candle cyclus (over {int(sleep_seconds // 60)} minuten)...")
        time.sleep(sleep_seconds)
