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
META_1D_BARS = 450
MIN_VOL_RATIO = 1.3
SQUEEZE_PCTL = 0.35
META_SL_ATR = 1.5
META_TP_ATR = 3.0
MIN_BOSS_SCORE = 75.0

exchange = ccxt.kucoinfutures({'enableRateLimit': True})


# --- Mini Webserver voor Render (Voorkomt 'No open ports detected' crash) ---
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
        pass  # Voorkomt spam in de logs


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
    path = f'markov_state_{asset.lower()}_4h.json'
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

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    
    bb_width = (sma20 + (std20 * 2) - (sma20 - (std20 * 2))) / sma20
    df['bb_width_pctl'] = bb_width.rolling(100).rank(pct=True)
    
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = (df['atr'] / close) * 100.0

    return df.dropna().reset_index(drop=True)


def compute_primary_setups(d4, d1):
    df = compute_indicators(d4)
    if df.empty:
        return pd.DataFrame()

    setups = []
    for i in range(50, len(df) - 1):  # Lopende candle uitsluiten
        row = df.iloc[i]
        
        is_squeeze = row['bb_width_pctl'] <= SQUEEZE_PCTL
        has_volume = row['vol_ratio'] >= MIN_VOL_RATIO
        
        if not (is_squeeze and has_volume):
            continue

        close = row['close']
        sma20 = d4['close'].rolling(20).mean().iloc[i]
        signal = 'long' if close > sma20 else 'short'

        # Markov filter is hier verwijderd: Markov blokkeert geen entries meer, 
        # maar beïnvloedt straks puur de risk multiplier / leverage.

        setups.append({
            'timestamp': int(row['timestamp']),
            'close': close,
            'atr': row['atr'],
            'signal': signal,
            'bb_width_pctl': row['bb_width_pctl'],
            'vol_ratio': row['vol_ratio']
        })

    return pd.DataFrame(setups)


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
            d4 = d4.iloc[:-1].reset_index(drop=True)

            markov_probs, risk_mult = load_markov_state(asset)
            setups = compute_primary_setups(d4, None)

            latest_signal = 'none'
            score = 50.0
            leverage = 2

            if not setups.empty:
                last_setup = setups.iloc[-1]
                latest_signal = last_setup['signal']
                score = 80.0
                
                # Basishefboom vermenigvuldigd met de Markov risk_multiplier (begrensd tussen 1x en 5x)
                base_lev = 3 if score >= 85 else 2
                leverage = int(np.clip(round(base_lev * risk_mult), 1, 5))

            state = {
                'updated_at': time.time(),
                'last_candle_ts': int(d4['timestamp'].iloc[-1]),
                'pattern_signal': latest_signal,
                'pattern_score': score,
                'recommended_leverage': leverage,
                'risk_multiplier': risk_mult,
                'atr_pct': float(d4['atr_pct'].iloc[-1]) if 'atr_pct' in d4 else 2.0,
                'min_score_required': MIN_BOSS_SCORE
            }

            filename = f'pattern_state_{asset.lower()}_4h.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            push_file_to_github(filename)
            written.append(filename)
            print(f"🚀 Squeeze Agent geüpdatet voor {asset}: signal={latest_signal}, lev={leverage} (risk_mult={risk_mult})")

            # Cruciaal: rustpauze tussen munten om KuCoin rate limits (429) te voorkomen
            time.sleep(3.0)

        except Exception as e:
            print(f"❌ Fout bij agent scan voor {asset}: {e}")

    print(f"✅ Agent scan voltooid. {len(written)} bestanden bijgewerkt.")


if __name__ == '__main__':
    # Start de mini webserver op de achtergrond zodat Render een open poort ziet
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Squeeze Agent Web Service gestart met ingebouwde health-check.")

    # Continue loop die de scans periodiek uitvoert
    while True:
        try:
            run_agent_scan()
        except Exception as e:
            print(f"⚠️ Fout in agent hoofdloop: {e}")
        
        # Wacht 30 minuten tot de volgende scan
        time.sleep(1800)
