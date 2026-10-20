import os

# --- OUTPUT & WARNING SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import ccxt
import pandas as pd
import numpy as np

from github_utils import github_put
from market_utils import fetch_ohlcv_paginated

# --- RENDER WEB SERVER (NODIG VOOR UPTIMEROBOT) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK - Markov 4h Worker is running")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Onderdrukt log-ruis van UptimeRobot pings

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"🌐 Markov Web service health check server gestart op poort {port}")
    server.serve_forever()


# --- MARKOV WORKER LOGICA ---
EXCHANGE = ccxt.kucoinfutures({'enableRateLimit': True})

def fetch_4h_candles(symbol, limit=1000):
    try:
        df = fetch_ohlcv_paginated(EXCHANGE, symbol, '4h', total_bars=limit)
        if df.empty or len(df) < 100:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"⚠️ Fout bij ophalen 4h candles voor {symbol}: {e}")
        return pd.DataFrame()

def compute_4h_candle_markov(df):
    df = df.copy()
    # 1. Bepaal of de candle groen of rood is
    df['color'] = np.where(df['close'] >= df['open'], 'green', 'red')
    df['prev_color'] = df['color'].shift(1)
    
    clean = df.dropna(subset=['color', 'prev_color']).copy()
    if len(clean) < 50:
        return None, "Te weinig data voor 4h markov"

    # 2. Bouw de 2x2 transitiematrix (van prev_color naar color)
    states = ['green', 'red']
    transition_matrix = pd.crosstab(clean['prev_color'], clean['color'], normalize='index')
    transition_matrix = transition_matrix.reindex(index=states, columns=states, fill_value=0.5)
    
    # 3. Status van de laatst afgesloten 4h candle
    current_state = clean['color'].iloc[-1]
    next_probs = transition_matrix.loc[current_state].to_dict()
    
    return {
        'current_4h_candle': current_state,
        'probabilities_next_4h': {
            'green': round(float(next_probs.get('green', 0.5)) * 100, 2),
            'red': round(float(next_probs.get('red', 0.5)) * 100, 2),
        },
        'transition_matrix': {
            'after_green': {
                'green': round(float(transition_matrix.loc['green', 'green']) * 100, 2),
                'red': round(float(transition_matrix.loc['green', 'red']) * 100, 2),
            },
            'after_red': {
                'green': round(float(transition_matrix.loc['red', 'green']) * 100, 2),
                'red': round(float(transition_matrix.loc['red', 'red']) * 100, 2),
            }
        },
        'updated_at': time.time()
    }, None

def run_4h_markov_scan():
    print("--- 📊 Start 4h Candle Markov Transitie Scan (Groen/Rood) ---")
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
    
    try:
        EXCHANGE.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        df = fetch_4h_candles(symbol, limit=1000)
        if df.empty:
            continue
            
        result, err = compute_4h_candle_markov(df)
        if err:
            print(f"⚠️ {asset}: {err}")
            continue
            
        filename = f"markov_4h_state_{asset.lower()}.json"
        content_bytes = json.dumps(result, indent=2).encode('utf-8')
        
        with open(filename, 'wb') as f:
            f.write(content_bytes)
            
        if github_put(filename, content_bytes, message=f"Auto-update 4h Markov state {asset} [skip ci]", agent='markov-4h-worker'):
            print(f"📈 {asset} Laatste 4h: {result['current_4h_candle'].upper()} | "
                  f"Kans op Groene volgende 4h: {result['probabilities_next_4h']['green']}%")
            
        time.sleep(2.0)

if __name__ == '__main__':
    # Start de HTTP-server in een achtergrond-thread zodat Render denkt dat dit een web service is
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 4h Markov Worker + Web Service gestart.")

    while True:
        try:
            run_4h_markov_scan()
        except Exception as e:
            print(f"⚠️ Fout in Markov scan lus: {e}")
        time.sleep(3600)  # Elk uur herberekenen en pushen naar GitHub
