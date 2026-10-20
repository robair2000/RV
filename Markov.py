import os
import time
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pandas as pd
import ccxt

from github_utils import github_put
from asset_utils import get_combined_trading_assets
from market_utils import fetch_ohlcv_paginated

exchange = ccxt.kucoinfutures({'enableRateLimit': True})


# --- Mini Webserver voor Render (Voorkomt 'No open ports detected' crash) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<html><body><h2>📊 Markov Worker is online</h2></body></html>".encode("utf-8"))

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
    if github_put(filename, content, agent='markov-worker'):
        print(f"☁️ Markov state {filename} gepusht naar GitHub.")


def calculate_markov_probabilities(df):
    if df is None or len(df) < 50:
        return {'green': 50.0, 'red': 50.0}
    
    closes = df['close'].values
    opens = df['open'].values
    colors = ['green' if c >= opens[i] else 'red' for i, c in enumerate(closes)]
    
    transitions = {'green': {'green': 0, 'red': 0}, 'red': {'green': 0, 'red': 0}}
    
    for i in range(len(colors) - 1):
        curr = colors[i]
        nxt = colors[i+1]
        transitions[curr][nxt] += 1
        
    last_color = colors[-1]
    total_green = transitions[last_color]['green'] + transitions[last_color]['red']
    
    if total_green == 0:
        return {'green': 50.0, 'red': 50.0}
        
    p_green = (transitions[last_color]['green'] / total_green) * 100.0
    p_red = 100.0 - p_green
    
    return {'green': round(p_green, 2), 'red': round(p_red, 2)}


def run_markov_loop():
    symbols = get_combined_trading_assets()
    
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        try:
            print(f"📊 Markov berekening starten voor {symbol}...")
            df = fetch_ohlcv_paginated(exchange, symbol, '4h', total_bars=500)
            if df is None or len(df) < 50:
                continue
            df = df.iloc[:-1].reset_index(drop=True)  # Lopende candle uitsluiten
            
            probs = calculate_markov_probabilities(df)
            
            state = {
                'updated_at': time.time(),
                'symbol': symbol,
                'probabilities_next_4h': probs
            }
            
            filename = f'markov_state_{asset.lower()}_4h.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
            push_file_to_github(filename)
            print(f"📊 Markov kansen voor {asset}: Groen={probs['green']}% | Rood={probs['red']}%")
            
            # Korte pauze tegen KuCoin rate limits
            time.sleep(2.0)
            
        except Exception as e:
            print(f"⚠️ Fout in Markov berekening voor {symbol}: {e}")


if __name__ == '__main__':
    # Start de mini webserver op de achtergrond voor Render
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Markov 4h Worker Web Service gestart met ingebouwde health-check.")
    
    while True:
        try:
            run_markov_loop()
        except Exception as e:
            print(f"⚠️ Fout in Markov hoofdloop: {e}")
            
        time.sleep(3600)  # Elk uur herberekenen bij nieuwe 4h candles
