import os
import time
import json
import numpy as np
import pandas as pd
import ccxt

from github_utils import github_put
from asset_utils import get_combined_trading_assets
from market_utils import fetch_ohlcv_paginated

exchange = ccxt.kucoinfutures({'enableRateLimit': True})


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
    
    # Bepaal of candles groen of rood zijn
    closes = df['close'].values
    opens = df['open'].values
    colors = ['green' if c >= o else 'red' for c, o in zip(closes, opens)]
    
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
            # Haal voldoende 4h data op en sluit de lopende candle uit
            df = fetch_ohlcv_paginated(exchange, symbol, '4h', total_bars=500)
            if df is None or len(df) < 50:
                continue
            df = df.iloc[:-1].reset_index(drop=True)
            
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
            
        except Exception as e:
            print(f"⚠️ Fout in Markov berekening voor {symbol}: {e}")


if __name__ == '__main__':
    print("🚀 Markov 4h Worker gestart...")
    while True:
        run_markov_loop()
        time.sleep(3600)  # Elk uur herberekenen bij nieuwe 4h candles
