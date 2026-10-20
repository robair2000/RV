import os
import json
import time
import ccxt
import pandas as pd
import numpy as np
from github_utils import github_put
from market_utils import fetch_ohlcv_paginated

EXCHANGE = ccxt.kucoinfutures({'enableRateLimit': True})

def fetch_4h_candles(symbol, limit=1000):
    try:
        # Haalt 4h candles op via de gedeelde paginated helper
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
    while True:
        run_4h_markov_scan()
        time.sleep(3600)  # Elk uur verversen voor actuele 4h stand
