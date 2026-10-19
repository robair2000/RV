import os
import json
import time
import ccxt
import pandas as pd
import numpy as np
from github_utils import github_put

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
EXCHANGE = ccxt.kucoinfutures({'enableRateLimit': True})

def fetch_daily_candles(symbol, limit=400):
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, '1d', limit=limit)
        if not ohlcv or len(ohlcv) < 50:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df.iloc[:-1].reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Fout bij ophalen dagelijkse candles voor {symbol}: {e}")
        return pd.DataFrame()

def compute_regime_markov(df, window=20, threshold=0.05):
    df = df.copy()
    
    # 1. Bereken de 20-daagse cumulatieve rendementen
    df['cum_return'] = df['close'].pct_change(window)
    
    # 2. Bepaal het regime: 1 = Bullish (> +5%), -1 = Bearish (< -5%), 0 = Sideways
    def classify_regime(val):
        if pd.isna(val):
            return 'sideways'
        if val > threshold:
            return 'bullish'
        elif val < -threshold:
            return 'bearish'
        else:
            return 'sideways'
            
    df['regime'] = df['cum_return'].apply(classify_regime)
    df['prev_regime'] = df['regime'].shift(1)
    
    clean = df.dropna(subset=['regime', 'prev_regime']).copy()
    if len(clean) < 50:
        return None, "Te weinig data na venster"

    # 3. Bouw de 3x3 Markov transitiematrix (van prev_regime naar regime)
    states = ['bullish', 'sideways', 'bearish']
    transition_matrix = pd.crosstab(clean['prev_regime'], clean['regime'], normalize='index')
    
    # Zorg dat alle staten (bullish, sideways, bearish) aanwezig zijn in de matrix
    transition_matrix = transition_matrix.reindex(index=states, columns=states, fill_value=0.33)
    
    # Huidige regime van de laatste afgesloten dag
    current_regime = clean['regime'].iloc[-1]
    
    # Haal de overgangskansen op voor de huidige state
    next_probs = transition_matrix.loc[current_regime].to_dict()
    
    return {
        'current_regime': current_regime,
        'cum_return_20d_pct': round(float(clean['cum_return'].iloc[-1]) * 100, 2),
        'probabilities_next_day': {
            'bullish': round(float(next_probs.get('bullish', 0)) * 100, 2),
            'sideways': round(float(next_probs.get('sideways', 0)) * 100, 2),
            'bearish': round(float(next_probs.get('bearish', 0)) * 100, 2),
        },
        'updated_at': time.time()
    }, None

def run_regime_markov_scan():
    print("--- 📊 Start 20-daagse Markov Regime & Transitie Scan ($\pm 5\%$) ---")
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
    
    try:
        EXCHANGE.load_markets()
    except Exception as e:
        print(f"Fout bij laden markets: {e}")
        return

    for symbol in core_symbols:
        asset = symbol.split('/')[0].split(':')[0]
        df = fetch_daily_candles(symbol, limit=400)
        
        if df.empty:
            continue
            
        result, err = compute_regime_markov(df)
        if err:
            print(f"⚠️ {asset}: {err}")
            continue
            
        filename = f"markov_regime_state_{asset.lower()}.json"
        content_bytes = json.dumps(result, indent=2).encode('utf-8')
        
        with open(filename, 'wb') as f:
            f.write(content_bytes)
            
        if github_put(filename, content_bytes, message=f"Auto-update Markov regime state {asset} [skip ci]", agent='markov-regime-analyzer'):
            print(f"📈 {asset} Regime: {result['current_regime'].upper()} (20d return: {result['cum_return_20d_pct']}%) | "
                  f"Kans op Bullish morgen: {result['probabilities_next_day']['bullish']}%")
            
        time.sleep(2.0)

if __name__ == '__main__':
    run_regime_markov_scan()
