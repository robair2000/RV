import os
import ccxt
import pandas as pd
import numpy as np
import time

# --- CONFIGURATIE ---
EXCHANGE = ccxt.kucoin({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
TIMEFRAME = '4h'

def fetch_top_performing_coins():
    print("🔄 Laden van KuCoin Futures markten...")
    try:
        EXCHANGE.load_markets()
    except Exception as e:
        print(f"❌ Fout bij laden markten: {e}")
        return

    # Filter ABSOLUUT uitsluitend op actieve lineaire USDT perpetual swaps (eindigend op :USDT)
    symbols = []
    for symbol, market in EXCHANGE.markets.items():
        if market.get('linear', False) and market.get('active', True) and symbol.endswith(':USDT'):
            # Zorg ervoor dat het een geldige futures notatie is met een slash en geen index of rare tokens
            if '/' in symbol and not 'DOWN' in symbol and not 'UP' in symbol:
                symbols.append(symbol)

    symbols = list(set(symbols))
    print(f"📊 Gevonden geldige en verhandelbare USDT Futures: {len(symbols)}. Start analyse...")
    
    results = []

    # Prioriteer de belangrijkste liquiditeitstoppers
    priority_assets = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'LINK/USDT:USDT', 'AVAX/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT']
    test_symbols = [s for s in priority_assets if s in symbols] + [s for s in symbols if s not in priority_assets]

    for symbol in test_symbols[:50]: # Test de top 50 geldige futures
        asset = symbol.split('/')[0]
        print(f"🔍 Analyseren van {asset:<10}", end="\r")
        
        try:
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=150)
            if not ohlcv or len(ohlcv) < 30:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
            
            close = df['close']
            high = df['high']
            low = df['low']
            
            pc = close.shift(1)
            tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_pct = (atr / close).mean() * 100.0
            
            sma20 = close.rolling(20).mean()
            trend_strength = ((close - sma20) / sma20).abs().mean() * 100
            
            bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
            avg_volume = df['volume'].tail(30).mean() * close.tail(30).mean()

            results.append({
                'Asset': asset,
                'Symbol': symbol,
                'Buy_Hold_%': round(bh_return, 2),
                'ATR_%': round(atr_pct, 2),
                'Trend_Score': round(trend_strength, 2),
                'Avg_Volume_USDT': round(avg_volume, 0),
                'Score': round((trend_strength * 1.5) + (atr_pct * 2.0) + (1 if bh_return > 0 else 0), 2)
            })
            
        except Exception as e:
            continue
            
        time.sleep(0.1)

    if not results:
        print("\n⚠️ Geen resultaten gevonden.")
        return

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    print("\n\n" + "="*65)
    print("🏆 TOP KUCOIN FUTURES VOOR V20 SWING TRADING")
    print("="*65)
    print(res_df.head(15).to_string(index=False))
    print("="*65)

    res_df.to_csv('kucoin_best_coins_ranking.csv', index=False)
    print("\n📁 Volledige gezoneerde ranking opgeslagen in: kucoin_best_coins_ranking.csv")

if __name__ == '__main__':
    fetch_top_performing_coins()
