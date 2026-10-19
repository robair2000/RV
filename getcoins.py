import os
import ccxt
import pandas as pd
import numpy as np
import time
import math

# --- CONFIGURATIE ---
EXCHANGE = ccxt.kucoinfutures({
    'enableRateLimit': True
})
TIMEFRAME = '4h'
MIN_DAILY_VOLUME_USDT = 25_000_000  # Minimum 25 miljoen USDT 24h volume voor echte institutionele liquiditeit
CORE_SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']

def fetch_top_performing_coins():
    print("🔄 Laden van KuCoin Futures markten en tickers...")
    try:
        EXCHANGE.load_markets()
    except Exception as e:
        print(f"❌ Fout bij laden markten: {e}")
        return

    try:
        tickers = EXCHANGE.fetch_tickers()
    except Exception:
        tickers = {}

    # Filter uitsluitend op actieve lineaire USDT perpetual swaps
    valid_symbols = []
    for symbol, market in EXCHANGE.markets.items():
        if market.get('linear', False) and market.get('active', True) and symbol.endswith(':USDT'):
            if '/' in symbol and 'DOWN' not in symbol and 'UP' not in symbol:
                valid_symbols.append(symbol)

    valid_symbols = list(set(valid_symbols))
    print(f"📊 Gevonden geldige USDT Futures: {len(valid_symbols)}. Sorteren op werkelijke 24u liquiditeit...")

    # STAP 1: Sorteer ALLE symbolen direct op 24u volume (Liquiditeit Eerst!)
    symbol_volumes = []
    for symbol in valid_symbols:
        ticker = tickers.get(symbol, {})
        vol = float(ticker.get('quoteVolume') or 0.0)
        symbol_volumes.append((symbol, vol))

    # Sorteer aflopend op volume
    symbol_volumes.sort(key=lambda x: x[1], reverse=True)

    # Neem de top 60 meest liquide munten, maar zorg dat CORE_SYMBOLS er ALTIJD in zitten
    top_liquid_symbols = [s for s, v in symbol_volumes[:60]]
    for core in CORE_SYMBOLS:
        if core in valid_symbols and core not in top_liquid_symbols:
            top_liquid_symbols.append(core)

    print(f"🎯 Geselecteerd voor diepgaande analyse (Top liquiditeit + Core): {len(top_liquid_symbols)} symbolen.")
    
    results = []

    for symbol in top_liquid_symbols:
        asset = symbol.split('/')[0]
        ticker = tickers.get(symbol, {})
        quote_volume_24h = float(ticker.get('quoteVolume') or 0.0)
        
        try:
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
            if not ohlcv or len(ohlcv) < 30:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
            
            close = df['close']
            high = df['high']
            low = df['low']
            
            recent_quote_vol = (df['volume'].tail(24) * close.tail(24)).mean() * 6
            if quote_volume_24h == 0:
                quote_volume_24h = recent_quote_vol

            if quote_volume_24h < MIN_DAILY_VOLUME_USDT and symbol not in CORE_SYMBOLS:
                continue

            pc = close.shift(1)
            tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_pct = (atr / close).mean() * 100.0
            
            sma20 = close.rolling(20).mean()
            trend_strength = ((close - sma20) / sma20).abs().mean() * 100
            
            bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

            # GEZONDE SCORING: Beloon diepe liquiditeit en stabiele trend, straf extreme grillige volatiliteit af
            liquidity_score = min(math.log10(quote_volume_24h + 1), 12.0)
            
            # Volatility penalty: te extreme ATR (>8%) duidt op jumpy illiquide gedrag of excessief risico
            volatility_penalty = max(0.0, (atr_pct - 6.0) * 0.8) if atr_pct > 6.0 else 0.0

            composite_score = round((trend_strength * 1.5) + (liquidity_score * 3.0) - volatility_penalty, 2)

            results.append({
                'Asset': asset,
                'Symbol': symbol,
                'Buy_Hold_%': round(bh_return, 2),
                'ATR_%': round(atr_pct, 2),
                'Trend_Score': round(trend_strength, 2),
                'Volume_24h_USDT': round(quote_volume_24h, 0),
                'Score': composite_score
            })
            
        except Exception:
            continue
            
        time.sleep(0.03)

    if not results:
        print("\n⚠️ Geen resultaten voldeden aan het strenge liquiditeitsfilter.")
        return

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    print("\n\n" + "="*65)
    print("🏆 GEOPTIMALISEERDE INSTITUTIONELE LIQUIDITEITS-RANKING")
    print("="*65)
    print(res_df.head(15).to_string(index=False))
    print("="*65)

    res_df.to_csv('kucoin_best_coins_ranking.csv', index=False)
    print("\n📁 Schone liquiditeitsgefilterde ranking opgeslagen in: kucoin_best_coins_ranking.csv")

if __name__ == '__main__':
    fetch_top_performing_coins()
