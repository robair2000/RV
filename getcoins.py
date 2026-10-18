import os
import ccxt
import pandas as pd
import numpy as np
import time
import math

# --- CONFIGURATIE ---
EXCHANGE = ccxt.kucoin({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
TIMEFRAME = '4h'
MIN_DAILY_VOLUME_USDT = 20_000_000  # Minimum 20 miljoen USDT 24h volume voor liquiditeitsbescherming

def fetch_top_performing_coins():
    print("🔄 Laden van KuCoin Futures markten...")
    try:
        EXCHANGE.load_markets()
    except Exception as e:
        print(f"❌ Fout bij laden markten: {e}")
        return

    # Haal ook tickers op om direct 24h volumes te controleren
    try:
        tickers = EXCHANGE.fetch_tickers()
    except Exception:
        tickers = {}

    # Filter ABSOLUUT uitsluitend op actieve lineaire USDT perpetual swaps (eindigend op :USDT)
    symbols = []
    for symbol, market in EXCHANGE.markets.items():
        if market.get('linear', False) and market.get('active', True) and symbol.endswith(':USDT'):
            if '/' in symbol and not 'DOWN' in symbol and not 'UP' in symbol:
                symbols.append(symbol)

    symbols = list(set(symbols))
    print(f"📊 Gevonden geldige USDT Futures: {len(symbols)}. Start liquiditeits- en trendanalyse...")
    
    results = []

    # Prioriteer de belangrijkste kernmunten
    priority_assets = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'LINK/USDT:USDT', 'AVAX/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT']
    test_symbols = [s for s in priority_assets if s in symbols] + [s for s in symbols if s not in priority_assets]

    # Beperk tot de top 50 om performance te waarborgen en vertraging te voorkomen
    for symbol in test_symbols[:50]:
        asset = symbol.split('/')[0]
        
        # 1. LIQUIDITEITSFILTER VIA TICKER (24u Volume in USDT)
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

            if quote_volume_24h < MIN_DAILY_VOLUME_USDT and symbol not in priority_assets:
                continue

            pc = close.shift(1)
            tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_pct = (atr / close).mean() * 100.0
            
            sma20 = close.rolling(20).mean()
            trend_strength = ((close - sma20) / sma20).abs().mean() * 100
            
            bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

            liquidity_score = min(math.log10(quote_volume_24h + 1), 10.0)
            composite_score = round((trend_strength * 1.2) + (atr_pct * 1.5) + (liquidity_score * 2.0), 2)

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
            
        time.sleep(0.05)

    if not results:
        print("\n⚠️ Geen resultaten voldeden aan het liquiditeitsfilter.")
        return

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    print("\n\n" + "="*65)
    print("🏆 GEOPTIMALISEERDE V20 INSTITUTIONELE FUTURES RANKING")
    print("="*65)
    print(res_df.head(15).to_string(index=False))
    print("="*65)

    res_df.to_csv('kucoin_best_coins_ranking.csv', index=False)
    print("\n📁 Schone liquiditeitsgefilterde ranking opgeslagen in: kucoin_best_coins_ranking.csv")

if __name__ == '__main__':
    fetch_top_performing_coins()
