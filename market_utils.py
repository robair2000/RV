"""Gedeelde marktdata-helpers (agent.py, quant_worker.py, backtest.py)."""
import os
import json
import time

import pandas as pd

from github_utils import github_pull, github_put

TF_MS = {
    '1h': 3_600_000,
    '4h': 14_400_000,
    '1d': 86_400_000,
}


def tf_to_ms(timeframe):
    return TF_MS[timeframe]


def fetch_ohlcv_paginated(exchange, symbol, timeframe, total_bars, page_limit=200, max_pages=40, pause=2.0):
    """Haalt tot `total_bars` candles op via meerdere calls (since-paginatie).

    KuCoin Futures geeft per call maar een beperkt aantal candles terug; zonder
    paginatie blijft de dataset te klein voor betrouwbare walk-forward validatie.
    Geeft een DataFrame terug met kolommen timestamp(ms)/open/high/low/close/volume,
    oplopend gesorteerd, ZONDER de laatste (nog lopende) candle.
    """
    step = tf_to_ms(timeframe)
    now_ms = int(time.time() * 1000)
    since = now_ms - int(total_bars) * step
    rows = []
    pages = 0
    while pages < max_pages:
        pages += 1
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=page_limit)
        except Exception as e:
            print(f"⚠️ Paginated fetch fout {symbol} {timeframe}: {e}")
            break
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + step
        if next_since <= since or last_ts >= now_ms - step:
            break
        since = next_since
        time.sleep(pause)

    if not rows:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    # Laatste candle is nog niet gesloten -> weg (voorkomt repainting)
    if len(df) > 0 and df['timestamp'].iloc[-1] + step > now_ms:
        df = df.iloc[:-1].reset_index(drop=True)
    return df.tail(int(total_bars)).reset_index(drop=True)


def fetch_cached_1d_shared(exchange, symbol, asset, total_bars=450, max_age_seconds=86400):
    """Haalt 1d candles op via een gedeelde lokale cache (en optioneel GitHub) om rate limits te voorkomen."""
    cache_file = f"shared_cache_1d_{asset.lower()}.json"
    now = time.time()

    # 1. Probeer lokaal te laden
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if now - cached.get('timestamp', 0) < max_age_seconds:
                df = pd.DataFrame(cached['data'])
                if not df.empty:
                    return df
        except Exception:
            pass

    # 2. Indien niet lokaal vers, probeer te pullen van GitHub voor het geval de Quant Worker het heeft geupdate
    if os.environ.get('GITHUB_TOKEN'):
        try:
            if github_pull(cache_file, min_interval=3600, agent='shared-cache'):
                if os.path.exists(cache_file):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                    if now - cached.get('timestamp', 0) < max_age_seconds:
                        df = pd.DataFrame(cached['data'])
                        if not df.empty:
                            return df
        except Exception:
            pass

    # 3. Als het echt nergens vers is, haal vers op via de API
    df = fetch_ohlcv_paginated(exchange, symbol, '1d', total_bars=total_bars)
    if not df.empty:
        try:
            payload = {
                'timestamp': now,
                'data': df.to_dict(orient='records')
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            
            # Direct pushen naar GitHub zodat de andere worker (of agent) het ook direct heeft
            if os.environ.get('GITHUB_TOKEN'):
                with open(cache_file, 'rb') as f_bin:
                    github_put(cache_file, f_bin.read(), message=f"Auto-update {cache_file} [skip ci]", agent='shared-cache')
        except Exception as e:
            print(f"⚠️ Kon 1d cache niet wegschrijven/pushten voor {asset}: {e}")

    return df


def fetch_cached_4h_shared(exchange, symbol, asset, total_bars=2000, max_age_seconds=14400):
    """Haalt 4h candles op via lokale cache (geldig voor 4 uur) om rate limits bij meta-training te voorkomen."""
    cache_file = f"shared_cache_4h_{asset.lower()}.json"
    now = time.time()

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if now - cached.get('timestamp', 0) < max_age_seconds:
                df = pd.DataFrame(cached['data'])
                if not df.empty:
                    return df
        except Exception:
            pass

    df = fetch_ohlcv_paginated(exchange, symbol, '4h', total_bars=total_bars)
    if not df.empty:
        try:
            payload = {
                'timestamp': now,
                'data': df.to_dict(orient='records')
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
        except Exception as e:
            print(f"⚠️ Kon 4h cache niet wegschrijven voor {asset}: {e}")

    return df
