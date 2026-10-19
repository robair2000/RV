"""Gedeelde marktdata-helpers (agent.py, quant_worker.py, backtest.py)."""
import time

import pandas as pd

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
