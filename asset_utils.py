"""Gedeelde asset-selectie voor agent, quant worker en beide traders.

Eén definitie i.p.v. vier kopieën. Scope = core-munten + de top-N uit de ranking-CSV.
N staat standaard op 1 (zoals de bots tot nu toe deden) en is voor ALLE services tegelijk
aan te passen met de env-variabele SCAN_TOP_N.
"""
import os

import pandas as pd

CORE_SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
RANKING_FILE = 'kucoin_best_coins_ranking.csv'

_LAST_SCOPE = {'key': None}


def scan_top_n():
    try:
        return max(0, int(os.environ.get('SCAN_TOP_N', 1)))
    except ValueError:
        return 1


def get_combined_trading_assets(top_n=None):
    n = scan_top_n() if top_n is None else max(0, int(top_n))
    dynamic = []
    if n > 0 and os.path.exists(RANKING_FILE):
        try:
            df = pd.read_csv(RANKING_FILE)
            if 'Symbol' in df.columns:
                for s in df['Symbol'].head(n).tolist():
                    if s not in dynamic and ':USDT' in s:
                        dynamic.append(s)
        except Exception as e:
            print(f"⚠️ Fout bij lezen van ranking CSV: {e}")

    combined = list(dict.fromkeys(CORE_SYMBOLS + dynamic))
    key = tuple(combined)
    if _LAST_SCOPE['key'] != key:  # alleen loggen bij wijziging (de traders roepen dit elke minuut aan)
        _LAST_SCOPE['key'] = key
        print(f"🎯 Actieve trading scope (core + top {n}, {len(combined)} symbols): {combined}")
    return combined
