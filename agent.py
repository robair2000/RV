import os
import json
import time
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pandas as pd
import ccxt

from github_utils import github_put, github_pull
from market_utils import fetch_ohlcv_paginated
from asset_utils import get_combined_trading_assets

# --- Configuraties & Constanten ---
# Alles is via environment variables aan te passen; de defaults = het oorspronkelijke gedrag.

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        print(f"⚠️ {name}={os.environ.get(name)!r} is geen getal, default {default} gebruikt.")
        return float(default)


def _env_int(name, default):
    return int(round(_env_float(name, default)))


# Data
META_HISTORY_4H = _env_int('META_HISTORY_4H', 2000)        # aantal 4h candles dat wordt opgehaald

# Bollinger Bands
BB_PERIOD = _env_int('BB_PERIOD', 20)                       # periode van SMA en standaarddeviatie
BB_STD = _env_float('BB_STD', 2.0)                          # bandbreedte in standaarddeviaties
BB_PCTL_WINDOW = _env_int('BB_PCTL_WINDOW', 100)            # lookback voor de squeeze-percentiel

# Signaalfilters
SQUEEZE_PCTL = _env_float('SQUEEZE_PCTL', 0.35)             # bandbreedte moet in de laagste X van het window zitten
VOL_MA_PERIOD = _env_int('VOL_MA_PERIOD', 20)               # periode van het gemiddelde volume
MIN_VOL_RATIO = _env_float('MIN_VOL_RATIO', 1.3)            # volume >= X keer het gemiddelde
PCTB_LONG = _env_float('PCTB_LONG', 0.90)                   # %B >= X -> long
PCTB_SHORT = _env_float('PCTB_SHORT', 0.10)                 # %B <= X -> short

# Volatiliteit (bepaalt SL/TP-afstand in de trader)
ATR_PERIOD = _env_int('ATR_PERIOD', 14)

# Score & leverage
SIGNAL_SCORE = _env_float('SIGNAL_SCORE', 85.0)             # score die een signaal krijgt
NO_SIGNAL_SCORE = 50.0
MIN_BOSS_SCORE = _env_float('MIN_BOSS_SCORE', 75.0)         # alleen informatief in het state-bestand
BASE_LEVERAGE = _env_int('BASE_LEVERAGE', 3)                # leverage vóór de Markov risk multiplier
MIN_LEVERAGE = _env_int('MIN_LEVERAGE', 1)
MAX_LEVERAGE = _env_int('MAX_LEVERAGE', 5)

# Timing
SCAN_DELAY_SECONDS = _env_int('SCAN_DELAY_SECONDS', 60)     # wachttijd na 4h candle-close voor de scan

# Minimum aantal candles nodig om alle indicatoren te kunnen berekenen
MIN_BARS_REQUIRED = max(BB_PERIOD, VOL_MA_PERIOD, ATR_PERIOD + 1) + BB_PCTL_WINDOW + 5

exchange = ccxt.kucoinfutures({'enableRateLimit': True})


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<html><body><h2>🚀 Squeeze Agent is online</h2></body></html>".encode("utf-8"))

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()


def push_file_to_github(filename):
    if not os.environ.get('GITHUB_TOKEN') or not os.path.exists(filename):
        return
    with open(filename, 'rb') as f:
        content = f.read()
    if github_put(filename, content, agent='squeeze-agent'):
        print(f"☁️ Bestand {filename} gepusht naar GitHub.")


def load_markov_state(asset):
    path = f'markov_4h_state_{asset.lower()}.json'
    # Wordt door een andere service geschreven -> eerst verse kopie van GitHub halen
    if os.environ.get('GITHUB_TOKEN'):
        github_pull(path, min_interval=600, agent='squeeze-agent')
    if not os.path.exists(path):
        return {'green': 50.0, 'red': 50.0}, 1.0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            probs = data.get('probabilities_next_4h', {'green': 50.0, 'red': 50.0})
            risk_mult = float(data.get('risk_multiplier', 1.0))
            return probs, risk_mult
    except Exception:
        return {'green': 50.0, 'red': 50.0}, 1.0


def compute_indicators(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    df['sma'] = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std()

    bb_upper = df['sma'] + (std * BB_STD)
    bb_lower = df['sma'] - (std * BB_STD)

    bb_width = (bb_upper - bb_lower) / df['sma']
    df['bb_width_pctl'] = bb_width.rolling(BB_PCTL_WINDOW).rank(pct=True)
    df['percent_b'] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-8)

    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(VOL_MA_PERIOD).mean() + 1e-8)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()
    df['atr_pct'] = (df['atr'] / close) * 100.0

    return df.dropna().reset_index(drop=True)

def run_agent_scan():
    try:
        subprocess.run(["python", "getcoins.py"], check=True)
        push_file_to_github('kucoin_best_coins_ranking.csv')
    except Exception as e:
        print(f"⚠️ getcoins.py overgeslagen: {e}")
        time.sleep(3.0)

    symbols = get_combined_trading_assets()
    written = []

    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        try:
            print(f"Bezig met ophalen en scannen van {symbol}...")
            
            # Robuuste fetch met retry-mechanisme en ruime pauzes tegen vastlopen/rate-limits
            d4 = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    d4 = fetch_ohlcv_paginated(exchange, symbol, '4h', total_bars=META_HISTORY_4H)
                    if d4 is not None and len(d4) >= 100:
                        break
                except Exception as fetch_err:
                    print(f"⚠️ Poging {attempt + 1}/{max_retries} mislukt voor {symbol}: {fetch_err}")
                
                if attempt < max_retries - 1:
                    time.sleep(6.0)

            if d4 is None or len(d4) < 100:
                print(f"❌ Onvoldoende data na {max_retries} pogingen voor {symbol}. Overslaan.")
                time.sleep(4.0)
                continue
            
            # NB: fetch_ohlcv_paginated verwijdert de lopende candle al. Hier NIET nog eens iloc[:-1]
            # doen: dan valt de net gesloten candle weg, is last_candle_ts altijd ~8h oud en wijst de
            # trader (MAX_CANDLE_AGE 5.5h) elk signaal af.
            if d4.empty:
                continue

            df = compute_indicators(d4)
            if df.empty:
                continue

            markov_probs, risk_mult = load_markov_state(asset)

            latest_signal = 'none'
            score = 50.0
            leverage = 2

            last_row = df.iloc[-1]
            last_candle_ts = int(last_row['timestamp'])

            is_squeeze = last_row['bb_width_pctl'] <= SQUEEZE_PCTL
            has_volume = last_row['vol_ratio'] >= MIN_VOL_RATIO

            if is_squeeze and has_volume:
                pct_b = last_row['percent_b']
                
                if pct_b >= 0.90:
                    latest_signal = 'long'
                    score = 85.0
                elif pct_b <= 0.10:
                    latest_signal = 'short'
                    score = 85.0
                else:
                    latest_signal = 'none'
                    score = 50.0

                if latest_signal != 'none':
                    base_lev = 3 if score >= 85 else 2
                    leverage = int(np.clip(round(base_lev * risk_mult), 1, 5))

            atr_pct_val = float(last_row['atr_pct']) if 'atr_pct' in last_row else 2.0

            state = {
                'updated_at': time.time(),
                'last_candle_ts': last_candle_ts,
                'pattern_signal': latest_signal,
                'pattern_score': score,
                'recommended_leverage': leverage,
                'risk_multiplier': risk_mult,
                'atr_pct': atr_pct_val,
                'min_score_required': MIN_BOSS_SCORE
            }

            filename = f'pattern_state_{asset.lower()}_4h.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            push_file_to_github(filename)
            written.append(filename)
            print(f"🚀 Squeeze Agent geüpdatet voor {asset}: signal={latest_signal}, lev={leverage} (risk_mult={risk_mult})")

            # Zeer ruime pauze tussen munten om rate-limits (429) te voorkomen
            time.sleep(8.0)

        except Exception as e:
            print(f"❌ Fout bij agent scan voor {asset}: {e}")
            time.sleep(6.0)  # Extra adempauze bij fouten

    print(f"✅ Agent scan voltooid. {len(written)} bestanden bijgewerkt.")


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Squeeze Agent Web Service gestart met ingebouwde candle-sync.")

    CANDLE_SECONDS = 4 * 3600
    SCAN_DELAY_SECONDS = 60  # Wacht 1 minuut na candle sluiting

    def seconds_until_next_scan():
        now = time.time()
        next_close = (int(now // CANDLE_SECONDS) + 1) * CANDLE_SECONDS
        return max(5, next_close + SCAN_DELAY_SECONDS - now)

    while True:
        try:
            print("🔍 Start gesynchroniseerde Squeeze Agent scan na candle close...")
            run_agent_scan()
        except Exception as e:
            print(f"⚠️ Fout in agent hoofdloop: {e}")
        
        sleep_seconds = seconds_until_next_scan()
        print(f"💤 Squeeze Agent wacht tot de volgende 4h candle cyclus (over {int(sleep_seconds // 60)} minuten)...")
        time.sleep(sleep_seconds)
