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
META_HISTORY_4H = _env_int('META_HISTORY_4H', 200)         # aantal 4h candles (200 = 1 API-call; min ~125)

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

# Score (graded): een signaal krijgt SCORE_FLOOR..SCORE_CEIL afhankelijk van de kwaliteit.
# De trader neemt alleen signalen met score >= zijn eigen MIN_BOSS_SCORE (default 75).
SCORE_FLOOR = _env_float('SCORE_FLOOR', 75.0)               # score van een signaal dat net door de filters komt
SCORE_CEIL = _env_float('SCORE_CEIL', 100.0)                # score van een perfect signaal
NO_SIGNAL_SCORE = 50.0
W_SQUEEZE = _env_float('W_SQUEEZE', 1.0)                    # gewicht: hoe diep de squeeze is
W_VOLUME = _env_float('W_VOLUME', 1.0)                      # gewicht: hoe sterk het volume is
W_BREAKOUT = _env_float('W_BREAKOUT', 1.0)                  # gewicht: hoe ver de close voorbij de %B-drempel ligt
W_MARKOV = _env_float('W_MARKOV', 1.0)                      # gewicht: of Markov de richting steunt
VOL_FULL = _env_float('VOL_FULL', 3.0)                      # volume-ratio die de volle volume-score geeft
BREAKOUT_FULL = _env_float('BREAKOUT_FULL', 0.30)           # %B-afstand voorbij de drempel voor de volle breakout-score
MIN_BOSS_SCORE = _env_float('MIN_BOSS_SCORE', 75.0)         # alleen informatief in het state-bestand

# Markov
MARKOV_FULL_PCT = _env_float('MARKOV_FULL_PCT', 10.0)       # kans boven 50% die als 'volle steun' telt (10 -> 60%)
MARKOV_LEV_RANGE = _env_float('MARKOV_LEV_RANGE', 0.2)      # leverage x(1 +/- dit) bij volle steun/tegenspraak
MARKOV_MAX_AGE_HOURS = _env_float('MARKOV_MAX_AGE_HOURS', 6.0)  # oudere Markov state -> neutraal

# Leverage
BASE_LEVERAGE = _env_int('BASE_LEVERAGE', 3)                # leverage bij neutrale Markov
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


NEUTRAL_PROBS = {'green': 50.0, 'red': 50.0}


def load_markov_probs(asset):
    """Leest de kansen van de Markov worker (markov_state_<asset>_4h.json).

    Geeft neutraal 50/50 terug als het bestand ontbreekt, onleesbaar is of ouder dan
    MARKOV_MAX_AGE_HOURS. De risk_multiplier uit dat bestand wordt bewust NIET gebruikt:
    die is richting-blind en geeft 0.8 bij 50/50. De agent rekent zelf een richting-bewuste
    multiplier uit (zie markov_agreement / markov_leverage_mult).
    """
    path = f'markov_state_{asset.lower()}_4h.json'
    if os.environ.get('GITHUB_TOKEN'):
        github_pull(path, min_interval=600, agent='squeeze-agent')
    if not os.path.exists(path):
        print(f"   ℹ️ {asset}: geen {path}, Markov neutraal (50/50).")
        return dict(NEUTRAL_PROBS)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        age_h = (time.time() - float(data.get('updated_at', 0) or 0)) / 3600
        if age_h > MARKOV_MAX_AGE_HOURS:
            print(f"   ℹ️ {asset}: Markov state {age_h:.1f}h oud (> {MARKOV_MAX_AGE_HOURS}h), neutraal gebruikt.")
            return dict(NEUTRAL_PROBS)
        probs = data.get('probabilities_next_4h') or NEUTRAL_PROBS
        return {'green': float(probs.get('green', 50.0)), 'red': float(probs.get('red', 50.0))}
    except Exception as e:
        print(f"   ⚠️ {asset}: Markov state onleesbaar ({e}), neutraal gebruikt.")
        return dict(NEUTRAL_PROBS)


def markov_agreement(probs, signal):
    """-1..1: hoe sterk Markov de signaalrichting steunt (+1) of tegenspreekt (-1). 0 = 50/50."""
    p_agree = probs['green'] if signal == 'long' else probs['red']
    return float(np.clip((p_agree - 50.0) / MARKOV_FULL_PCT, -1.0, 1.0))


def markov_leverage_mult(agreement):
    """1.0 bij neutraal, 1 + MARKOV_LEV_RANGE bij volle steun, 1 - MARKOV_LEV_RANGE bij volle tegenspraak."""
    return round(1.0 + agreement * MARKOV_LEV_RANGE, 3)


def grade_signal(row, signal, probs):
    """Score tussen SCORE_FLOOR en SCORE_CEIL voor een signaal dat de harde filters al passeerde.

    Vier componenten, elk 0..1:
      squeeze  : 0 = band-breedte precies op SQUEEZE_PCTL, 1 = smalste ooit in het window
      volume   : 0 = precies MIN_VOL_RATIO, 1 = VOL_FULL of meer
      breakout : 0 = %B precies op de drempel, 1 = BREAKOUT_FULL voorbij de drempel
      markov   : 0 = volle tegenspraak, 0.5 = neutraal, 1 = volle steun
    """
    squeeze = float(np.clip((SQUEEZE_PCTL - row['bb_width_pctl']) / max(SQUEEZE_PCTL, 1e-9), 0, 1))
    volume = float(np.clip((row['vol_ratio'] - MIN_VOL_RATIO) / max(VOL_FULL - MIN_VOL_RATIO, 1e-9), 0, 1))
    if signal == 'long':
        excess = row['percent_b'] - PCTB_LONG
    else:
        excess = PCTB_SHORT - row['percent_b']
    breakout = float(np.clip(excess / max(BREAKOUT_FULL, 1e-9), 0, 1))
    agreement = markov_agreement(probs, signal)
    markov = (agreement + 1.0) / 2.0

    parts = {'squeeze': squeeze, 'volume': volume, 'breakout': breakout, 'markov': markov}
    weights = {'squeeze': W_SQUEEZE, 'volume': W_VOLUME, 'breakout': W_BREAKOUT, 'markov': W_MARKOV}
    total_w = sum(weights.values())
    quality = sum(parts[k] * weights[k] for k in parts) / total_w if total_w > 0 else 0.0
    score = round(SCORE_FLOOR + (SCORE_CEIL - SCORE_FLOOR) * quality, 1)
    return score, {k: round(v, 2) for k, v in parts.items()}, agreement


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
                    if d4 is not None and len(d4) >= MIN_BARS_REQUIRED:
                        break
                except Exception as fetch_err:
                    print(f"⚠️ Poging {attempt + 1}/{max_retries} mislukt voor {symbol}: {fetch_err}")
                
                if attempt < max_retries - 1:
                    time.sleep(6.0)

            if d4 is None or len(d4) < MIN_BARS_REQUIRED:
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

            markov_probs = load_markov_probs(asset)

            latest_signal = 'none'
            score = NO_SIGNAL_SCORE
            leverage = MIN_LEVERAGE
            risk_mult = 1.0
            breakdown = {}

            last_row = df.iloc[-1]
            last_candle_ts = int(last_row['timestamp'])

            is_squeeze = last_row['bb_width_pctl'] <= SQUEEZE_PCTL
            has_volume = last_row['vol_ratio'] >= MIN_VOL_RATIO

            if is_squeeze and has_volume:
                pct_b = last_row['percent_b']

                if pct_b >= PCTB_LONG:
                    latest_signal = 'long'
                elif pct_b <= PCTB_SHORT:
                    latest_signal = 'short'

                if latest_signal != 'none':
                    score, breakdown, agreement = grade_signal(last_row, latest_signal, markov_probs)
                    risk_mult = markov_leverage_mult(agreement)
                    leverage = int(np.clip(round(BASE_LEVERAGE * risk_mult), MIN_LEVERAGE, MAX_LEVERAGE))
                    print(f"   🎯 {asset}: {latest_signal.upper()} score={score} {breakdown} "
                          f"| Markov groen/rood={markov_probs['green']:.1f}/{markov_probs['red']:.1f} "
                          f"-> lev {BASE_LEVERAGE}x{risk_mult} = {leverage}")

            # Laat per munt zien hoe dicht hij bij een signaal zat (handig bij het finetunen)
            print(f"   📐 {asset}: squeeze-pctl={last_row['bb_width_pctl']:.2f} (<= {SQUEEZE_PCTL}), "
                  f"vol={last_row['vol_ratio']:.2f}x (>= {MIN_VOL_RATIO}), "
                  f"%B={last_row['percent_b']:.2f} (>= {PCTB_LONG} long / <= {PCTB_SHORT} short)")

            atr_pct_val = float(last_row['atr_pct']) if 'atr_pct' in last_row else 2.0

            state = {
                'updated_at': time.time(),
                'last_candle_ts': last_candle_ts,
                'pattern_signal': latest_signal,
                'pattern_score': score,
                'recommended_leverage': leverage,
                'risk_multiplier': risk_mult,
                'markov_probs': markov_probs,
                'score_breakdown': breakdown,
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
    print(f"⚙️ Config: BB {BB_PERIOD}/{BB_STD}, pctl-window {BB_PCTL_WINDOW}, squeeze <= {SQUEEZE_PCTL}, "
          f"vol >= {MIN_VOL_RATIO}x (MA {VOL_MA_PERIOD}), %B {PCTB_SHORT}/{PCTB_LONG}, ATR {ATR_PERIOD}, "
          f"score {SCORE_FLOOR}-{SCORE_CEIL} (w sq/vol/bo/mk {W_SQUEEZE}/{W_VOLUME}/{W_BREAKOUT}/{W_MARKOV}), leverage {BASE_LEVERAGE} [{MIN_LEVERAGE}-{MAX_LEVERAGE}], "
          f"history {META_HISTORY_4H}, scan-delay {SCAN_DELAY_SECONDS}s")

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
