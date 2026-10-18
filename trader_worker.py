import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt


# --- RENDER WEB SERVER (VOOR POORT-BINDING & HEALTH CHECKS) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()


# --- CONFIGURATIE FUTURES LIVE TRADING ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h']
TRADE_CHECK_INTERVAL_SECONDS = 900    # Elke 15 minuten nieuwe trades zoeken
POSITION_CHECK_INTERVAL_SECONDS = 60  # Elke minuut open posities controleren

TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 20.0))   # margin per trade
DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')
EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'kucoin')

# Risico & portfolio grenzen
MAX_CONCURRENT_POSITIONS = 2  # Bescherming tegen over-exposure

# Basis & risico drempels
MIN_DIRECTION_PROBABILITY = float(os.environ.get('MIN_DIRECTION_PROBABILITY', 65.0))
MIN_PREDICTED_RETURN_PCT = float(os.environ.get('MIN_PREDICTED_RETURN_PCT', 0.15))
MIN_HOLDOUT_ACCURACY = float(os.environ.get('MIN_HOLDOUT_ACCURACY', 0.55))
MIN_RRR = 1.40
MIN_ATR_PCT = 0.25
MAX_ATR_PCT = 5.5
MAX_FUNDING_RATE = 0.0005
COOLDOWN_SECONDS = 3600

TP_ATR_mult_BASE = 2.5
MAX_NOTIONAL_OVERSHOOT = 1.5
MIN_SCORE = 2

MAX_STATE_AGE_SECONDS = {'15m': 40 * 60, '1h': 2 * 3600 + 30 * 60}
MIN_ORDERBOOK_IMBALANCE = 0.55

PAUSE_BETWEEN_ASSETS = float(os.environ.get('PAUSE_BETWEEN_ASSETS', 4.0))
PAUSE_AFTER_CALL = float(os.environ.get('PAUSE_AFTER_CALL', 1.0))
PAUSE_RETRY = 1.0
ENABLE_BTC_CORRELATION = True

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

active_trades = {}
symbol_cooldowns = {}
missing_counts = {}
MISSING_CHECKS_REQUIRED = 2

exchange = getattr(ccxt, EXCHANGE_ID)({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})


# =====================================================================
# HULPFUNCTIE: RETRY LOGICA MET BACKOFF
# =====================================================================
def execute_with_retry(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise e
            print(f"⚠️ Exchange actie mislukt ({e}), poging {attempt + 1} van {retries} over {delay}s...")
            time.sleep(delay)
            delay *= 2


# =====================================================================
# STATE LADEN VANAF GITHUB
# =====================================================================
def pull_latest_states_from_github():
    token = os.environ.get('GITHUB_TOKEN', '')
    failed = 0
    files_to_pull = []
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            files_to_pull.append(f'model_state_{asset.lower()}_{tf}.json')
            files_to_pull.append(f'pattern_state_{asset.lower()}_{tf}.json')

    for filename in files_to_pull:
        url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
        headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'trader-worker'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
            tmp = filename + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(content)
            os.replace(tmp, filename)
        except Exception:
            failed += 1
    if failed < len(files_to_pull):
        print("🔄 Quant states & Final Boss states opgehaald van GitHub.")


def load_state(asset, tf):
    path = f'model_state_{asset.lower()}_{tf}.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            st = json.load(f)
    except Exception:
        return None
    if st.get('data_status') != 'ok' or st.get('signal') not in ('long', 'short', 'none'):
        return None
    age = time.time() - float(st.get('updated_at', 0) or 0)
    if age > MAX_STATE_AGE_SECONDS.get(tf, 3600):
        return None
    return st


def load_pattern_state(asset, tf='15m'):
    path = f'pattern_state_{asset.lower()}_{tf}.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


# =====================================================================
# ORDERS & POSITIES
# =====================================================================
def close_position(symbol, side, contracts):
    close_side = 'sell' if side == 'long' else 'buy'
    if DRY_RUN:
        print(f"[DRY_RUN] Zou {symbol} sluiten ({close_side} {contracts})")
        return True
    try:
        order = execute_with_retry(lambda: exchange.create_order(
            symbol, 'market', close_side, contracts, None,
            {'reduceOnly': True, 'marginMode': DESIRED_MARGIN_MODE}
        ))
        print(f"✅ Positie {symbol} gesloten: {order.get('id') if isinstance(order, dict) else order}")
        return True
    except Exception as e:
        print(f"⛔ Fout bij sluiten {side} {symbol}: {repr(e)}")
        return False


def pnl_fraction(side, entry, price):
    if not entry:
        return 0.0
    move = (price - entry) / entry
    return move if side == 'long' else -move


def manage_open_positions_and_trailing_stops():
    try:
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception as e:
        print(f"Fout bij ophalen open posities: {e}")
        return

    open_symbols = set()
    for pos in positions:
        contracts = abs(float(pos.get('contracts') or 0))
        if contracts == 0:
            continue
        symbol = pos.get('symbol')
        side = pos.get('side')
        if side not in ('long', 'short'):
            continue
        open_symbols.add(symbol)
        entry = float(pos.get('entryPrice') or 0)

        try:
            price = float(exchange.fetch_ticker(symbol)['last'])
        except Exception:
            continue

        trade = active_trades.get(symbol)
        if trade is None:
            base = symbol.split('/')[0]
            st = load_state(base, '15m')
            pst = load_pattern_state(base, '15m')
            trail_mult = float(pst.get('recommended_trailing_mult', 1.5)) if pst else 1.5

            if st and st.get('atr_pct'):
                atr = entry * float(st['atr_pct']) / 100.0
            else:
                atr = entry * 0.015
            sgn = 1 if side == 'long' else -1
            tp_mult = max(TP_ATR_mult_BASE, trail_mult * (MIN_RRR + 0.1))
            trade = active_trades[symbol] = {
                'side': side, 'entry': entry, 'peak': price, 'atr': atr,
                'sl': entry - sgn * trail_mult * atr,
                'tp': entry + sgn * tp_mult * atr,
                'trailing_mult': trail_mult,
            }

        atr = trade['atr']
        trail_mult = trade.get('trailing_mult', 1.5)
        current_pnl_pct = pnl_fraction(side, entry, price)
        
        # Zuivere logica per kant zonder scope-lekkage
        if side == 'long':
            if price > trade['peak']:
                trade['peak'] = price
                if current_pnl_pct >= 0.015:
                    new_sl = price - trail_mult * atr
                    if new_sl > trade['sl']:
                        trade['sl'] = new_sl
                        print(f"📈 Trailing stop {symbol} geactiveerd naar ${trade['sl']:.4f} (Boss mult: {trail_mult}x)")
            stop_hit = price <= trade['sl']
            tp_hit = price >= trade['tp']
        else:
            if price < trade['peak']:
                trade['peak'] = price
                if current_pnl_pct >= 0.015:
                    new_sl = price + trail_mult * atr
                    if new_sl < trade['sl']:
                        trade['sl'] = new_sl
                        print(f"📉 Trailing stop {symbol} geactiveerd naar ${trade['sl']:.4f} (Boss mult: {trail_mult}x)")
            stop_hit = price >= trade['sl']
            tp_hit = price <= trade['tp']
            
        if stop_hit or tp_hit:
            reason = 'stop' if stop_hit else 'take-profit'
            print(f"🛑 {reason} geraakt voor {side} {symbol} op ${price}. Positie sluiten...")
            if close_position(symbol, side, contracts):
                if pnl_fraction(side, entry, price) < 0:
                    symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)

    for symbol in list(missing_counts.keys()):
        if symbol in open_symbols or symbol not in active_trades:
            missing_counts.pop(symbol, None)
    for symbol in list(active_trades.keys()):
        if symbol in open_symbols:
            continue
        missing_counts[symbol] = missing_counts.get(symbol, 0) + 1
        if missing_counts[symbol] < MISSING_CHECKS_REQUIRED:
            continue
        missing_counts.pop(symbol, None)
        active_trades.pop(symbol)


def ensure_margin_and_leverage(symbol, target_leverage):
    if DRY_RUN:
        print(f"[DRY_RUN] Zou leverage instellen op {target_leverage}x voor {symbol}")
        return True
    has = getattr(exchange, 'has', {}) or {}
    try:
        if has.get('setMarginMode'):
            for attempt in range(2):
                try:
                    exchange.set_margin_mode(DESIRED_MARGIN_MODE, symbol)
                    break
                except Exception:
                    if attempt == 1:
                        raise
                    time.sleep(PAUSE_RETRY)
        for attempt in range(2):
            try:
                try:
                    exchange.set_leverage(target_leverage, symbol)
                except Exception:
                    exchange.set_leverage(target_leverage, symbol, {'marginMode': DESIRED_MARGIN_MODE})
                print(f"⚙️ Leverage ingesteld op {target_leverage}x voor {symbol}")
                return True
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(PAUSE_RETRY)
    except Exception as e:
        print(f"⛔ Margin/leverage setup mislukt voor {symbol}: {repr(e)}")
    return False


def compute_contract_amount(symbol, price, leverage):
    target = TRADE_AMOUNT_USDT * leverage
    try:
        contract_size = float(exchange.market(symbol).get('contractSize') or 1.0)
    except Exception:
        return None
    per_contract = price * contract_size
    if per_contract <= 0:
        return None
    amount = math.floor(target / per_contract)
    if amount < 1:
        if per_contract <= target * MAX_NOTIONAL_OVERSHOOT:
            amount = 1
        else:
            return None
    return float(amount)


# =====================================================================
# SCAN & EVALUATIE (MET PORTFOLIO EXPOSURE & FINAL BOSS GATEKEEPER)
# =====================================================================
def resolve_symbol(asset):
    primary = f"{asset}/USDT:USDT"
    if primary in exchange.markets:
        return primary
    fallback = f"{asset}/USDT"
    m = exchange.markets.get(fallback)
    if m and (m.get('swap') or m.get('contract')):
        return fallback
    return None


def execute_live_trades_scan():
    pull_latest_states_from_github()
    print("--- [Futures Live Trader Worker] Start scan ---")

    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin Futures: {e}")
        return

    try:
        positions = execute_with_retry(lambda: exchange.fetch_positions())
        active_symbols = {p.get('symbol') for p in positions if abs(float(p.get('contracts') or 0)) > 0}
    except Exception as e:
        print(f"⛔ Kan open posities niet ophalen ({e}); scan afgebroken.")
        return

    if len(active_symbols) >= MAX_CONCURRENT_POSITIONS:
        print(f"🛡️ Maximaal aantal gelijktijdige posities ({MAX_CONCURRENT_POSITIONS}) bereikt.")
        return

    for asset in ASSETS:
        symbol = resolve_symbol(asset)
        if symbol is None:
            continue
        if symbol in active_symbols:
            continue
        if symbol in symbol_cooldowns:
            elapsed = time.time() - symbol_cooldowns[symbol]
            if elapsed < COOLDOWN_SECONDS:
                continue
            del symbol_cooldowns[symbol]

        try:
            evaluate_and_trade(asset, symbol)
        except Exception as e:
            print(f"Fout bij verwerken van {asset}: {e}")
        time.sleep(PAUSE_BETWEEN_ASSETS)


def evaluate_and_trade(asset, symbol):
    state_15m = load_state(asset, '15m')
    if state_15m is None:
        return

    signal = state_15m['signal']
    if signal == 'none':
        return
    is_long = signal == 'long'
    prob = float(state_15m.get('signal_probability', 50.0))
    pred_move_pct = abs(float(state_15m.get('predicted_return_pct', 0.0)))
    holdout_acc = state_15m.get('holdout_direction_accuracy')
    atr_pct = float(state_15m.get('atr_pct') or 0.0)

    price = float(exchange.fetch_ticker(symbol)['last'])
    if atr_pct <= 0:
        return
    atr = price * atr_pct / 100.0

    # =====================================================================
    # 👑 FINAL BOSS GATEKEEPER & ADAPTIEVE DREMPEL CONTROLE
    # =====================================================================
    pattern_st = load_pattern_state(asset, '15m')
    if pattern_st:
        boss_signal = pattern_st.get('pattern_signal', 'none')
        boss_score = pattern_st.get('pattern_score', 50.0)
        boss_leverage = int(pattern_st.get('recommended_leverage', 10))
        boss_trailing_mult = float(pattern_st.get('recommended_trailing_mult', 1.5))
        
        dynamic_min_score = float(pattern_st.get('min_score_required', 75.0))
        strictness_mode = pattern_st.get('strictness_mode', 'strict')
        is_squeeze = pattern_st.get('bollinger_squeeze', False)
        
        print(f"👑 [Final Boss] {asset}: Signaal={boss_signal.upper()} | Score={boss_score} (Vereist: {dynamic_min_score}, Modus: {strictness_mode}) | Lev={boss_leverage}x | Squeeze={is_squeeze}")
        
        if boss_signal != signal:
            print(f"⛔ [FINAL BOSS VETO] Richting komt niet overeen met bot ('{boss_signal}' vs '{signal}').")
            return
        if boss_score < dynamic_min_score:
            print(f"⛔ [FINAL BOSS VETO] Score ({boss_score}) is te laag volgens de agent zijn huidige regime ({dynamic_min_score}).")
            return
        print(f"✅ [FINAL BOSS GOEDKEURING] Groen licht onder '{strictness_mode}' regime!")
    else:
        print(f"⚠️ [Final Boss] Geen state gevonden voor {asset}; overgeslagen.")
        return

    sgn = 1 if is_long else -1
    stop_loss_val = price - sgn * boss_trailing_mult * atr
    
    # Dynamische TP op basis van trailing_mult zodat squeeze-trades (mult=2.0) altijd aan RRR >= MIN_RRR voldoen
    effective_tp_mult = max(TP_ATR_mult_BASE, boss_trailing_mult * (MIN_RRR + 0.1))
    take_profit_val = price + sgn * effective_tp_mult * atr

    # ========== HARDE & ZACHTE FILTERS (MET DEBUG LOGS) ==========
    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
        print(f"⛔ [FILTER VETO] ATR% ({atr_pct:.2f}%) buiten bereik [{MIN_ATR_PCT}, {MAX_ATR_PCT}].")
        return
        
    rrr = abs(take_profit_val - price) / abs(price - stop_loss_val)
    if rrr < MIN_RRR:
        print(f"⛔ [FILTER VETO] RRR ({rrr:.2f}) is lager dan vereist ({MIN_RRR}).")
        return
        
    if holdout_acc is None or holdout_acc < MIN_HOLDOUT_ACCURACY:
        print(f"⛔ [FILTER VETO] Holdout accuracy ({holdout_acc}) te laag (Min: {MIN_HOLDOUT_ACCURACY}).")
        return

    score, max_score = 0, 4
    state_1h = load_state(asset, '1h')
    if state_1h is not None and state_1h['signal'] == signal:
        score += 1

    if asset == 'BTC':
        score += 1
    elif ENABLE_BTC_CORRELATION:
        btc = load_state('BTC', '15m')
        opposite = 'short' if is_long else 'long
