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
LEVERAGE = int(os.environ.get('LEVERAGE', 10))
DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
# Zet DRY_RUN=true om alles te loggen zonder echte orders te plaatsen.
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')
EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'kucoin')   # zoals in je werkende opzet; alternatief: 'kucoinfutures'

# Basis & risico drempels
MIN_DIRECTION_PROBABILITY = float(os.environ.get('MIN_DIRECTION_PROBABILITY', 65.0))
# Minimale door het model voorspelde beweging (%). Moet boven de round-trip kosten liggen
# (~0.12% taker-fees + slippage), anders is een correcte voorspelling nog verliesgevend.
MIN_PREDICTED_RETURN_PCT = float(os.environ.get('MIN_PREDICTED_RETURN_PCT', 0.15))
# Minimale out-of-sample richtingsnauwkeurigheid van het model (holdout) om te mogen handelen.
MIN_HOLDOUT_ACCURACY = float(os.environ.get('MIN_HOLDOUT_ACCURACY', 0.55))
MIN_RRR = 1.55
MIN_ATR_PCT = 0.25
MAX_ATR_PCT = 5.5
MAX_FUNDING_RATE = 0.0005
COOLDOWN_SECONDS = 3600
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MAX_NOTIONAL_OVERSHOOT = 1.5          # positie mag max 1.5x de gewenste notional zijn (bij grote contracten)
MIN_SCORE = 2                         # minimaal aantal zachte punten (van 4)

# Een state ouder dan dit wordt niet gebruikt (de worker is dan waarschijnlijk gestopt)
MAX_STATE_AGE_SECONDS = {'15m': 40 * 60, '1h': 2 * 3600 + 30 * 60}

MIN_ORDERBOOK_IMBALANCE = 0.55

# Pauzes tegen 429 rate limits (overgenomen uit je 'safe'-branch)
PAUSE_BETWEEN_ASSETS = float(os.environ.get('PAUSE_BETWEEN_ASSETS', 4.0))
PAUSE_AFTER_CALL = float(os.environ.get('PAUSE_AFTER_CALL', 1.0))
PAUSE_RETRY = 1.0
ENABLE_BTC_CORRELATION = True

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

# Geheugen voor tracking
active_trades = {}      # {symbol: {side, entry, peak, sl, tp, atr}}
symbol_cooldowns = {}   # {symbol: timestamp_laatste_verlies}

exchange = getattr(ccxt, EXCHANGE_ID)({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})


# =====================================================================
# STATE LADEN
# =====================================================================
def pull_latest_states_from_github():
    """Haalt de state-bestanden op via de GitHub API (geen CDN-cache van raw.githubusercontent)."""
    token = os.environ.get('GITHUB_TOKEN', '')
    failed = 0
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            filename = f'model_state_{asset.lower()}_{tf}.json'
            url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
            headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'trader-worker'}
            if token:
                headers['Authorization'] = f'Bearer {token}'
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read()
                json.loads(content.decode('utf-8'))      # alleen geldige JSON wegschrijven
                tmp = filename + '.tmp'
                with open(tmp, 'wb') as f:
                    f.write(content)
                os.replace(tmp, filename)
            except Exception as e:
                failed += 1
                print(f"⚠️ Kon {filename} niet ophalen: {e}")
    if failed:
        print(f"⚠️ {failed} state-bestand(en) niet ververst; verouderde states worden door de leeftijdscheck geweigerd.")
    else:
        print("🔄 Verse state-bestanden opgehaald van GitHub.")


def load_state(asset, tf):
    """Geeft de state terug als hij bestaat, geldig is, vers genoeg is en een voorspelling bevat; anders None."""
    path = f'model_state_{asset.lower()}_{tf}.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            st = json.load(f)
    except Exception as e:
        print(f"⚠️ State {path} onleesbaar: {e}")
        return None
    if st.get('data_status') != 'ok' or st.get('signal') not in ('long', 'short', 'none'):
        return None   # oud formaat of onvoldoende data
    age = time.time() - float(st.get('updated_at', 0) or 0)
    if age > MAX_STATE_AGE_SECONDS.get(tf, 3600):
        print(f"⚠️ State {path} is {age / 60:.0f} min oud (max {MAX_STATE_AGE_SECONDS.get(tf, 3600) / 60:.0f}); genegeerd.")
        return None
    return st


# =====================================================================
# ORDERS
# =====================================================================
def close_position(symbol, side, contracts):
    """Sluit een positie met een reduce-only market order. Retourneert True bij succes."""
    close_side = 'sell' if side == 'long' else 'buy'
    if DRY_RUN:
        print(f"[DRY_RUN] Zou {symbol} sluiten ({close_side} {contracts})")
        return True
    try:
        order = exchange.create_order(symbol, 'market', close_side, contracts, None,
                                      {'reduceOnly': True, 'marginMode': DESIRED_MARGIN_MODE})
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
    """
    Beheert open posities (elke minuut): trailing stop, harde stop-loss en take-profit.
    De stop is een echte stop: hij sluit de positie zodra hij geraakt wordt, ongeacht wat het model zegt.
    """
    try:
        positions = exchange.fetch_positions()
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
            # Na een herstart: stops opnieuw opbouwen vanuit de ATR in de laatste 15m-state
            base = symbol.split('/')[0]
            st = load_state(base, '15m')
            if st and st.get('atr_pct'):
                atr = entry * float(st['atr_pct']) / 100.0
            else:
                atr = entry * 0.015
                print(f"⚠️ Geen verse ATR voor {symbol}; terugval op 1.5% voor de stops.")
            sgn = 1 if side == 'long' else -1
            trade = active_trades[symbol] = {
                'side': side, 'entry': entry, 'peak': price, 'atr': atr,
                'sl': entry - sgn * SL_ATR_MULT * atr,
                'tp': entry + sgn * TP_ATR_MULT * atr,
            }

        atr = trade['atr']
        if side == 'long':
            if price > trade['peak']:
                trade['peak'] = price
                new_sl = price - SL_ATR_MULT * atr
                if new_sl > trade['sl']:
                    trade['sl'] = new_sl
                    print(f"📈 Trailing stop {symbol} omhoog naar ${trade['sl']:.4f}")
            stop_hit = price <= trade['sl']
            tp_hit = price >= trade['tp']
        else:
            if price < trade['peak']:
                trade['peak'] = price
                new_sl = price + SL_ATR_MULT * atr
                if new_sl < trade['sl']:
                    trade['sl'] = new_sl
                    print(f"📉 Trailing stop {symbol} omlaag naar ${trade['sl']:.4f}")
            stop_hit = price >= trade['sl']
            tp_hit = price <= trade['tp']

        if stop_hit or tp_hit:
            reason = 'stop' if stop_hit else 'take-profit'
            print(f"🛑 {reason} geraakt voor {side} {symbol} op ${price}. Positie sluiten...")
            if close_position(symbol, side, contracts):
                if pnl_fraction(side, entry, price) < 0:
                    symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)

    # Posities die niet meer open zijn (gesloten door de SL/TP op de exchange of handmatig)
    for symbol in list(active_trades.keys()):
        if symbol in open_symbols:
            continue
        trade = active_trades.pop(symbol)
        try:
            price = float(exchange.fetch_ticker(symbol)['last'])
            if pnl_fraction(trade['side'], trade['entry'], price) < 0:
                symbol_cooldowns[symbol] = time.time()
                print(f"ℹ️ {symbol} is extern gesloten, waarschijnlijk met verlies -> cooldown.")
        except Exception:
            pass


def ensure_margin_and_leverage(symbol):
    """Stelt margin mode en hefboom in. Retourneert True alleen als beide gelukt zijn."""
    if DRY_RUN:
        return True
    has = getattr(exchange, 'has', {}) or {}
    try:
        if has.get('setMarginMode'):
            for attempt in range(2):
                try:
                    exchange.set_margin_mode(DESIRED_MARGIN_MODE, symbol)
                    break
                except Exception as e:
                    if attempt == 1:
                        raise
                    print(f"⚠️ margin mode poging 1 mislukt voor {symbol}: {repr(e)}")
                    time.sleep(PAUSE_RETRY)
        for attempt in range(2):
            try:
                try:
                    exchange.set_leverage(LEVERAGE, symbol)
                except Exception:
                    exchange.set_leverage(LEVERAGE, symbol, {'marginMode': DESIRED_MARGIN_MODE})
                return True
            except Exception as e:
                if attempt == 1:
                    raise
                print(f"⚠️ leverage poging 1 mislukt voor {symbol}: {repr(e)}")
                time.sleep(PAUSE_RETRY)
    except Exception as e:
        print(f"⛔ Margin/leverage setup mislukt voor {symbol}: {repr(e)}")
    return False


def compute_contract_amount(symbol, price):
    """
    Rekent de gewenste notional (margin x hefboom) om naar hele contracten.
    Rondt naar BENEDEN af; neemt 1 contract alleen als dat de notional niet te ver overschrijdt.
    Retourneert None als er geen veilige hoeveelheid is.
    """
    target = TRADE_AMOUNT_USDT * LEVERAGE
    try:
        contract_size = float(exchange.market(symbol).get('contractSize') or 1.0)
    except Exception:
        print(f"⛔ Contractgrootte van {symbol} onbekend; trade overgeslagen.")
        return None
    per_contract = price * contract_size
    if per_contract <= 0:
        return None
    amount = math.floor(target / per_contract)
    if amount < 1:
        if per_contract <= target * MAX_NOTIONAL_OVERSHOOT:
            amount = 1
        else:
            print(f"⚠️ 1 contract {symbol} = ${per_contract:.2f}, ver boven de gewenste ${target:.2f}; overgeslagen.")
            return None
    return float(amount)


# =====================================================================
# SCAN
# =====================================================================
def resolve_symbol(asset):
    """Kiest de perpetual-markt. Valt terug op 'ASSET/USDT' maar alleen als dat echt een swap/contract is (nooit spot)."""
    primary = f"{asset}/USDT:USDT"
    if primary in exchange.markets:
        return primary
    fallback = f"{asset}/USDT"
    m = exchange.markets.get(fallback)
    if m and (m.get('swap') or m.get('contract')):
        return fallback
    return None


def execute_live_trades_scan():
    """Zoekt nieuwe trades (elke 15 min) op basis van harde filters + puntensysteem."""
    pull_latest_states_from_github()
    print("--- [Futures Live Trader Worker] Start scan ---")

    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin Futures: {e}")
        return

    try:
        active_symbols = {p.get('symbol') for p in exchange.fetch_positions()
                          if abs(float(p.get('contracts') or 0)) > 0}
    except Exception as e:
        print(f"⛔ Kan open posities niet ophalen ({e}); scan afgebroken om dubbele posities te voorkomen.")
        return

    for asset in ASSETS:
        symbol = resolve_symbol(asset)
        if symbol is None:
            print(f"⚠️ Geen swap/futures-markt voor {asset} op {EXCHANGE_ID}; overgeslagen.")
            continue

        if symbol in active_symbols:
            print(f"⏳ Open positie voor {symbol}, geen nieuwe trade.")
            continue

        if symbol in symbol_cooldowns:
            elapsed = time.time() - symbol_cooldowns[symbol]
            if elapsed < COOLDOWN_SECONDS:
                print(f"⏳ Cooldown {symbol}: nog {int((COOLDOWN_SECONDS - elapsed) / 60)} min.")
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
        print(f"ℹ️ Geen bruikbare/verse 15m-state voor {asset}.")
        return

    signal = state_15m['signal']
    if signal == 'none':
        print(f"ℹ️ {asset}: model zonder eenduidig signaal (regressie en classifier oneens).")
        return
    is_long = signal == 'long'
    prob = float(state_15m.get('signal_probability', 50.0))
    pred_move_pct = abs(float(state_15m.get('predicted_return_pct', 0.0)))
    holdout_acc = state_15m.get('holdout_direction_accuracy')
    atr_pct = float(state_15m.get('atr_pct') or 0.0)

    price = float(exchange.fetch_ticker(symbol)['last'])

    # --- ATR/SL/TP uit de echte ATR van het model ---
    if atr_pct <= 0:
        print(f"⚠️ {asset}: geen ATR in state; overgeslagen.")
        return
    atr = price * atr_pct / 100.0
    sgn = 1 if is_long else -1
    stop_loss_val = price - sgn * SL_ATR_MULT * atr
    take_profit_val = price + sgn * TP_ATR_MULT * atr

    # ========== HARDE FILTERS ==========
    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
        print(f"⚠️ Volatiliteitsfilter {asset}: ATR {atr_pct:.2f}% buiten [{MIN_ATR_PCT}, {MAX_ATR_PCT}].")
        return
    rrr = abs(take_profit_val - price) / abs(price - stop_loss_val)
    if rrr < MIN_RRR:
        print(f"⚠️ RRR-filter {asset}: {rrr:.2f} < {MIN_RRR}")
        return
    if holdout_acc is None or holdout_acc < MIN_HOLDOUT_ACCURACY:
        print(f"⚠️ {asset}: out-of-sample richtingsnauwkeurigheid {holdout_acc} < {MIN_HOLDOUT_ACCURACY}; geen edge aangetoond.")
        return

    # ========== ZACHTE FILTERS ==========
    score, max_score = 0, 4

    # 1) 15m en 1h eens over de richting
    state_1h = load_state(asset, '1h')
    if state_1h is not None and state_1h['signal'] == signal:
        score += 1
    else:
        print(f"ℹ️ [Score] 1h-model bevestigt {signal} niet voor {asset} (geen punt).")

    # 2) Niet tegen de BTC-richting in (voor BTC zelf altijd ok)
    if asset == 'BTC':
        score += 1
    elif ENABLE_BTC_CORRELATION:
        btc = load_state('BTC', '15m')
        opposite = 'short' if is_long else 'long'
        if btc is not None and btc['signal'] != opposite:
            score += 1
        else:
            print(f"ℹ️ [Score] BTC-state onbekend of tegen {signal} voor {asset} (geen punt).")
    else:
        score += 1

    # 3) Orderboek-imbalance (bij fouten geen punt)
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        time.sleep(PAUSE_AFTER_CALL)
        bids = sum(b[1] for b in ob.get('bids', []))
        asks = sum(a[1] for a in ob.get('asks', []))
        if bids + asks > 0:
            ratio = (bids if is_long else asks) / (bids + asks)
            if ratio >= MIN_ORDERBOOK_IMBALANCE:
                score += 1
            else:
                print(f"ℹ️ [Score] Orderboekdruk {ratio:.2f} te laag voor {asset} (geen punt).")
    except Exception as e:
        print(f"ℹ️ [Score] Orderboek niet beschikbaar voor {asset}: {e} (geen punt).")

    # 4) Funding rate (bij fouten geen punt)
    try:
        funding = exchange.fetch_funding_rate(symbol).get('fundingRate')
        time.sleep(PAUSE_AFTER_CALL)
        if funding is not None and not ((is_long and funding > MAX_FUNDING_RATE) or
                                        ((not is_long) and funding < -MAX_FUNDING_RATE)):
            score += 1
        else:
            print(f"ℹ️ [Score] Ongunstige/onbekende funding voor {asset} (geen punt).")
    except Exception as e:
        print(f"ℹ️ [Score] Funding niet beschikbaar voor {asset}: {e} (geen punt).")

    direction_str = "LONG 📈" if is_long else "SHORT 📉"
    print(f"Beoordeling {asset} ({direction_str}) -> Score {score}/{max_score} | Kans {prob:.1f}% | "
          f"Voorspelde move {pred_move_pct:.3f}% | ATR {atr_pct:.2f}% | holdout-acc {holdout_acc:.2f}")

    if not (prob >= MIN_DIRECTION_PROBABILITY and pred_move_pct >= MIN_PREDICTED_RETURN_PCT and score >= MIN_SCORE):
        print(f"ℹ️ {asset}: onder drempel (kans>={MIN_DIRECTION_PROBABILITY}, move>={MIN_PREDICTED_RETURN_PCT}%, score>={MIN_SCORE}).")
        return

    # ========== UITVOERING ==========
    if not ensure_margin_and_leverage(symbol):
        print(f"⛔ Trade {asset} geannuleerd: margin/leverage niet gegarandeerd.")
        return
    amount = compute_contract_amount(symbol, price)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    print(f"🔥 {direction_str} {symbol}: {amount} contracten | SL ${stop_loss_val:.4f} | TP ${take_profit_val:.4f}")
    if DRY_RUN:
        print("[DRY_RUN] Geen order geplaatst.")
        return

    params = {
        'stopLoss': {'triggerPrice': stop_loss_val},
        'takeProfit': {'triggerPrice': take_profit_val},
        'marginMode': DESIRED_MARGIN_MODE,
    }
    order = None
    try:
        order = exchange.create_order(symbol, 'market', side, amount, None, params)
    except Exception as e:
        print(f"⚠️ Order met SL/TP-parameters mislukt ({repr(e)}); poging zonder (software-stop neemt het over).")
        try:
            order = exchange.create_order(symbol, 'market', side, amount, None,
                                          {'marginMode': DESIRED_MARGIN_MODE})
        except Exception as e2:
            print(f"⛔ Fout bij orderplaatsing voor {symbol}: {repr(e2)}")
            return

    print(f"✅ Order geplaatst: {order.get('id') if isinstance(order, dict) else order}")
    # De software-stop (elke minuut gecontroleerd) is altijd actief, ook als de exchange de SL/TP negeerde
    active_trades[symbol] = {
        'side': 'long' if is_long else 'short', 'entry': price, 'peak': price, 'atr': atr,
        'sl': stop_loss_val, 'tp': take_profit_val,
    }


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print(f"Futures Live Trader Worker gestart ({EXCHANGE_ID}, ${TRADE_AMOUNT_USDT} margin, {LEVERAGE}x {DESIRED_MARGIN_MODE}"
          f"{', DRY_RUN' if DRY_RUN else ''}).")
    last_trade_check_time = 0

    while True:
        try:
            manage_open_positions_and_trailing_stops()
        except Exception as e:
            print(f"Fout in positiebeheer lus: {e}")

        if time.time() - last_trade_check_time >= TRADE_CHECK_INTERVAL_SECONDS:
            try:
                execute_live_trades_scan()
            except Exception as e:
                print(f"Fout in trades scan lus: {e}")
            last_trade_check_time = time.time()

        time.sleep(POSITION_CHECK_INTERVAL_SECONDS)
