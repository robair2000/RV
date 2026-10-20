import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt
import pandas as pd

from github_utils import github_pull, github_put
from asset_utils import get_combined_trading_assets
from order_utils import (cancel_orders, describe_sizing, fetch_all_open_orders, find_protective_levels,
                         replace_protective_orders, replace_stop_loss)
from risk_utils import DAILY_LOSS_CLOSE_POSITIONS, DailyLossGuard, log_trade

BOT = 'V20'
PREFIX = 'v20_'


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/ranking':
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            html = "<html><head><title>V20 Trader - Live Ranking</title></head><body style='font-family: Arial; padding: 20px; background-color: #f4f4f9;'>"
            html += "<h2 style='color: #333;'>🏆 V20 Live Kucoin Coins Ranking (Trader Dashboard - Gehard)</h2>"

            ranking_file = 'kucoin_best_coins_ranking.csv'
            if os.path.exists(ranking_file):
                try:
                    df = pd.read_csv(ranking_file)
                    html += df.to_html(index=False, classes='table', border=1)
                except Exception as e:
                    html += f"<p style='color: red;'>⚠️ Fout bij lezen van CSV: {e}</p>"
            else:
                html += "<p style='color: orange;'>⏳ Nog geen ranking CSV gevonden. De scanner moet nog draaien.</p>"

            html += "</body></html>"
            self.wfile.write(html.encode("utf-8"))
        else:
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


# --- Pull-frequentie: state verandert per 30 min tot enkele uren, dus 60s-polling is verspilling ---
STATE_PULL_INTERVAL = int(os.environ.get('STATE_PULL_INTERVAL', 600))

AGENT = 'trader-worker'


def pull_ranking_from_github():
    github_pull('kucoin_best_coins_ranking.csv', min_interval=STATE_PULL_INTERVAL, agent=AGENT)




TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 20.0))
MIN_BOSS_SCORE = float(os.environ.get('MIN_BOSS_SCORE', 84.0))
COOLDOWN_SECONDS = int(os.environ.get('COOLDOWN_SECONDS', 7200))

DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

MAX_CONCURRENT_POSITIONS = int(os.environ.get('MAX_CONCURRENT_POSITIONS', 5))
# Crypto-majors zijn sterk gecorreleerd: 5 longs tegelijk in één cross-pool is feitelijk één gehevelde bet.
# Max aantal posities/orders van deze bot in DEZELFDE richting. 0 = uit (ongewijzigd gedrag).
MAX_SAME_DIRECTION = int(os.environ.get('MAX_SAME_DIRECTION', 0))
MIN_RRR = 3.0
TP_ATR_mult_BASE = 5.0
MAX_STATE_AGE_SECONDS = {'4h': 8 * 3600, '1d': 48 * 3600}
# last_candle_ts = OPEN-tijd van de laatst gesloten 4h-candle, dus een vers signaal is al ~4h oud.
# 5.5u = 4h + agent-scaninterval (30 min) + pull-interval + marge; ouder = signaal van 2 candles terug.
MAX_CANDLE_AGE_SECONDS = float(os.environ.get('MAX_CANDLE_AGE_HOURS', 5.5)) * 3600
# Een limit-entry geldt maar voor de candle waarop het signaal gebaseerd is (default 1 x 4h)
PENDING_ORDER_TTL_SECONDS = int(os.environ.get('PENDING_ORDER_TTL_SECONDS', 4 * 3600))

active_trades = {}
symbol_cooldowns = {}
symbol_last_traded_candles = {}


exchange = ccxt.kucoinfutures({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
})

guard = DailyLossGuard(exchange)


def pull_status_from_github():
    global active_trades, symbol_cooldowns, symbol_last_traded_candles
    if not github_pull('trader_status.json', timeout=10, agent=AGENT):
        print("ℹ️ Geen eerdere V20 status gevonden op GitHub, starten met schone lei.")
        return
    try:
        with open('trader_status.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        active_trades = data.get('active_trades', {})
        symbol_cooldowns = data.get('cooldowns', {})
        symbol_last_traded_candles = data.get('last_traded_candles', {})
        guard.state = data.get('daily_guard', {}) or {}
        print(f"📥 V20 Status hersteld van GitHub: {len(active_trades)} actieve/pending trades.")
    except Exception as e:
        print(f"⚠️ V20 status-bestand onleesbaar ({e}), starten met schone lei.")


def execute_with_retry(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2


def push_trader_status_to_github():
    data = {
        'updated_at': time.time(),
        'active_trades': active_trades,
        'cooldowns': symbol_cooldowns,
        'last_traded_candles': symbol_last_traded_candles,
        'daily_guard': guard.state,
    }
    filename = 'trader_status.json'
    content_bytes = json.dumps(data, indent=2).encode('utf-8')

    with open(filename, 'wb') as f:
        f.write(content_bytes)

    if os.environ.get('GITHUB_TOKEN'):
        if not github_put(filename, content_bytes, message="Auto-update trader_status.json [skip ci]", agent=AGENT):
            print("⚠️ trader_status.json lokaal bijgewerkt maar NIET naar GitHub gepusht.")


def pull_latest_states_from_github(symbols):
    files_to_pull = set()
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        files_to_pull.add(f'model_state_{asset.lower()}_4h.json')
        files_to_pull.add(f'pattern_state_{asset.lower()}_4h.json')
    for filename in files_to_pull:
        github_pull(filename, min_interval=STATE_PULL_INTERVAL, agent=AGENT)


def load_state(asset, tf):
    path = f'model_state_{asset.lower()}_{tf}.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            st = json.load(f)
    except Exception as e:
        print(f"⚠️ Model state {path} onleesbaar: {e}")
        return None
    if st.get('data_status') != 'ok' or st.get('signal') not in ('long', 'short', 'none'):
        return None
    if time.time() - float(st.get('updated_at', 0) or 0) > MAX_STATE_AGE_SECONDS.get(tf, 86400):
        return None
    return st


MAX_PATTERN_AGE_SECONDS = 8 * 3600


def load_pattern_state(asset):
    path = f'pattern_state_{asset.lower()}_4h.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            pst = json.load(f)
    except Exception as e:
        print(f"⚠️ Pattern state {path} onleesbaar: {e}")
        return None

    updated_at = float(pst.get('updated_at', 0) or 0)
    raw_candle_ts = float(pst.get('last_candle_ts', 0) or 0)
    current_time = time.time()

    if current_time - updated_at > MAX_PATTERN_AGE_SECONDS:
        return None

    # FIX (bug 1): agent.py schrijft seconden; oudere states kunnen milliseconden bevatten.
    # Detecteer de eenheid i.p.v. blind door 1000 te delen.
    if raw_candle_ts > 0:
        candle_ts_sec = raw_candle_ts / 1000.0 if raw_candle_ts > 1e10 else raw_candle_ts
        candle_age_seconds = current_time - candle_ts_sec
        if candle_age_seconds > MAX_CANDLE_AGE_SECONDS:
            print(f"⚠️ V20 Waarschuwing: Pattern state voor {asset} heeft te oude candle ({int(candle_age_seconds / 3600)} uur oud).")
            return None

    return pst


def _candle_ts_seconds(pst):
    raw = float(pst.get('last_candle_ts', 0) or 0)
    return int(raw / 1000.0 if raw > 1e10 else raw)


def last_price(symbol, default=None):
    try:
        return float(exchange.fetch_ticker(symbol)['last'])
    except Exception:
        return default


def close_position(symbol, side, contracts):
    close_side = 'sell' if side == 'long' else 'buy'
    if DRY_RUN:
        return True
    try:
        try:
            cancel_orders(exchange, fetch_all_open_orders(exchange, symbol), PREFIX)
        except Exception as e:
            print(f"⚠️ V20: openstaande orders annuleren voor {symbol} mislukt: {e}")

        execute_with_retry(lambda: exchange.create_order(
            symbol, 'market', close_side, contracts, None,
            {'reduceOnly': True, 'marginMode': DESIRED_MARGIN_MODE}
        ))
        return True
    except Exception as e:
        print(f"⚠️ V20 sluitfout voor {symbol}: {e}")
        return False


def same_direction_count(side):
    return sum(1 for t in active_trades.values() if t.get('side') == side)


def adopt_orphan_position(symbol, side, contracts, pos_entry, open_orders):
    """Bouwt de trade-administratie opnieuw op voor een open positie zonder state-entry.

    Voorwaarden (anders None, positie blijft ongemoeid): open orders zijn bekend, de entry-prijs is bekend
    en er staat een eigen stop-loss (clientOrderId 'v20_sl_*') voor dit symbool. ATR komt uit de 4h model-state
    (zelfde bron als bij entry), met terugval op |entry - SL| / SL-multiple. De trailing hervat vanaf de
    huidige koers; een al verschoven SL blijft de SL van de exchange-order."""
    if open_orders is None or pos_entry <= 0 or side not in ('long', 'short'):
        return None
    sl, tp = find_protective_levels(open_orders, symbol, PREFIX)
    if sl is None:
        return None
    sgn = 1 if side == 'long' else -1
    price = last_price(symbol, default=pos_entry)
    asset = symbol.split('/')[0].split(':')[0]
    st = load_state(asset, '4h')
    atr_pct = float((st or {}).get('atr_pct') or 0.0)
    atr = price * atr_pct / 100.0 if atr_pct > 0 else abs(pos_entry - sl) / 2.5
    if not atr > 0:
        return None
    tp_val = tp if tp else pos_entry + sgn * max(TP_ATR_mult_BASE, 2.5 * (MIN_RRR + 0.2)) * atr
    trade = {
        'status': 'open', 'created_at': time.time(), 'adopted': True, 'side': side,
        'entry': pos_entry, 'peak': max(pos_entry, price) if side == 'long' else min(pos_entry, price),
        'atr': atr, 'sl': sl, 'tp': tp_val, 'trailing_mult': 2.5,
        'breakeven_secured': (sl >= pos_entry) if side == 'long' else (sl <= pos_entry),
        'protected_contracts': contracts,
    }
    print(f"♻️ V20: positie {symbol} ({side} {contracts}) overgenomen na verloren state | entry {pos_entry} SL {sl} TP {tp_val}")
    log_trade(BOT, 'adopted', symbol, side, price=price, amount=contracts, entry=pos_entry, sl=sl, tp=tp_val,
              note='state verloren; overgenomen via eigen v20_sl_-order')
    return trade


def manage_open_positions_and_trailing_stops(symbols):
    try:
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception as e:
        print(f"⚠️ V20: posities ophalen mislukt: {e}")
        return

    try:
        open_orders = fetch_all_open_orders(exchange)
    except Exception as e:
        print(f"⚠️ V20: open orders ophalen mislukt: {e}")
        open_orders = None

    now = time.time()
    open_symbols = set()
    state_changed = False

    for pos in positions:
        contracts = abs(float(pos.get('contracts') or 0))
        if contracts == 0:
            continue
        symbol = pos.get('symbol')
        side = pos.get('side')
        if side not in ('long', 'short'):
            continue

        trade = active_trades.get(symbol)
        if trade is None:
            # Status verloren (herstart zonder GitHub-state)? Neem een positie over die aantoonbaar van deze bot
            # is: alleen als er een eigen v20_sl_-order voor staat. Handmatige posities blijven onaangeroerd.
            trade = adopt_orphan_position(symbol, side, contracts, float(pos.get('entryPrice') or 0), open_orders)
            if trade is None:
                continue
            active_trades[symbol] = trade
            state_changed = True

        open_symbols.add(symbol)
        pos_entry = float(pos.get('entryPrice') or 0)
        if pos_entry > 0:
            trade['entry'] = pos_entry
        entry = float(trade['entry'])

        # Limit order (deels) gevuld
        if trade.get('status') == 'pending':
            print(f"🚀 V20: Limit order gevuld voor {symbol} ({contracts} contracten)! Rest van de entry wordt geannuleerd, SL/TP volgen.")
            if not DRY_RUN and open_orders is not None:
                cancel_orders(exchange, [o for o in open_orders if o.get('symbol') == symbol], f"{PREFIX}entry_")
            trade['status'] = 'open'
            trade['filled_at'] = now
            state_changed = True
            log_trade(BOT, 'entry_filled', symbol, side, price=entry, amount=contracts,
                      entry=entry, sl=trade['sl'], tp=trade['tp'])

        # FIX deelvullingen: bescherming altijd afstemmen op de ACTUELE positiegrootte
        if trade.get('status') == 'open' and not DRY_RUN and trade.get('protected_contracts') != contracts:
            if replace_protective_orders(exchange, symbol, side, contracts, trade['sl'], trade['tp'], PREFIX):
                trade['protected_contracts'] = contracts
                state_changed = True
                print(f"🛡️ V20: SL/TP voor {symbol} gezet op {contracts} contracten.")
            else:
                print(f"🚨 V20: positie {symbol} heeft GEEN stop-loss (plaatsen mislukt); nieuwe poging volgende ronde.")

        price = last_price(symbol)
        if price is None:
            continue

        atr = trade['atr']
        trail_mult = trade.get('trailing_mult', 2.5)
        sgn = 1 if side == 'long' else -1

        distance_in_atr = sgn * (price - entry) / atr if atr > 0 else 0.0
        old_sl = trade['sl']

        if distance_in_atr >= 2.4 and not trade.get('breakeven_secured', False):
            if side == 'long' and trade['sl'] < entry:
                trade['sl'] = entry
                trade['breakeven_secured'] = True
                state_changed = True
            elif side == 'short' and trade['sl'] > entry:
                trade['sl'] = entry
                trade['breakeven_secured'] = True
                state_changed = True

        if side == 'long':
            if price > trade['peak']:
                trade['peak'] = price
                if distance_in_atr >= 2.4:
                    new_sl = price - (trail_mult * 0.8) * atr
                    if new_sl > trade['sl']:
                        trade['sl'] = new_sl
                        state_changed = True
            stop_hit = price <= trade['sl']
            tp_hit = price >= trade['tp']
        else:
            if price < trade['peak']:
                trade['peak'] = price
                if distance_in_atr >= 2.4:
                    new_sl = price + (trail_mult * 0.8) * atr
                    if new_sl < trade['sl']:
                        trade['sl'] = new_sl
                        state_changed = True
            stop_hit = price >= trade['sl']
            tp_hit = price <= trade['tp']

        if trade['sl'] != old_sl and trade.get('status') == 'open' and not DRY_RUN:
            replace_stop_loss(exchange, symbol, side, contracts, trade['sl'], PREFIX)

        if stop_hit or tp_hit:
            if close_position(symbol, side, contracts):
                reason = 'tp' if tp_hit else 'stop'
                log_trade(BOT, f'exit_{reason}', symbol, side, price=price, amount=contracts,
                          entry=entry, sl=trade['sl'], tp=trade['tp'], note='software exit')
                symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)
                state_changed = True

    # Pending entries: TTL + opruiming (bug 4). Orders met clientOrderId v20_entry_* tellen als entry.
    entry_order_symbols = None
    if open_orders is not None:
        entry_order_symbols = {o.get('symbol') for o in open_orders
                               if str(o.get('clientOrderId') or (o.get('info') or {}).get('clientOid') or '').startswith(f"{PREFIX}entry_")}

    to_remove = []
    for sym, trd in list(active_trades.items()):
        if sym in open_symbols:
            continue
        age = now - trd.get('created_at', now)
        if trd.get('status') == 'pending':
            has_order = True if entry_order_symbols is None else (sym in entry_order_symbols)
            if has_order and age > PENDING_ORDER_TTL_SECONDS:
                if not DRY_RUN and open_orders is not None:
                    n = cancel_orders(exchange, [o for o in open_orders if o.get('symbol') == sym], f"{PREFIX}entry_")
                    print(f"⏱️ V20: verlopen limit order voor {sym} geannuleerd ({n} order(s), {age / 3600:.1f} uur oud).")
                log_trade(BOT, 'cancel_stale_entry', sym, trd.get('side', ''), entry=trd.get('entry'),
                          note=f'ttl {PENDING_ORDER_TTL_SECONDS}s')
                to_remove.append(sym)
            elif not has_order and age > 120:
                to_remove.append(sym)  # niet gevuld en niet meer open (geannuleerd/afgewezen)
        elif trd.get('status') == 'open':
            # Positie is op de exchange gesloten (SL/TP getriggerd): restanten opruimen + cooldown
            if not DRY_RUN and open_orders is not None:
                cancel_orders(exchange, [o for o in open_orders if o.get('symbol') == sym], PREFIX)
            px = last_price(sym, default=trd.get('entry'))
            log_trade(BOT, 'exit_exchange', sym, trd.get('side', ''), price=px, entry=trd.get('entry'),
                      sl=trd.get('sl'), tp=trd.get('tp'), note='positie gesloten op exchange (prijs = laatste koers, indicatief)')
            symbol_cooldowns[sym] = time.time()
            to_remove.append(sym)

    if to_remove:
        for sym in to_remove:
            active_trades.pop(sym, None)
        state_changed = True

    if state_changed or guard.dirty:
        guard.dirty = False
        push_trader_status_to_github()


def ensure_margin_and_leverage(symbol, target_leverage):
    if DRY_RUN:
        return True
    try:
        exchange.set_margin_mode(DESIRED_MARGIN_MODE, symbol)
        exchange.set_leverage(target_leverage, symbol)
        return True
    except Exception as e:
        print(f"⚠️ V20: margin/leverage instellen voor {symbol} mislukt: {e}")
        return False


def compute_contract_amount(asset, symbol, price, leverage):
    target_notional = TRADE_AMOUNT_USDT * leverage

    try:
        market_info = exchange.market(symbol)
        contract_size = float(market_info.get('contractSize') or 1.0)
        min_amount = float(market_info.get('limits', {}).get('amount', {}).get('min') or 1.0)
    except Exception:
        contract_size = 1.0
        min_amount = 1.0

    per_contract_notional = price * contract_size
    if per_contract_notional <= 0:
        return None

    lots = math.floor(target_notional / per_contract_notional)
    if lots < max(1, min_amount):
        print(f"ℹ️ V20: {symbol} overgeslagen, 1 contract (${per_contract_notional:,.2f}) > notional ${target_notional:,.2f}.")
        return None

    return float(lots)


def flatten_bot_positions(positions):
    """Dagverlies-limiet bereikt: pending entries annuleren en bot-posities sluiten (eenmalig per dag)."""
    if guard.state.get('flattened'):
        return
    print("🛑 V20: dagverlies-limiet bereikt -> pending entries annuleren en bot-posities sluiten.")
    try:
        orders = fetch_all_open_orders(exchange)
        cancel_orders(exchange, orders, f"{PREFIX}entry_")
    except Exception as e:
        print(f"⚠️ V20: entries annuleren mislukt: {e}")
    held = {p.get('symbol'): p for p in positions if abs(float(p.get('contracts') or 0)) > 0}
    for sym in list(active_trades.keys()):
        trd = active_trades[sym]
        p = held.get(sym)
        if p is not None and p.get('side') in ('long', 'short'):
            c = abs(float(p.get('contracts') or 0))
            px = last_price(sym, default=trd.get('entry'))
            if close_position(sym, p['side'], c):
                log_trade(BOT, 'exit_daily_loss', sym, p['side'], price=px, amount=c, entry=trd.get('entry'),
                          note='MAX_DAILY_LOSS_USDT bereikt')
                symbol_cooldowns[sym] = time.time()
                active_trades.pop(sym, None)
        elif trd.get('status') == 'pending':
            active_trades.pop(sym, None)
    guard.state['flattened'] = True
    guard.dirty = True


def execute_live_trades_scan(symbols):
    current_time = time.time()
    expired_cooldowns = [sym for sym, ts in symbol_cooldowns.items() if current_time - ts > COOLDOWN_SECONDS]
    for sym in expired_cooldowns:
        symbol_cooldowns.pop(sym, None)
    if expired_cooldowns:
        push_trader_status_to_github()

    pull_latest_states_from_github(symbols)
    try:
        exchange.load_markets()
        positions = execute_with_retry(lambda: exchange.fetch_positions())
        open_orders = fetch_all_open_orders(exchange)
    except Exception as e:
        print(f"⚠️ Fout bij ophalen posities/orders van KuCoin Futures: {e}")
        return

    # Kill-switch: max dagverlies (MAX_DAILY_LOSS_USDT, 0 = uit)
    halted = guard.check()
    if halted and DAILY_LOSS_CLOSE_POSITIONS:
        flatten_bot_positions(positions)
    if guard.dirty:
        guard.dirty = False
        push_trader_status_to_github()
    if halted:
        return

    active_symbols = set()
    for p in positions:
        if abs(float(p.get('contracts') or 0)) > 0:
            active_symbols.add(p.get('symbol'))

    pending_order_symbols = {o.get('symbol') for o in open_orders if not o.get('_stop')}

    my_active_count = len([s for s in active_symbols if s in active_trades]) + \
        len([s for s in pending_order_symbols if s in active_trades and s not in active_symbols])
    if my_active_count >= MAX_CONCURRENT_POSITIONS:
        return

    for symbol in symbols:
        if symbol not in exchange.markets or symbol in symbol_cooldowns:
            continue
        if symbol in active_symbols or symbol in pending_order_symbols:
            continue
        try:
            asset = symbol.split('/')[0].split(':')[0]
            evaluate_and_trade(asset, symbol)
        except Exception as e:
            print(f"⚠️ Fout bij evalueren van {symbol}: {e}")


def evaluate_and_trade(asset, symbol):
    state_4h = load_state(asset, '4h')
    if state_4h is None:
        return

    quant_signal = state_4h['signal']
    if quant_signal not in ('long', 'short'):
        return

    pattern_st = load_pattern_state(asset)
    if pattern_st is None:
        return

    candle_ts = _candle_ts_seconds(pattern_st)
    raw_last = float(symbol_last_traded_candles.get(symbol, 0) or 0)
    last_traded_ts = int(raw_last / 1000.0 if raw_last > 1e10 else raw_last)  # oude states kunnen ms bevatten
    if candle_ts > 0 and candle_ts <= last_traded_ts:
        return

    price = float(exchange.fetch_ticker(symbol)['last'])
    atr_pct = float(state_4h.get('atr_pct') or 0.0)
    if atr_pct <= 0:
        return
    atr = price * atr_pct / 100.0

    boss_signal = pattern_st.get('pattern_signal', 'none')
    boss_score = float(pattern_st.get('pattern_score', 50.0))
    boss_leverage = int(pattern_st.get('recommended_leverage', 2))
    boss_trailing_mult = float(pattern_st.get('recommended_trailing_mult', 2.5))
    # FIX (bug 3): de strengste van env-drempel (MIN_BOSS_SCORE) en agent-drempel geldt
    min_score_req = max(MIN_BOSS_SCORE, float(pattern_st.get('min_score_required', MIN_BOSS_SCORE)))

    if boss_signal != quant_signal or boss_score < min_score_req:
        return

    is_long = (quant_signal == 'long')
    if MAX_SAME_DIRECTION > 0 and same_direction_count(quant_signal) >= MAX_SAME_DIRECTION:
        print(f"ℹ️ V20: {symbol} {quant_signal} overgeslagen, al {MAX_SAME_DIRECTION} posities/orders in die richting (MAX_SAME_DIRECTION).")
        return
    direction_prob = float(state_4h.get('direction_probability', 50.0))
    if is_long and direction_prob < 50.0:
        return
    if not is_long and direction_prob > 50.0:
        return

    if is_long:
        limit_price = price - (0.2 * atr)
    else:
        limit_price = price + (0.2 * atr)

    sgn = 1 if is_long else -1
    stop_loss_val = limit_price - sgn * boss_trailing_mult * atr
    effective_tp_mult = max(TP_ATR_mult_BASE, boss_trailing_mult * (MIN_RRR + 0.2))
    take_profit_val = limit_price + sgn * effective_tp_mult * atr

    # Eerst sizing: als 1 contract al groter is dan de notional slaan we over, ZONDER margin/leverage
    # van dit symbool (en dus van een eventuele bestaande positie/andere bot) aan te raken.
    amount = compute_contract_amount(asset, symbol, limit_price, boss_leverage)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    if DRY_RUN:
        print(f"   🧪 [DRY RUN V20] Zou limit order plaatsen: {side} {amount}x {symbol} @ {limit_price} (score {boss_score} >= {min_score_req})")
        return

    if not ensure_margin_and_leverage(symbol, boss_leverage):
        return

    try:
        # Zuivere limit entry; SL/TP volgen zodra (een deel van) de order gevuld is
        execute_with_retry(lambda: exchange.create_order(symbol, 'limit', side, amount, limit_price, {
            'marginMode': DESIRED_MARGIN_MODE,
            'clientOrderId': f"{PREFIX}entry_{int(time.time() * 1000)}"
        }))

        print(f"   🚀 V20 SUCCES: Limit order geplaatst voor {symbol} ({side} {amount} contracten @ {limit_price}). Wacht op fill voor SL/TP.")

        if candle_ts > 0:
            symbol_last_traded_candles[symbol] = candle_ts

        active_trades[symbol] = {
            'status': 'pending',
            'created_at': time.time(),
            'side': 'long' if is_long else 'short',
            'entry': limit_price,
            'peak': limit_price,
            'atr': atr,
            'sl': stop_loss_val,
            'tp': take_profit_val,
            'trailing_mult': boss_trailing_mult,
            'breakeven_secured': False,
            'protected_contracts': None,
        }
        log_trade(BOT, 'entry_order', symbol, 'long' if is_long else 'short', price=limit_price, amount=amount,
                  entry=limit_price, sl=stop_loss_val, tp=take_profit_val, note=f'score={boss_score} lev={boss_leverage}')
        push_trader_status_to_github()
    except Exception as e:
        print(f"   ❌ Fout bij plaatsen limit order voor {symbol}: {e}")


if __name__ == '__main__':
    pull_status_from_github()
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 V20 Trader Worker gestart (scan interval: 60 seconden).")
    try:
        exchange.load_markets()
        print(f"📏 Sizing-check (margin {TRADE_AMOUNT_USDT} USDT, hefboom 2x = notional {TRADE_AMOUNT_USDT * 2:.0f} USDT; V20 gebruikt 1-3x):")
        for _s, _per, _ok in describe_sizing(exchange, get_combined_trading_assets(), TRADE_AMOUNT_USDT, 2):
            flag = '❓' if _ok is None else ('✅' if _ok else '⛔ wordt overgeslagen')
            print(f"   {flag} {_s}: 1 contract = {'?' if _per is None else f'${_per:,.2f}'}")
    except Exception as _e:
        print(f"⚠️ Sizing-check mislukt: {_e}")
    if guard.enabled:
        print(f"🛡️ MAX_DAILY_LOSS_USDT actief: {guard.max_loss} USDT (sluit posities bij limiet: {DAILY_LOSS_CLOSE_POSITIONS}).")
    else:
        print("⚠️ MAX_DAILY_LOSS_USDT is 0 -> kill-switch UIT. Zet de env-variabele om dagverlies te begrenzen.")
    while True:
        try:
            pull_ranking_from_github()
            current_symbols = get_combined_trading_assets()

            manage_open_positions_and_trailing_stops(current_symbols)
            execute_live_trades_scan(current_symbols)
        except Exception as e:
            print(f"⚠️ Fout in trader lus: {e}")
        time.sleep(60)
