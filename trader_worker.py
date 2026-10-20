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

BOT = 'V20_QUANT'
PREFIX = 'v20q_'


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/ranking':
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            html = "<html><head><title>V20 Quant Trader</title></head><body style='font-family: Arial; padding: 20px; background-color: #0f172a; color: #f8fafc;'>"
            html += "<h2 style='color: #38bdf8;'>🚀 V20 Quant-Only Live Trader Dashboard</h2>"

            ranking_file = 'kucoin_best_coins_ranking.csv'
            if os.path.exists(ranking_file):
                try:
                    df = pd.read_csv(ranking_file)
                    html += df.to_html(index=False, classes='table', border=1)
                except Exception as e:
                    html += f"<p style='color: red;'>⚠️ Fout bij lezen van CSV: {e}</p>"
            else:
                html += "<p style='color: orange;'>⏳ Nog geen ranking CSV gevonden.</p>"

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


STATE_PULL_INTERVAL = int(os.environ.get('STATE_PULL_INTERVAL', 600))
AGENT = 'trader-worker'


def pull_ranking_from_github():
    github_pull('kucoin_best_coins_ranking.csv', min_interval=STATE_PULL_INTERVAL, agent=AGENT)


TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 20.0))
COOLDOWN_SECONDS = int(os.environ.get('COOLDOWN_SECONDS', 7200))

DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

MAX_CONCURRENT_POSITIONS = int(os.environ.get('MAX_CONCURRENT_POSITIONS', 5))
MAX_SAME_DIRECTION = int(os.environ.get('MAX_SAME_DIRECTION', 0))

MAX_STATE_AGE_SECONDS = {'4h': 8 * 3600, '1d': 48 * 3600}
MAX_CANDLE_AGE_SECONDS = float(os.environ.get('MAX_CANDLE_AGE_HOURS', 5.5)) * 3600
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
    if not github_pull('trader_status_quant.json', timeout=10, agent=AGENT):
        print("ℹ️ Geen eerdere Quant trader status gevonden op GitHub, starten met schone lei.")
        return
    try:
        with open('trader_status_quant.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        active_trades = data.get('active_trades', {})
        symbol_cooldowns = data.get('cooldowns', {})
        symbol_last_traded_candles = data.get('last_traded_candles', {})
        guard.state = data.get('daily_guard', {}) or {}
        print(f"📥 Quant Status hersteld van GitHub: {len(active_trades)} actieve/pending trades.")
    except Exception as e:
        print(f"⚠️ Quant status-bestand onleesbaar ({e}), starten met schone lei.")


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
    filename = 'trader_status_quant.json'
    content_bytes = json.dumps(data, indent=2).encode('utf-8')

    with open(filename, 'wb') as f:
        f.write(content_bytes)

    if os.environ.get('GITHUB_TOKEN'):
        github_put(filename, content_bytes, message="Auto-update trader_status_quant.json [skip ci]", agent=AGENT)


def pull_latest_states_from_github(symbols):
    files_to_pull = set()
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        files_to_pull.add(f'model_state_{asset.lower()}_4h.json')
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

    if st.get('data_status') != 'ok' or st.get('signal') not in ('long', 'short'):
        return None

    updated_at = float(st.get('updated_at', 0) or 0)
    raw_candle_ts = float(st.get('last_candle_ts', 0) or 0)
    current_time = time.time()

    if current_time - updated_at > MAX_STATE_AGE_SECONDS.get(tf, 86400):
        return None

    if raw_candle_ts > 0:
        candle_ts_sec = raw_candle_ts / 1000.0 if raw_candle_ts > 1e10 else raw_candle_ts
        if current_time - candle_ts_sec > MAX_CANDLE_AGE_SECONDS:
            return None

    return st


def _candle_ts_seconds(st):
    raw = float(st.get('last_candle_ts', 0) or 0)
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
        except Exception:
            pass

        execute_with_retry(lambda: exchange.create_order(
            symbol, 'market', close_side, contracts, None,
            {'reduceOnly': True, 'marginMode': DESIRED_MARGIN_MODE}
        ))
        return True
    except Exception as e:
        print(f"⚠️ Sluitfout voor {symbol}: {e}")
        return False


def same_direction_count(side):
    return sum(1 for t in active_trades.values() if t.get('side') == side)


def adopt_orphan_position(symbol, side, contracts, pos_entry, open_orders):
    if open_orders is None or pos_entry <= 0 or side not in ('long', 'short'):
        return None
    sl, tp = find_protective_levels(open_orders, symbol, PREFIX)
    if sl is None:
        return None
    price = last_price(symbol, default=pos_entry)
    asset = symbol.split('/')[0].split(':')[0]
    st = load_state(asset, '4h')
    atr_pct = float((st or {}).get('atr_pct') or 0.0)
    atr = price * atr_pct / 100.0 if atr_pct > 0 else abs(pos_entry - sl) / 2.5
    if not atr > 0:
        return None
    sgn = 1 if side == 'long' else -1
    tp_val = tp if tp else pos_entry + sgn * 5.0 * atr
    trade = {
        'status': 'open', 'created_at': time.time(), 'adopted': True, 'side': side,
        'entry': pos_entry, 'peak': max(pos_entry, price) if side == 'long' else min(pos_entry, price),
        'atr': atr, 'sl': sl, 'tp': tp_val,
        'breakeven_secured': (sl >= pos_entry) if side == 'long' else (sl <= pos_entry),
        'protected_contracts': contracts,
    }
    print(f"♻️ Quant: positie {symbol} ({side} {contracts}) overgenomen | entry {pos_entry} SL {sl} TP {tp_val}")
    log_trade(BOT, 'adopted', symbol, side, price=price, amount=contracts, entry=pos_entry, sl=sl, tp=tp_val)
    return trade


def manage_open_positions_and_trailing_stops(symbols):
    try:
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception as e:
        print(f"⚠️ Posities ophalen mislukt: {e}")
        return

    try:
        open_orders = fetch_all_open_orders(exchange)
    except Exception:
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

        if trade.get('status') == 'pending':
            print(f"🚀 Limit order gevuld voor {symbol} ({contracts} contracten)!")
            if not DRY_RUN and open_orders is not None:
                cancel_orders(exchange, [o for o in open_orders if o.get('symbol') == symbol], f"{PREFIX}entry_")
            trade['status'] = 'open'
            state_changed = True
            log_trade(BOT, 'entry_filled', symbol, side, price=entry, amount=contracts, entry=entry, sl=trade['sl'], tp=trade['tp'])

        if trade.get('status') == 'open' and not DRY_RUN and trade.get('protected_contracts') != contracts:
            if replace_protective_orders(exchange, symbol, side, contracts, trade['sl'], trade['tp'], PREFIX):
                trade['protected_contracts'] = contracts
                state_changed = True

        price = last_price(symbol)
        if price is None:
            continue

        atr = trade['atr']
        sgn = 1 if side == 'long' else -1
        distance_in_atr = sgn * (price - entry) / atr if atr > 0 else 0.0
        old_sl = trade['sl']

        if distance_in_atr >= 2.0 and not trade.get('breakeven_secured', False):
            if side == 'long' and trade['sl'] < entry:
                trade['sl'] = entry
                trade['breakeven_secured'] = True
                state_changed = True
            elif side == 'short' and trade['sl'] > entry:
                trade['sl'] = entry
                trade['breakeven_secured'] = True
                state_changed = True

        if side == 'long':
            stop_hit = price <= trade['sl']
            tp_hit = price >= trade['tp']
        else:
            stop_hit = price >= trade['sl']
            tp_hit = price <= trade['tp']

        if trade['sl'] != old_sl and trade.get('status') == 'open' and not DRY_RUN:
            replace_stop_loss(exchange, symbol, side, contracts, trade['sl'], PREFIX)

        if stop_hit or tp_hit:
            if close_position(symbol, side, contracts):
                reason = 'tp' if tp_hit else 'stop'
                log_trade(BOT, f'exit_{reason}', symbol, side, price=price, amount=contracts, entry=entry, sl=trade['sl'], tp=trade['tp'])
                symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)
                state_changed = True

    entry_order_symbols = None
    if open_orders is not None:
        entry_order_symbols = {o.get('symbol') for o in open_orders
                               if str(o.get('clientOrderId') or '').startswith(f"{PREFIX}entry_")}

    to_remove = []
    for sym, trd in list(active_trades.items()):
        if sym in open_symbols:
            continue
        age = now - trd.get('created_at', now)
        if trd.get('status') == 'pending':
            has_order = True if entry_order_symbols is None else (sym in entry_order_symbols)
            if has_order and age > PENDING_ORDER_TTL_SECONDS:
                if not DRY_RUN and open_orders is not None:
                    cancel_orders(exchange, [o for o in open_orders if o.get('symbol') == sym], f"{PREFIX}entry_")
                to_remove.append(sym)
            elif not has_order and age > 120:
                to_remove.append(sym)
        elif trd.get('status') == 'open':
            if not DRY_RUN and open_orders is not None:
                cancel_orders(exchange, [o for o in open_orders if o.get('symbol'] == sym], PREFIX)
            px = last_price(sym, default=trd.get('entry'))
            log_trade(BOT, 'exit_exchange', sym, trd.get('side', ''), price=px, entry=trd.get('entry'), sl=trd.get('sl'), tp=trd.get('tp'))
            symbol_cooldowns[sym] = time.time()
            to_remove.append(sym)

    if to_remove:
        for sym in to_remove:
            active_trades.pop(sym, None)
        state_changed = True

    if state_changed or guard.dirty:
        guard.dirty = False
        push_trader_status_to_github()


def compute_contract_amount(symbol, price, target_notional=20.0):
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
        return None
    return float(lots)


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
        print(f"⚠️ Fout bij ophalen posities: {e}")
        return

    halted = guard.check()
    if halted:
        return

    active_symbols = {p.get('symbol') for p in positions if abs(float(p.get('contracts') or 0)) > 0}
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
    # Luistert UITSLUITEND naar de quant worker model_state
    state_4h = load_state(asset, '4h')
    if state_4h is None:
        return

    quant_signal = state_4h.get('signal')
    if quant_signal not in ('long', 'short'):
        return

    candle_ts = _candle_ts_seconds(state_4h)
    raw_last = float(symbol_last_traded_candles.get(symbol, 0) or 0)
    last_traded_ts = int(raw_last / 1000.0 if raw_last > 1e10 else raw_last)
    if candle_ts > 0 and candle_ts <= last_traded_ts:
        return

    if MAX_SAME_DIRECTION > 0 and same_direction_count(quant_signal) >= MAX_SAME_DIRECTION:
        return

    price = float(exchange.fetch_ticker(symbol)['last'])
    atr = float(state_4h.get('atr', 0.0))
    if atr <= 0:
        atr = price * 0.02

    is_long = (quant_signal == 'long')
    sgn = 1 if is_long else -1

    # Haalt rechtstreeks de berekende SL en TP op uit de quant worker state
    stop_loss_val = float(state_4h.get('stop_loss', price - sgn * 2.5 * atr))
    take_profit_val = float(state_4h.get('take_profit', price + sgn * 5.0 * atr))

    # Kleine limit offset (0.2 ATR) voor betere uitvoering
    limit_price = price - (0.2 * atr) if is_long else price + (0.2 * atr)

    leverage = 2  # Standaard veilige hefboom voor quant trades
    amount = compute_contract_amount(symbol, limit_price, target_notional=TRADE_AMOUNT_USDT * leverage)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    if DRY_RUN:
        print(f"🧪 [DRY RUN QUANT] {side} {amount}x {symbol} @ {limit_price} | SL: {stop_loss_val} | TP: {take_profit_val}")
        return

    try:
        exchange.set_margin_mode(DESIRED_MARGIN_MODE, symbol)
        exchange.set_leverage(leverage, symbol)
    except Exception as e:
        print(f"⚠️ Hefboom instellen mislukt: {e}")
        return

    try:
        execute_with_retry(lambda: exchange.create_order(symbol, 'limit', side, amount, limit_price, {
            'marginMode': DESIRED_MARGIN_MODE,
            'clientOrderId': f"{PREFIX}entry_{int(time.time() * 1000)}"
        }))

        print(f"🚀 QUANT TRADER SUCCES: Limit order geplaatst voor {symbol} ({side} {amount}x @ {limit_price}).")

        if candle_ts > 0:
            symbol_last_traded_candles[symbol] = candle_ts

        active_trades[symbol] = {
            'status': 'pending',
            'created_at': time.time(),
            'side': quant_signal,
            'entry': limit_price,
            'peak': limit_price,
            'atr': atr,
            'sl': stop_loss_val,
            'tp': take_profit_val,
            'breakeven_secured': False,
            'protected_contracts': None,
        }
        log_trade(BOT, 'entry_order', symbol, quant_signal, price=limit_price, amount=amount,
                  entry=limit_price, sl=stop_loss_val, tp=take_profit_val, note='quant-only signal')
        push_trader_status_to_github()
    except Exception as e:
        print(f"❌ Fout bij plaatsen quant order voor {symbol}: {e}")


if __name__ == '__main__':
    pull_status_from_github()
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Quant-Only Trader Worker gestart (luistert enkel naar model_state_*_4h.json).")
    while True:
        try:
            pull_ranking_from_github()
            current_symbols = get_combined_trading_assets()
            manage_open_positions_and_trailing_stops(current_symbols)
            execute_live_trades_scan(current_symbols)
        except Exception as e:
            print(f"⚠️ Fout in quant trader lus: {e}")
        time.sleep(60)
