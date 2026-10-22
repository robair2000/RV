import os
import json
import math
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import ccxt
import pandas as pd

from github_utils import github_pull, github_put
from asset_utils import get_combined_trading_assets
from order_utils import (fetch_all_open_orders, cancel_orders, create_stop_loss,
                         create_take_profit, replace_stop_loss, replace_protective_orders,
                         describe_sizing)
from risk_utils import DailyLossGuard, log_trade, DAILY_LOSS_CLOSE_POSITIONS

BOT = 'SQUEEZE_TRADER'
PREFIX = 'sqz_'
AGENT = 'squeeze-trader-worker'


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/ranking':
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            
            html = "<html><head><title>Squeeze Breakout Sniper</title></head><body style='font-family: Arial; padding: 20px; background-color: #0f172a; color: #f8fafc;'>"
            html += "<h2 style='color: #38bdf8;'>🚀 Squeeze & Breakout Sniper (Live Trader - Gehard)</h2>"
            
            ranking_file = 'kucoin_best_coins_ranking.csv'
            if os.path.exists(ranking_file):
                try:
                    df = pd.read_csv(ranking_file)
                    html += df.to_html(index=False, classes='table', border=1)
                except Exception as e:
                    html += f"<p>⚠️ Ranking onleesbaar: {e}</p>"
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

def pull_ranking_from_github():
    github_pull('kucoin_best_coins_ranking.csv', min_interval=STATE_PULL_INTERVAL, agent=AGENT)

def pull_latest_states_from_github(symbols):
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        github_pull(f'pattern_state_{asset.lower()}_4h.json', min_interval=STATE_PULL_INTERVAL, agent=AGENT)


TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 20.0))
MIN_SCORE = float(os.environ.get('MIN_BOSS_SCORE', 75.0))
COOLDOWN_SECONDS = int(os.environ.get('COOLDOWN_SECONDS', 3600))

DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

MAX_CONCURRENT_POSITIONS = int(os.environ.get('MAX_CONCURRENT_POSITIONS', 5))
SL_ATR_MULT = float(os.environ.get('SL_ATR_MULT', 1.5))      # stop-loss afstand in ATR
TP_ATR_MULT = float(os.environ.get('TP_ATR_MULT', 3.0))      # take-profit afstand in ATR
BREAKEVEN_ATR = float(os.environ.get('BREAKEVEN_ATR', 1.5))  # SL naar entry zodra de koers zoveel ATR in winst staat

MAX_STATE_AGE_SECONDS = 8 * 3600
MAX_CANDLE_AGE_SECONDS = float(os.environ.get('MAX_CANDLE_AGE_HOURS', 5.5)) * 3600

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


def _norm_ts(raw):
    raw = float(raw or 0)
    return int(raw / 1000.0 if raw > 1e10 else raw)


def pull_status_from_github():
    global active_trades, symbol_cooldowns, symbol_last_traded_candles
    if not github_pull('trader_status_squeeze.json', timeout=10, agent=AGENT):
        print("ℹ️ Geen eerdere Squeeze trader status gevonden op GitHub, starten met schone lei.")
        return
    try:
        with open('trader_status_squeeze.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        active_trades = data.get('active_trades', {})
        symbol_cooldowns = data.get('cooldowns', {})
        symbol_last_traded_candles = {k: _norm_ts(v) for k, v in (data.get('last_traded_candles', {}) or {}).items()}
        guard.state = data.get('daily_guard', {}) or {}
        print(f"📥 Status hersteld van GitHub: {len(active_trades)} actieve trades.")
    except Exception as e:
        print(f"⚠️ Status-bestand onleesbaar ({e}), starten met schone lei.")


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
    filename = 'trader_status_squeeze.json'
    content_bytes = json.dumps(data, indent=2).encode('utf-8')

    with open(filename, 'wb') as f:
        f.write(content_bytes)

    if os.environ.get('GITHUB_TOKEN'):
        github_put(filename, content_bytes, message="Auto-update trader_status_squeeze.json [skip ci]", agent=AGENT)


_LOGGED_ONCE = set()


def log_once(key, msg):
    """Print een melding maar één keer per key (de hoofdlus draait elke 60s)."""
    if key in _LOGGED_ONCE:
        return
    _LOGGED_ONCE.add(key)
    if len(_LOGGED_ONCE) > 5000:
        _LOGGED_ONCE.clear()
    print(msg)


def load_pattern_state(asset):
    path = f'pattern_state_{asset.lower()}_4h.json'
    if not os.path.exists(path):
        log_once(('missing', asset), f"⏭️ {asset}: geen {path} aanwezig (nog niet gepusht door de agent?)")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            pst = json.load(f)
    except Exception as e:
        log_once(('unreadable', asset), f"⏭️ {asset}: {path} onleesbaar: {e}")
        return None

    updated_at = float(pst.get('updated_at', 0) or 0)
    candle_ts = _norm_ts(pst.get('last_candle_ts', 0))
    current_time = time.time()

    if candle_ts <= 0 or current_time - updated_at > MAX_STATE_AGE_SECONDS:
        log_once(('stale_state', asset, updated_at),
                 f"⏭️ {asset}: state te oud (updated {(current_time - updated_at) / 3600:.1f}h geleden, max {MAX_STATE_AGE_SECONDS / 3600:.1f}h)")
        return None

    if current_time - candle_ts > MAX_CANDLE_AGE_SECONDS:
        log_once(('stale_candle', asset, candle_ts),
                 f"⏭️ {asset}: candle te oud ({(current_time - candle_ts) / 3600:.1f}h, max {MAX_CANDLE_AGE_SECONDS / 3600:.1f}h), "
                 f"signal={pst.get('pattern_signal')} genegeerd")
        return None

    return pst


def last_price(symbol, default=None):
    try:
        return float(exchange.fetch_ticker(symbol)['last'])
    except Exception:
        return default


def close_position(symbol, side, contracts):
    if side not in ('long', 'short'):
        return False
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


def manage_open_positions_and_trailing_stops(symbols):
    try:
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception as e:
        print(f"⚠️ TWM: posities ophalen mislukt (API-hickup overgeslagen): {e}")
        return

    try:
        open_orders = fetch_all_open_orders(exchange)
    except Exception:
        open_orders = None

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
            continue

        open_symbols.add(symbol)
        entry = float(pos.get('entryPrice') or 0) or float(trade['entry'])

        if not DRY_RUN and trade.get('protected_contracts') != contracts:
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

        if distance_in_atr >= BREAKEVEN_ATR and not trade.get('breakeven_secured', False):
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

        if trade['sl'] != old_sl and not DRY_RUN:
            replace_stop_loss(exchange, symbol, side, contracts, trade['sl'], PREFIX)

        if stop_hit or tp_hit:
            if close_position(symbol, side, contracts):
                reason = 'tp' if tp_hit else 'stop'
                log_trade(BOT, f'exit_{reason}', symbol, side, price=price, amount=contracts,
                          entry=entry, sl=trade['sl'], tp=trade['tp'], note='breakout exit')
                symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)
                state_changed = True

    to_remove = [sym for sym in active_trades if sym not in open_symbols]
    for sym in to_remove:
        trd = active_trades[sym]
        if not DRY_RUN and open_orders is not None:
            cancel_orders(exchange, [o for o in open_orders if o.get('symbol') == sym], PREFIX)
        px = last_price(sym, default=trd.get('entry'))
        log_trade(BOT, 'exit_exchange', sym, trd.get('side', ''), price=px, entry=trd.get('entry'),
                  sl=trd.get('sl'), tp=trd.get('tp'), note='positie gesloten op exchange')
        symbol_cooldowns[sym] = time.time()
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
        print(f"⚠️ Margin/leverage instellen mislukt: {e}")
        return False


def contract_notional(symbol, price):
    try:
        return price * float(exchange.market(symbol).get('contractSize') or 1.0)
    except Exception:
        return price


def compute_contract_amount(symbol, price, leverage):
    """TRADE_AMOUNT_USDT = eigen inleg (margin). Positiegrootte = inleg x leverage.
    Er wordt naar beneden afgerond op hele contracten, dus de echte inleg is <= TRADE_AMOUNT_USDT."""
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
        log_once(('sizing', symbol, leverage),
                 f"⚠️ {symbol}: positie te klein -> overgeslagen. Doel-notional {target_notional:.2f} USDT "
                 f"({TRADE_AMOUNT_USDT} x {leverage}x) < 1 contract ({per_contract_notional:.2f} USDT). "
                 f"Verhoog TRADE_AMOUNT_USDT.")
        return None
    return float(lots)


def flatten_bot_positions(positions):
    if guard.state.get('flattened'):
        return
    print("🛑 Dagverlies-limiet bereikt -> posities sluiten.")
    held = {p.get('symbol'): p for p in positions if abs(float(p.get('contracts') or 0)) > 0}
    for sym in list(active_trades.keys()):
        trd = active_trades[sym]
        p = held.get(sym)
        if p is not None and p.get('side') in ('long', 'short'):
            c = abs(float(p.get('contracts') or 0))
            px = last_price(sym, default=trd.get('entry'))
            if close_position(sym, p['side'], c):
                log_trade(BOT, 'exit_daily_loss', sym, p['side'], price=px, amount=c, entry=trd.get('entry'), note='MAX_DAILY_LOSS_USDT')
                symbol_cooldowns[sym] = time.time()
                active_trades.pop(sym, None)
    guard.state['flattened'] = True
    guard.dirty = True


def execute_live_trades_scan(symbols):
    current_time = time.time()
    expired = [sym for sym, ts in symbol_cooldowns.items() if current_time - ts > COOLDOWN_SECONDS]
    for sym in expired:
        symbol_cooldowns.pop(sym, None)
    if expired:
        push_trader_status_to_github()

    pull_latest_states_from_github(symbols)

    try:
        exchange.load_markets()
        positions = execute_with_retry(lambda: exchange.fetch_positions())
        open_orders = fetch_all_open_orders(exchange)
    except Exception as e:
        print(f"⚠️ Scan overgeslagen: markten/posities/orders ophalen mislukt: {e}")
        return

    halted = guard.check()
    if halted and DAILY_LOSS_CLOSE_POSITIONS:
        flatten_bot_positions(positions)
    if guard.dirty:
        guard.dirty = False
        push_trader_status_to_github()
    if halted:
        log_once(('halted', guard.state.get('date')),
                 f"🛑 Nieuwe entries geblokkeerd door MAX_DAILY_LOSS_USDT (dag {guard.state.get('date')}).")
        return

    active_symbols = {p.get('symbol') for p in positions if abs(float(p.get('contracts') or 0)) > 0}
    pending_symbols = {o.get('symbol') for o in open_orders if not o.get('_stop')}

    if len([s for s in active_symbols if s in active_trades]) >= MAX_CONCURRENT_POSITIONS:
        return

    for symbol in symbols:
        if symbol not in exchange.markets or symbol in symbol_cooldowns:
            continue
        if symbol in active_symbols or symbol in pending_symbols:
            continue
        asset = symbol.split('/')[0].split(':')[0]
        try:
            evaluate_and_trade_breakout(asset, symbol)
        except Exception as e:
            print(f"⚠️ Fout bij evalueren van {symbol}: {e}")


def evaluate_and_trade_breakout(asset, symbol):
    pst = load_pattern_state(asset)
    if pst is None:
        return

    candle_ts = _norm_ts(pst.get('last_candle_ts', 0))
    if candle_ts > 0 and candle_ts <= symbol_last_traded_candles.get(symbol, 0):
        return

    signal = pst.get('pattern_signal', 'none')
    score = float(pst.get('pattern_score', 50.0))

    if signal not in ('long', 'short') or score < MIN_SCORE:
        return

    price = last_price(symbol)
    if price is None:
        return

    atr_pct = float(pst.get('atr_pct', 2.0))
    atr = price * (atr_pct / 100.0)
    is_long = (signal == 'long')
    sgn = 1 if is_long else -1
    position_side = 'long' if is_long else 'short'

    risk = SL_ATR_MULT * atr
    reward = TP_ATR_MULT * atr

    stop_loss_val = price - sgn * risk
    take_profit_val = price + sgn * reward
    
    # Haalt de door de agent berekende leverage op (inclusief Markov risk multiplier)
    leverage = int(pst.get('recommended_leverage', 2))
    risk_mult = float(pst.get('risk_multiplier', 1.0))

    amount = compute_contract_amount(symbol, price, leverage)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    notional = amount * contract_notional(symbol, price)
    margin = notional / max(leverage, 1)
    sizing_txt = f"inleg {margin:.2f} USDT (max {TRADE_AMOUNT_USDT}) x {leverage}x = positie {notional:.2f} USDT"
    if DRY_RUN:
        print(f"🧪 [DRY RUN SQUEEZE] {side} {amount} contracten {symbol} | score {score} | {sizing_txt} (risk_mult={risk_mult}) | SL {stop_loss_val:.4f} | TP {take_profit_val:.4f}")
        return

    if not ensure_margin_and_leverage(symbol, leverage):
        return

    try:
        execute_with_retry(lambda: exchange.create_order(symbol, 'market', side, amount, None, {
            'marginMode': DESIRED_MARGIN_MODE,
            'clientOrderId': f"{PREFIX}entry_{int(time.time() * 1000)}"
        }))
    except Exception as e:
        print(f"⚠️ Entry-fout {symbol}: {e}")
        return

    close_side = 'sell' if is_long else 'buy'

    try:
        create_stop_loss(exchange, symbol, close_side, amount, stop_loss_val, PREFIX)
    except Exception as sl_err:
        print(f"🚨 KRITISCHE FOUT: SL mislukt voor {symbol} ({sl_err}). Positie sluiten.")
        close_position(symbol, position_side, amount)
        return

    try:
        create_take_profit(exchange, symbol, close_side, amount, take_profit_val, PREFIX)
    except Exception as tp_err:
        print(f"⚠️ Waarschuwing: TP plaatsen mislukt: {tp_err}")

    if candle_ts > 0:
        symbol_last_traded_candles[symbol] = candle_ts

    active_trades[symbol] = {
        'side': position_side, 'entry': price, 'peak': price, 'atr': atr,
        'sl': stop_loss_val, 'tp': take_profit_val, 'breakeven_secured': False, 'protected_contracts': amount,
    }
    log_trade(BOT, 'entry_filled', symbol, position_side, price=price, amount=amount,
              entry=price, sl=stop_loss_val, tp=take_profit_val,
              note=f'score={score} lev={leverage} margin={margin:.2f} notional={notional:.2f} risk_mult={risk_mult} breakout')
    push_trader_status_to_github()
    print(f"🚀 SQUEEZE TRADER SUCCES: {symbol} ({side} {amount} contracten), score {score}, {sizing_txt}, SL={stop_loss_val:.4f}, TP={take_profit_val:.4f}")


if __name__ == '__main__':
    pull_status_from_github()
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 Squeeze Breakout Trader gestart.")

    # Eenmalige sizing-check: welke symbols kunnen met TRADE_AMOUNT_USDT überhaupt 1 contract openen?
    try:
        pull_ranking_from_github()
        exchange.load_markets()
        print(f"📏 Sizing-check (TRADE_AMOUNT_USDT={TRADE_AMOUNT_USDT}, laagste leverage 1x / typisch 3x):")
        for lev in (1, 3):
            for sym, per_contract, ok in describe_sizing(exchange, get_combined_trading_assets(), TRADE_AMOUNT_USDT, lev):
                if per_contract is None:
                    print(f"   {lev}x {sym}: onbekend (market/ticker niet op te halen)")
                else:
                    print(f"   {lev}x {sym}: 1 contract = {per_contract:.2f} USDT -> {'OK' if ok else 'TE KLEIN, wordt nooit getraded'}")
    except Exception as e:
        print(f"⚠️ Sizing-check mislukt: {e}")

    while True:
        try:
            pull_ranking_from_github()
            current_symbols = get_combined_trading_assets()
            manage_open_positions_and_trailing_stops(current_symbols)
            execute_live_trades_scan(current_symbols)
        except Exception as e:
            print(f"⚠️ Lus fout: {e}")
        time.sleep(60)
