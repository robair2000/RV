import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt
import pandas as pd

from github_utils import github_pull, github_put
from order_utils import (cancel_orders, create_stop_loss, create_take_profit, fetch_all_open_orders,
                         replace_protective_orders, replace_stop_loss)
from risk_utils import DAILY_LOSS_CLOSE_POSITIONS, DailyLossGuard, log_trade

BOT = 'TWM'
PREFIX = 'twm_'
AGENT = 'twm-sniper-worker'


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/ranking':
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            html = "<html><head><title>TWM - Live Mean Reversion Sniper</title></head><body style='font-family: Arial; padding: 20px; background-color: #0f172a; color: #f8fafc;'>"
            html += "<h2 style='color: #38bdf8;'>🚀 TWM - Exhaustion Reversal Sniper (Geharde Versie)</h2>"

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


def get_combined_trading_assets(top_n=5):
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'LINK/USDT:USDT', 'AVAX/USDT:USDT']
    ranking_file = 'kucoin_best_coins_ranking.csv'
    dynamic_symbols = []

    if os.path.exists(ranking_file):
        try:
            df = pd.read_csv(ranking_file)
            if 'Symbol' in df.columns:
                for s in df['Symbol'].head(top_n).tolist():
                    if s not in dynamic_symbols and ':USDT' in s:
                        dynamic_symbols.append(s)
        except Exception as e:
            print(f"⚠️ TWM: ranking CSV onleesbaar: {e}")

    return list(dict.fromkeys(core_symbols + dynamic_symbols))


TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 20.0))
# Eigen variabele voor TWM, zodat de V20-drempel (MIN_BOSS_SCORE) in een gedeelde env-groep TWM niet stilzet
MIN_TWM_SCORE = float(os.environ.get('TWM_MIN_SCORE', 75.0))
COOLDOWN_SECONDS = int(os.environ.get('COOLDOWN_SECONDS', 3600))

DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

MAX_CONCURRENT_POSITIONS = int(os.environ.get('MAX_CONCURRENT_POSITIONS', 5))
TRAILING_SL_MULT = 2.0
# Target = terug naar het gemiddelde (SMA20). Skip als dat te dichtbij of de RRR te laag is.
TWM_MIN_TP_ATR = float(os.environ.get('TWM_MIN_TP_ATR', 1.0))
TWM_MIN_RRR = float(os.environ.get('TWM_MIN_RRR', 1.0))
MAX_STATE_AGE_SECONDS = 8 * 3600
MAX_CANDLE_AGE_SECONDS = 8.75 * 3600  # 4h bar + 4h buffer + marge

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
    if not github_pull('trader_status_momentum.json', timeout=10, agent=AGENT):
        print("ℹ️ Geen eerdere TWM status gevonden op GitHub, starten met schone lei.")
        return
    try:
        with open('trader_status_momentum.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        active_trades = data.get('active_trades', {})
        symbol_cooldowns = data.get('cooldowns', {})
        symbol_last_traded_candles = {k: _norm_ts(v) for k, v in (data.get('last_traded_candles', {}) or {}).items()}
        guard.state = data.get('daily_guard', {}) or {}
        print(f"📥 TWM Status hersteld van GitHub: {len(active_trades)} actieve trades.")
    except Exception as e:
        print(f"⚠️ TWM status-bestand onleesbaar ({e}), starten met schone lei.")


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
    filename = 'trader_status_momentum.json'
    content_bytes = json.dumps(data, indent=2).encode('utf-8')

    with open(filename, 'wb') as f:
        f.write(content_bytes)

    if os.environ.get('GITHUB_TOKEN'):
        if not github_put(filename, content_bytes, message="Auto-update trader_status_momentum.json [skip ci]", agent=AGENT):
            print("⚠️ trader_status_momentum.json lokaal bijgewerkt maar NIET naar GitHub gepusht.")


def load_pattern_state(asset):
    path = f'pattern_state_{asset.lower()}_4h.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            pst = json.load(f)
    except Exception as e:
        print(f"⚠️ TWM: pattern state {path} onleesbaar: {e}")
        return None

    updated_at = float(pst.get('updated_at', 0) or 0)
    raw_candle_ts = float(pst.get('last_candle_ts', 0) or 0)
    candle_ts = _norm_ts(raw_candle_ts)  # <--- HIER DE FIX: netjes omzetten naar seconden!
    current_time = time.time()

    if current_time - candle_ts > MAX_CANDLE_AGE_SECONDS:
        print(f"⚠️ TWM Waarsuchwing: Pattern state voor {asset} heeft te oude candle.")
        return None

    if raw_candle_ts > 0:
        candle_age_seconds = current_time - _norm_ts(raw_candle_ts)
        if candle_age_seconds > MAX_CANDLE_AGE_SECONDS:
            print(f"⚠️ TWM Waarschuwing: Pattern state voor {asset} heeft te oude candle ({int(candle_age_seconds / 3600)} uur oud).")
            return None

    return pst


def last_price(symbol, default=None):
    try:
        return float(exchange.fetch_ticker(symbol)['last'])
    except Exception:
        return default


def close_position(symbol, side, contracts):
    """side moet 'long' of 'short' zijn (de richting van de POSITIE, niet van de order)."""
    if side not in ('long', 'short'):
        print(f"🚨 TWM close_position: ongeldige side '{side}' voor {symbol}")
        return False
    close_side = 'sell' if side == 'long' else 'buy'
    if DRY_RUN:
        return True
    try:
        try:
            cancel_orders(exchange, fetch_all_open_orders(exchange, symbol), PREFIX)
        except Exception as e:
            print(f"⚠️ TWM: openstaande orders annuleren voor {symbol} mislukt: {e}")

        execute_with_retry(lambda: exchange.create_order(
            symbol, 'market', close_side, contracts, None,
            {'reduceOnly': True, 'marginMode': DESIRED_MARGIN_MODE}
        ))
        return True
    except Exception as e:
        print(f"⚠️ TWM sluitfout voor {symbol}: {e}")
        return False


def manage_open_positions_and_trailing_stops(symbols):
    try:
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception as e:
        print(f"⚠️ TWM: posities ophalen mislukt: {e}")
        return

    try:
        open_orders = fetch_all_open_orders(exchange)
    except Exception as e:
        print(f"⚠️ TWM: open orders ophalen mislukt: {e}")
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

        # Deelvulling / afwijkende positiegrootte: SL/TP afstemmen op de ACTUELE positie
        if not DRY_RUN and trade.get('protected_contracts') != contracts:
            if replace_protective_orders(exchange, symbol, side, contracts, trade['sl'], trade['tp'], PREFIX):
                trade['protected_contracts'] = contracts
                state_changed = True
                print(f"🛡️ TWM: SL/TP voor {symbol} gezet op {contracts} contracten.")
            else:
                print(f"🚨 TWM: positie {symbol} heeft GEEN stop-loss (plaatsen mislukt); nieuwe poging volgende ronde.")

        price = last_price(symbol)
        if price is None:
            continue

        atr = trade['atr']
        sgn = 1 if side == 'long' else -1
        distance_in_atr = sgn * (price - entry) / atr if atr > 0 else 0.0
        old_sl = trade['sl']

        if distance_in_atr >= 1.5 and not trade.get('breakeven_secured', False):
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
            stop_hit = price <= trade['sl']
            tp_hit = price >= trade['tp']
        else:
            if price < trade['peak']:
                trade['peak'] = price
            stop_hit = price >= trade['sl']
            tp_hit = price <= trade['tp']

        if trade['sl'] != old_sl and not DRY_RUN:
            replace_stop_loss(exchange, symbol, side, contracts, trade['sl'], PREFIX)

        if stop_hit or tp_hit:
            if close_position(symbol, side, contracts):
                reason = 'tp' if tp_hit else 'stop'
                log_trade(BOT, f'exit_{reason}', symbol, side, price=price, amount=contracts,
                          entry=entry, sl=trade['sl'], tp=trade['tp'], note='software exit')
                symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)
                state_changed = True

    # Positie niet meer open: gesloten op exchange (SL/TP). Restorders opruimen + cooldown + journal.
    to_remove = [sym for sym in active_trades if sym not in open_symbols]
    for sym in to_remove:
        trd = active_trades[sym]
        if not DRY_RUN and open_orders is not None:
            cancel_orders(exchange, [o for o in open_orders if o.get('symbol') == sym], PREFIX)
        px = last_price(sym, default=trd.get('entry'))
        log_trade(BOT, 'exit_exchange', sym, trd.get('side', ''), price=px, entry=trd.get('entry'),
                  sl=trd.get('sl'), tp=trd.get('tp'), note='positie gesloten op exchange (prijs = laatste koers, indicatief)')
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
        print(f"⚠️ TWM: margin/leverage instellen voor {symbol} mislukt: {e}")
        return False


def compute_contract_amount(symbol, price, leverage):
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
        print(f"ℹ️ TWM: {symbol} overgeslagen, 1 contract (${per_contract_notional:,.2f}) > notional ${target_notional:,.2f}.")
        return None
    return float(lots)


def flatten_bot_positions(positions):
    """Dagverlies-limiet bereikt: bot-posities sluiten (eenmalig per dag)."""
    if guard.state.get('flattened'):
        return
    print("🛑 TWM: dagverlies-limiet bereikt -> bot-posities sluiten.")
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
        print(f"⚠️ TWM: posities/orders ophalen mislukt: {e}")
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
    pending_symbols = {o.get('symbol') for o in open_orders if not o.get('_stop')}

    my_active_count = len([s for s in active_symbols if s in active_trades])
    if my_active_count >= MAX_CONCURRENT_POSITIONS:
        return

    for symbol in symbols:
        if symbol not in exchange.markets or symbol in symbol_cooldowns:
            continue
        if symbol in active_symbols or symbol in pending_symbols:
            continue
        asset = symbol.split('/')[0].split(':')[0]
        try:
            evaluate_and_trade_reversal(asset, symbol)
        except Exception as e:
            print(f"⚠️ TWM: evalueren van {symbol} mislukt: {e}")


def evaluate_and_trade_reversal(asset, symbol):
    pst = load_pattern_state(asset)
    if pst is None:
        return

    candle_ts = _norm_ts(pst.get('last_candle_ts', 0))
    if candle_ts > 0 and candle_ts <= symbol_last_traded_candles.get(symbol, 0):
        return

    twm_signal = pst.get('twm_signal', 'none')
    twm_score = float(pst.get('twm_score', 50.0))

    # twm_score is nu gegradeerd (65..100), dus deze drempel filtert echt
    if twm_signal not in ('long', 'short') or twm_score < MIN_TWM_SCORE:
        return

    adx_4h = float(pst.get('adx_4h', 20.0))
    if adx_4h > 28.0:
        return

    price = last_price(symbol)
    if price is None:
        return

    atr_pct = float(pst.get('atr_pct', 2.0))
    atr = price * (atr_pct / 100.0)
    is_long = (twm_signal == 'long')
    sgn = 1 if is_long else -1
    position_side = 'long' if is_long else 'short'

    # Target = SMA20 (mean reversion). Geen SMA of te weinig ruimte/RRR -> geen trade.
    sma20 = pst.get('sma20')
    if sma20 is None:
        print(f"ℹ️ TWM: {symbol} overgeslagen, geen sma20 in pattern state.")
        return
    reward = sgn * (float(sma20) - price)
    risk = TRAILING_SL_MULT * atr
    if reward < TWM_MIN_TP_ATR * atr or reward / risk < TWM_MIN_RRR:
        print(f"ℹ️ TWM: {symbol} overgeslagen, SMA20-target te dichtbij (reward {reward / atr:.2f} ATR, RRR {reward / risk:.2f}).")
        return

    stop_loss_val = price - sgn * risk
    take_profit_val = float(sma20)
    leverage = 2

    if not ensure_margin_and_leverage(symbol, leverage):
        return
    amount = compute_contract_amount(symbol, price, leverage)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    if DRY_RUN:
        print(f"🧪 [DRY RUN TWM] Zou order plaatsen: {side} {amount}x {symbol} | SL {stop_loss_val:.4f} | TP(SMA20) {take_profit_val:.4f}")
        return

    try:
        execute_with_retry(lambda: exchange.create_order(symbol, 'market', side, amount, None, {
            'marginMode': DESIRED_MARGIN_MODE,
            'clientOrderId': f"{PREFIX}entry_{int(time.time() * 1000)}"
        }))
    except Exception as e:
        print(f"⚠️ TWM entry-fout {symbol}: {e}")
        return

    close_side = 'sell' if is_long else 'buy'

    # Beveiliging tegen naked positions: als de SL mislukt, positie direct sluiten.
    # FIX (bug 2): close_position verwacht 'long'/'short' (positie-richting), niet 'buy'/'sell'.
    try:
        create_stop_loss(exchange, symbol, close_side, amount, stop_loss_val, PREFIX)
    except Exception as sl_err:
        print(f"🚨 KRITISCHE FOUT: SL mislukt voor {symbol} na marktentry ({sl_err}). Positie wordt direct gesloten.")
        if not close_position(symbol, position_side, amount):
            print(f"🚨🚨 {symbol}: positie sluiten MISLUKT - HANDMATIG CONTROLEREN, positie is onbeschermd!")
        log_trade(BOT, 'exit_sl_failed', symbol, position_side, price=price, amount=amount, entry=price,
                  note=f'SL plaatsen mislukt: {sl_err}')
        return

    try:
        create_take_profit(exchange, symbol, close_side, amount, take_profit_val, PREFIX)
    except Exception as tp_err:
        print(f"⚠️ Waarschuwing: TP plaatsen mislukt voor {symbol}: {tp_err} (software-TP blijft actief)")

    if candle_ts > 0:
        symbol_last_traded_candles[symbol] = candle_ts

    active_trades[symbol] = {
        'side': position_side, 'entry': price, 'peak': price, 'atr': atr,
        'sl': stop_loss_val, 'tp': take_profit_val, 'trailing_mult': TRAILING_SL_MULT,
        'breakeven_secured': False, 'protected_contracts': amount,
    }
    log_trade(BOT, 'entry_filled', symbol, position_side, price=price, amount=amount,
              entry=price, sl=stop_loss_val, tp=take_profit_val, note=f'score={twm_score} target=SMA20')
    push_trader_status_to_github()
    print(f"🚀 TWM SUCCES: Positie + Stop-Loss geplaatst voor {symbol} ({side} {amount}), TP = SMA20 {take_profit_val:.4f}")


if __name__ == '__main__':
    pull_status_from_github()
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 TWM Mean-Reversion Sniper gestart.")
    if guard.enabled:
        print(f"🛡️ MAX_DAILY_LOSS_USDT actief: {guard.max_loss} USDT (sluit posities bij limiet: {DAILY_LOSS_CLOSE_POSITIONS}).")
    else:
        print("⚠️ MAX_DAILY_LOSS_USDT is 0 -> kill-switch UIT. Zet de env-variabele om dagverlies te begrenzen.")
    while True:
        try:
            pull_ranking_from_github()
            current_symbols = get_combined_trading_assets(top_n=5)
            manage_open_positions_and_trailing_stops(current_symbols)
            execute_live_trades_scan(current_symbols)
        except Exception as e:
            print(f"⚠️ TWM lus fout: {e}")
        time.sleep(60)
