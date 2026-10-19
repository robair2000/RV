import base64
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import ccxt
import pandas as pd


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/ranking':
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            
            html = "<html><head><title>V20 Trader - Live Ranking</title></head><body style='font-family: Arial; padding: 20px; background-color: #f4f4f9;'>"
            html += "<h2 style='color: #333;'>🏆 V20 Live Kucoin Coins Ranking (Trader Dashboard)</h2>"
            
            ranking_file = 'kucoin_best_coins_ranking.csv'
            if os.path.exists(ranking_file):
                try:
                    df = pd.read_csv(ranking_file)
                    html += df.to_html(index=False, classes='table', border=1)
                    html += """
                    <style>
                        table.table { border-collapse: collapse; width: 100%; background: white; margin-top: 15px; }
                        table.table th, table.table td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }
                        table.table th { background-color: #4CAF50; color: white; }
                        table.table tr:hover { background-color: #f1f1f1; }
                    </style>
                    """
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


GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

def pull_ranking_from_github():
    filename = 'kucoin_best_coins_ranking.csv'
    token = os.environ.get('GITHUB_TOKEN', '')
    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
    headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'trader-worker'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(filename + '.tmp', 'wb') as f:
                f.write(resp.read())
        os.replace(filename + '.tmp', filename)
    except Exception:
        pass


def get_combined_trading_assets(top_n=5):
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
    ranking_file = 'kucoin_best_coins_ranking.csv'
    dynamic_symbols = []
    
    if os.path.exists(ranking_file):
        try:
            df = pd.read_csv(ranking_file)
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].head(top_n).tolist()
                for s in symbols:
                    if s not in dynamic_symbols and ':USDT' in s:
                        dynamic_symbols.append(s)
        except Exception:
            pass
            
    return list(dict.fromkeys(core_symbols + dynamic_symbols))


TIMEFRAMES = ['4h', '1d']

# --- CONFIGURATIE FUTURES SWING TRADING ---
TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 20.0))
MIN_BOSS_SCORE = float(os.environ.get('MIN_BOSS_SCORE', 84.0))
COOLDOWN_SECONDS = int(os.environ.get('COOLDOWN_SECONDS', 7200))

DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

MAX_CONCURRENT_POSITIONS = 2
MIN_RRR = 3.0
TP_ATR_mult_BASE = 5.0
MAX_STATE_AGE_SECONDS = {'4h': 8 * 3600, '1d': 48 * 3600}

active_trades = {}
symbol_cooldowns = {}
symbol_last_traded_candles = {}  # Geheugen om re-entry op dezelfde 4h candle te blokkeren

exchange = ccxt.kucoinfutures({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
})


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
        'cooldowns': symbol_cooldowns
    }
    filename = 'trader_status.json'
    content_bytes = json.dumps(data, indent=2).encode('utf-8')

    with open(filename, 'wb') as f:
        f.write(content_bytes)

    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "trader-worker"
    }

    sha = None
    try:
        req = urllib.request.Request(url + f"?ref={GITHUB_BRANCH}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            sha = json.loads(resp.read().decode('utf-8')).get('sha')
    except Exception:
        pass

    payload = {
        "message": "Auto-update trader_status.json [skip ci]",
        "content": base64.b64encode(content_bytes).decode('ascii'),
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    try:
        req_data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, headers=headers, method='PUT')
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception:
        pass


def pull_latest_states_from_github(symbols):
    token = os.environ.get('GITHUB_TOKEN', '')
    files_to_pull = []
    for symbol in symbols:
        asset = symbol.split('/')[0].split(':')[0]
        for tf in TIMEFRAMES:
            files_to_pull.append(f'model_state_{asset.lower()}_{tf}.json')
            files_to_pull.append(f'pattern_state_{asset.lower()}_4h.json')

    for filename in set(files_to_pull):
        url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
        headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'trader-worker'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(filename + '.tmp', 'wb') as f:
                    f.write(resp.read())
            os.replace(filename + '.tmp', filename)
        except Exception:
            pass


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
    except Exception:
        return None
        
    updated_at = float(pst.get('updated_at', 0) or 0)
    if time.time() - updated_at > MAX_PATTERN_AGE_SECONDS:
        print(f"⚠️ Waarschuwing: Pattern state voor {asset} is te oud ({int((time.time() - updated_at)/3600)} uur). Agent lijkt inactief!")
        return None
        
    return pst


def close_position(symbol, side, contracts):
    close_side = 'sell' if side == 'long' else 'buy'
    if DRY_RUN:
        return True
    try:
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            for o in open_orders:
                if 'v20_' in str(o.get('clientOrderId', '')):
                    exchange.cancel_order(o['id'], symbol)
        except Exception:
            pass

        execute_with_retry(lambda: exchange.create_order(
            symbol, 'market', close_side, contracts, None,
            {'reduceOnly': True, 'marginMode': DESIRED_MARGIN_MODE}
        ))
        return True
    except Exception:
        return False


def update_exchange_stop_loss(symbol, side, contracts, new_sl, old_sl_order_id=None):
    if DRY_RUN:
        return None
    try:
        close_side = 'sell' if side == 'long' else 'buy'
        client_order_id = f"v20_sl_{int(time.time() * 1000)}"
        
        new_order = exchange.create_order(symbol, 'stop', close_side, contracts, None, {
            'stopPrice': new_sl,
            'reduceOnly': True,
            'clientOrderId': client_order_id
        })
        new_order_id = new_order.get('id') if new_order else None

        if old_sl_order_id:
            try:
                exchange.cancel_order(old_sl_order_id, symbol)
            except Exception:
                pass

        open_orders = exchange.fetch_open_orders(symbol)
        for o in open_orders:
            order_id = o.get('id')
            c_id = str(o.get('clientOrderId', ''))
            o_type = str(o.get('type', '')).lower()
            
            if 'take profit' in o_type or 'tp' in c_id or o_type == 'limit':
                continue
            
            if order_id != new_order_id and ('v20_sl_' in c_id or o_type == 'stop'):
                try:
                    exchange.cancel_order(order_id, symbol)
                except Exception:
                    pass
                    
        return new_order_id
    except Exception as e:
        print(f"⚠️ Kon exchange stop-loss niet bijwerken voor {symbol}: {e}")
        return old_sl_order_id


def manage_open_positions_and_trailing_stops(symbols):
    try:
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception:
        return

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
        open_symbols.add(symbol)
        entry = float(pos.get('entryPrice') or 0)

        try:
            price = float(exchange.fetch_ticker(symbol)['last'])
        except Exception:
            continue

        trade = active_trades.get(symbol)
        if trade is None:
            base = symbol.split('/')[0].split(':')[0]
            st = load_state(base, '4h')
            pst = load_pattern_state(base)
            trail_mult = float(pst.get('recommended_trailing_mult', 2.5)) if pst else 2.5

            atr = entry * float(st['atr_pct']) / 100.0 if (st and st.get('atr_pct')) else entry * 0.02
            sgn = 1 if side == 'long' else -1
            tp_mult = max(TP_ATR_mult_BASE, trail_mult * (MIN_RRR + 0.2))
            trade = active_trades[symbol] = {
                'side': side, 'entry': entry, 'peak': price, 'atr': atr,
                'sl': entry - sgn * trail_mult * atr,
                'tp': entry + sgn * tp_mult * atr,
                'trailing_mult': trail_mult,
                'breakeven_secured': False,
            }
            state_changed = True

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

        if trade['sl'] != old_sl:
            update_exchange_stop_loss(symbol, side, contracts, trade['sl'])
            
        if stop_hit or tp_hit:
            if close_position(symbol, side, contracts):
                symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)
                state_changed = True
                
    to_remove = [sym for sym in active_trades if sym not in open_symbols]
    if to_remove:
        for sym in to_remove:
            active_trades.pop(sym, None)
        state_changed = True

    if state_changed:
        push_trader_status_to_github()


def ensure_margin_and_leverage(symbol, target_leverage):
    if DRY_RUN:
        return True
    try:
        exchange.set_margin_mode(DESIRED_MARGIN_MODE, symbol)
        exchange.set_leverage(target_leverage, symbol)
        return True
    except Exception:
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
        print(f"⚠️ {symbol}: 1 contract = {per_contract_notional:.2f} USDT notional, doel {target_notional:.2f} "
              f"({TRADE_AMOUNT_USDT} margin x {leverage}x). Verhoog TRADE_AMOUNT_USDT. Overgeslagen.")
        return None

    used_margin = lots * per_contract_notional / leverage
    print(f"📐 {symbol}: {lots} contract(en), notional {lots*per_contract_notional:.2f}, marge {used_margin:.2f} USDT @ {leverage}x")
    return float(lots)


def execute_live_trades_scan(symbols):
    print(f"🔍 [Trader Worker] Start marktscan voor symbolen: {symbols}")
    current_time = time.time()
    expired_cooldowns = [sym for sym, ts in symbol_cooldowns.items() if current_time - ts > COOLDOWN_SECONDS]
    for sym in expired_cooldowns:
        symbol_cooldowns.pop(sym, None)

    pull_latest_states_from_github(symbols)
    try:
        exchange.load_markets()
        positions = execute_with_retry(lambda: exchange.fetch_positions())
        open_orders = execute_with_retry(lambda: exchange.fetch_open_orders())
    except Exception as e:
        print(f"⚠️ Fout bij ophalen posities/orders van KuCoin Futures: {e}")
        return

    active_symbols = set()
    for p in positions:
        if abs(float(p.get('contracts') or 0)) > 0:
            active_symbols.add(p.get('symbol'))

    pending_order_symbols = set()
    for o in open_orders:
        if o.get('status') == 'open':
            pending_order_symbols.add(o.get('symbol'))

    if len(active_symbols) >= MAX_CONCURRENT_POSITIONS:
        print(f"ℹ️ Maximaal aantal actieve posities bereikt ({len(active_symbols)}/{MAX_CONCURRENT_POSITIONS}).")
        return

    for symbol in symbols:
        if symbol not in exchange.markets or symbol in active_symbols or symbol in symbol_cooldowns:
            continue
        if symbol in pending_order_symbols:
            continue
        if current_time - symbol_cooldowns.get(symbol, 0) < COOLDOWN_SECONDS:
            continue
        try:
            asset = symbol.split('/')[0].split(':')[0]
            print(f"🔎 Evalueren van {symbol} (Asset: {asset})...")
            evaluate_and_trade(asset, symbol)
        except Exception as e:
            print(f"⚠️ Fout bij evalueren van {symbol}: {e}")


def evaluate_and_trade(asset, symbol):
    state_4h = load_state(asset, '4h')
    if state_4h is None:
        print(f"   ⏳ {symbol}: Geen geldige 4h state gevonden of state is te oud.")
        return

    quant_signal = state_4h['signal']
    if quant_signal not in ('long', 'short'):
        print(f"   ℹ️ {symbol}: Quant signaal is neutraal ('{quant_signal}').")
        return

    pattern_st = load_pattern_state(asset)
    if pattern_st is None:
        print(f"   ⏳ {symbol}: Geen S&R pattern state gevonden voor 4h.")
        return

    # --- CANDLE TIMESTAMP LOCK CHECK (VOORKOMT STALE RE-ENTRIES) ---
    candle_ts = int(pattern_st.get('last_candle_ts', 0))
    last_traded_ts = symbol_last_traded_candles.get(symbol, 0)
    if candle_ts > 0 and candle_ts <= last_traded_ts:
        print(f"   ⏳ {symbol}: Re-entry geblokkeerd. Dezelfde 4h candle ({candle_ts}) is reeds verhandeld.")
        return
    # -------------------------------------------------------------

    price = float(exchange.fetch_ticker(symbol)['last'])
    atr_pct = float(state_4h.get('atr_pct') or 0.0)
    if atr_pct <= 0:
        return
    atr = price * atr_pct / 100.0

    boss_signal = pattern_st.get('pattern_signal', 'none')
    boss_score = pattern_st.get('pattern_score', 50.0)
    boss_leverage = int(pattern_st.get('recommended_leverage', 2))
    boss_trailing_mult = float(pattern_st.get('recommended_trailing_mult', 2.5))
    min_score_req = pattern_st.get('min_score_required', MIN_BOSS_SCORE)

    print(f"   📊 {symbol}: Quant={quant_signal.upper()} | Boss={boss_signal.upper()} | Score={boss_score} (Min. vereist: {min_score_req})")

    if boss_signal != quant_signal or boss_score < min_score_req:
        print(f"   🚫 {symbol}: VETO - Signalen komen niet overeen of score is te laag.")
        return

    is_long = (quant_signal == 'long')

    direction_prob = float(state_4h.get('direction_probability', 50.0))
    if is_long and direction_prob < 50.0:
        print(f"   🚫 {symbol}: Quant kans op stijging te laag ({direction_prob}%).")
        return
    if not is_long and direction_prob > 50.0:
        print(f"   🚫 {symbol}: Quant kans op daling te laag ({direction_prob}%).")
        return

    if is_long:
        limit_price = price - (0.2 * atr)
    else:
        limit_price = price + (0.2 * atr)

    sgn = 1 if is_long else -1
    stop_loss_val = limit_price - sgn * boss_trailing_mult * atr
    effective_tp_mult = max(TP_ATR_mult_BASE, boss_trailing_mult * (MIN_RRR + 0.2))
    take_profit_val = limit_price + sgn * effective_tp_mult * atr

    if not ensure_margin_and_leverage(symbol, boss_leverage):
        print(f"   ⚠️ {symbol}: Kon hefboom/margin mode niet instellen.")
        return
        
    amount = compute_contract_amount(asset, symbol, limit_price, boss_leverage)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    if DRY_RUN:
        print(f"   🧪 [DRY RUN] Zou order plaatsen: {side} {amount}x {symbol} op limietprijs {limit_price}")
        return

    params = {
        'stopLoss': {'triggerPrice': stop_loss_val},
        'takeProfit': {'triggerPrice': take_profit_val},
        'marginMode': DESIRED_MARGIN_MODE,
        'clientOrderId': f"v20_entry_{int(time.time() * 1000)}"
    }
    try:
        order = execute_with_retry(lambda: exchange.create_order(symbol, 'limit', side, amount, limit_price, params))
        print(f"   🚀 SUCCES: Limit order geplaatst voor {symbol} ({side} {amount} contracten @ {limit_price})!")
        
        # Sla op dat deze 4h candle nu officieel verhandeld is
        if candle_ts > 0:
            symbol_last_traded_candles[symbol] = candle_ts

        active_trades[symbol] = {
            'side': 'long' if is_long else 'short', 
            'entry': limit_price, 'peak': limit_price, 'atr': atr,
            'sl': stop_loss_val, 'tp': take_profit_val,
            'trailing_mult': boss_trailing_mult,
            'breakeven_secured': False,
        }
        push_trader_status_to_github()
    except Exception as e:
        print(f"   ❌ Fout bij plaatsen zone-based limit order voor {symbol}: {e}")


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 V20 Trader Worker gestart (scan interval: 60 seconden).")
    while True:
        try:
            pull_ranking_from_github()
            current_symbols = get_combined_trading_assets(top_n=5)

            manage_open_positions_and_trailing_stops(current_symbols)
            execute_live_trades_scan(current_symbols)
        except Exception as e:
            print(f"⚠️ Fout in trader lus: {e}")
        time.sleep(60)
