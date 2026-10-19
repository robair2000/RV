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


# --- RENDER WEB SERVER (HEALTH CHECK FOR BACKGROUND WORKERS) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/ranking':
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            
            html = "<html><head><title>TWM - Live Explosive Reversal Dashboard</title></head><body style='font-family: Arial; padding: 20px; background-color: #0f172a; color: #f8fafc;'>"
            html += "<h2 style='color: #38bdf8;'>🚀 TWM - Reversal Sniper (Pure S&R, Volume & Squeeze Mode)</h2>"
            
            ranking_file = 'kucoin_best_coins_ranking.csv'
            if os.path.exists(ranking_file):
                try:
                    df = pd.read_csv(ranking_file)
                    html += df.to_html(index=False, classes='table', border=1)
                    html += """
                    <style>
                        table.table { border-collapse: collapse; width: 100%; background: #1e293b; margin-top: 15px; color: #f8fafc; }
                        table.table th, table.table td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #334155; }
                        table.table th { background-color: #0284c7; color: white; }
                        table.table tr:hover { background-color: #334155; }
                    </style>
                    """
                except Exception as e:
                    html += f"<p style='color: #ef4444;'>⚠️ Fout bij lezen van CSV: {e}</p>"
            else:
                html += "<p style='color: #f59e0b;'>⏳ Nog geen ranking CSV gevonden op schijf.</p>"
                
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
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"🌐 TWM Health Check Server gestart op poort {port}")
    server.serve_forever()


GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')


def pull_ranking_from_github():
    filename = 'kucoin_best_coins_ranking.csv'
    token = os.environ.get('GITHUB_TOKEN', '')
    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
    headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'twm-sniper-worker'}
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
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'LINK/USDT:USDT', 'AVAX/USDT:USDT']
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

# --- CONFIGURATIE TWM (SNIPER SPECIFIEK) ---
TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 20.0))
MIN_BOSS_SCORE = float(os.environ.get('MIN_BOSS_SCORE', 78.0))       
MIN_VOLUME_EXPANSION = float(os.environ.get('MIN_VOLUME_EXPANSION', 1.30)) 
COOLDOWN_SECONDS = int(os.environ.get('COOLDOWN_SECONDS', 3600))  

DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

MAX_CONCURRENT_POSITIONS = 2
TP_ATR_MULT_BASE = 8.0  
TRAILING_SL_MULT = 2.5  
MAX_STATE_AGE_SECONDS = {'4h': 8 * 3600, '1d': 48 * 3600}

active_trades = {}
symbol_cooldowns = {}
symbol_last_traded_candles = {}  # Geheugen om re-entry op dezelfde 4h candle te blokkeren

# GEBRUIK STRICT KUCOIN FUTURES VOOR CORRECTE SWAP ENDPOINTS
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
    filename = 'trader_status_momentum.json'
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
        "User-Agent": "twm-sniper-worker"
    }

    sha = None
    try:
        req = urllib.request.Request(url + f"?ref={GITHUB_BRANCH}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            sha = json.loads(resp.read().decode('utf-8')).get('sha')
    except Exception:
        pass

    payload = {
        "message": "Auto-update trader_status_momentum.json [skip ci]",
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
        files_to_pull.append(f'pattern_state_{asset.lower()}_4h.json')

    for filename in set(files_to_pull):
        url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
        headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'twm-sniper-worker'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(filename + '.tmp', 'wb' ) as f:
                    f.write(resp.read())
            os.replace(filename + '.tmp', filename)
        except Exception:
            pass


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
                # Annuleer alleen eigen TWM orders bij sluiten
                if 'twm_' in str(o.get('clientOrderId', '')):
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
        client_order_id = f"twm_sl_{int(time.time() * 1000)}"  # Correcte TWM prefix
        
        # 1. Plaats direct de nieuwe Stop Loss order op de exchange
        new_order = exchange.create_order(symbol, 'stop', close_side, contracts, None, {
            'stopPrice': new_sl,
            'reduceOnly': True,
            'clientOrderId': client_order_id
        })
        new_order_id = new_order.get('id') if new_order else None

        # 2. Haal actieve open orders op om oude eigen TWM SL orders netjes op te ruimen
        open_orders = exchange.fetch_open_orders(symbol)
        for o in open_orders:
            order_id = o.get('id')
            c_id = str(o.get('clientOrderId', ''))
            o_type = str(o.get('type', '')).lower()
            
            if old_sl_order_id and order_id == old_sl_order_id:
                try:
                    exchange.cancel_order(order_id, symbol)
                except Exception:
                    pass
                continue
            
            # BLIJF AF van take profit orders en orders van andere bots
            if 'take' in o_type or 'tp' in c_id or o_type == 'limit':
                continue
            
            if order_id != new_order_id and ('twm_sl_' in c_id or o_type == 'stop'):
                try:
                    exchange.cancel_order(order_id, symbol)
                except Exception:
                    pass
                    
        return new_order_id
    except Exception as e:
        print(f"⚠️ Kon TWM exchange stop-loss niet bijwerken voor {symbol}: {e}")
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
            trail_mult = TRAILING_SL_MULT
            
            asset = symbol.split('/')[0].split(':')[0]
            pst = load_pattern_state(asset)
            atr_pct = float(pst.get('atr_pct', 2.0)) if pst else 2.0
            atr = entry * (atr_pct / 100.0)
            
            sgn = 1 if side == 'long' else -1
            tp_mult = TP_ATR_MULT_BASE
            trade = active_trades[symbol] = {
                'side': side, 'entry': entry, 'peak': price, 'atr': atr,
                'sl': entry - sgn * trail_mult * atr,
                'tp': entry + sgn * tp_mult * atr,
                'trailing_mult': trail_mult,
                'breakeven_secured': False,
            }
            state_changed = True

        atr = trade['atr']
        trail_mult = trade.get('trailing_mult', TRAILING_SL_MULT)
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
            if price > trade['peak']:
                trade['peak'] = price
                if distance_in_atr >= 2.0:
                    new_sl = price - (trail_mult * 0.8) * atr
                    if new_sl > trade['sl']:
                        trade['sl'] = new_sl
                        state_changed = True
            stop_hit = price <= trade['sl']
            tp_hit = price >= trade['tp']
        else:
            if price < trade['peak']:
                trade['peak'] = price
                if distance_in_atr >= 2.0:
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
        return None

    return float(lots)


def execute_live_trades_scan(symbols):
    print(f"🚀 [TWM Reversal Sniper] Start marktscan op: {symbols}")
    current_time = time.time()
    expired_cooldowns = [sym for sym, ts in symbol_cooldowns.items() if current_time - ts > COOLDOWN_SECONDS]
    for sym in expired_cooldowns:
        symbol_cooldowns.pop(sym, None)

    pull_latest_states_from_github(symbols)
    try:
        exchange.load_markets()
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception as e:
        print(f"⚠️ TWM Sniper: Fout bij ophalen posities: {e}")
        return

    active_symbols = set()
    for p in positions:
        if abs(float(p.get('contracts') or 0)) > 0:
            active_symbols.add(p.get('symbol'))

    if len(active_symbols) >= MAX_CONCURRENT_POSITIONS:
        print(f"ℹ️ TWM Sniper: Maximaal aantal actieve posities bereikt ({len(active_symbols)}/{MAX_CONCURRENT_POSITIONS}).")
        return

    for symbol in symbols:
        if symbol not in exchange.markets or symbol in active_symbols or symbol in symbol_cooldowns:
            continue
            
        if current_time - symbol_cooldowns.get(symbol, 0) < COOLDOWN_SECONDS:
            continue

        try:
            pending_orders = exchange.fetch_open_orders(symbol)
            if pending_orders:
                print(f"   🛡️ TWM Sniper SKIPPEN {symbol}: Er staan al openstaande orders/limieten op dit symbool.")
                continue
        except Exception:
            pass

        try:
            asset = symbol.split('/')[0].split(':')[0]
            print(f"🔎 TWM Sniper Evalueren van {symbol} (Asset: {asset})...")
            evaluate_and_trade_sniper(asset, symbol)
        except Exception as e:
            print(f"⚠️ TWM Sniper fout bij {symbol}: {e}")


def evaluate_and_trade_sniper(asset, symbol):
    pattern_st = load_pattern_state(asset)
    if pattern_st is None:
        return

    # --- CANDLE TIMESTAMP LOCK CHECK (VOORKOMT STALE RE-ENTRIES) ---
    candle_ts = int(pattern_st.get('last_candle_ts', 0))
    last_traded_ts = symbol_last_traded_candles.get(symbol, 0)
    if candle_ts > 0 and candle_ts <= last_traded_ts:
        print(f"   ⏳ {symbol}: Re-entry geblokkeerd. Dezelfde 4h candle ({candle_ts}) is reeds verhandeld.")
        return
    # -------------------------------------------------------------

    boss_signal = pattern_st.get('pattern_signal', 'none')
    boss_score = pattern_st.get('pattern_score', 50.0)
    boss_leverage = int(pattern_st.get('recommended_leverage', 2))
    
    vol_expansion = float(pattern_st.get('volume_expansion_ratio', 1.0))
    is_squeeze = bool(pattern_st.get('is_bollinger_squeeze', False))
    
    atr_pct = float(pattern_st.get('atr_pct', 2.0))

    print(f"   🎯 TWM Sniper Check {symbol}: Signal={boss_signal.upper()} | Score={boss_score} | VolRatio={vol_expansion} | Squeeze={is_squeeze} | ATR%={atr_pct}")

    if boss_signal not in ('long', 'short') or boss_score < MIN_BOSS_SCORE:
        return

    if vol_expansion < MIN_VOLUME_EXPANSION and not is_squeeze:
        print(f"   🚫 TWM Sniper VETO {symbol}: Geen volume-piek ({vol_expansion}) en geen Bollinger Squeeze.")
        return

    try:
        price = float(exchange.fetch_ticker(symbol)['last'])
    except Exception:
        return

    atr = price * (atr_pct / 100.0)
    
    is_long = (boss_signal == 'long')
    sgn = 1 if is_long else -1
    stop_loss_val = price - sgn * TRAILING_SL_MULT * atr
    take_profit_val = price + sgn * TP_ATR_MULT_BASE * atr

    if not ensure_margin_and_leverage(symbol, boss_leverage):
        return
        
    amount = compute_contract_amount(symbol, price, boss_leverage)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    if DRY_RUN:
        print(f"   🧪 [DRY RUN] Zou order plaatsen: {side} {amount}x {symbol} @ {price} (SL: {stop_loss_val:.2f}, TP: {take_profit_val:.2f})")
        return

    params = {
        'stopLoss': {'triggerPrice': stop_loss_val},
        'takeProfit': {'triggerPrice': take_profit_val},
        'marginMode': DESIRED_MARGIN_MODE,
        'clientOrderId': f"twm_entry_{int(time.time() * 1000)}"
    }
    try:
        order = execute_with_retry(lambda: exchange.create_order(symbol, 'market', side, amount, None, params))
        print(f"   🚀 TWM SNIPER RAAK: Order geplaatst voor {symbol} ({side} {amount} contracten @ {price})!")
        
        # Sla op dat deze 4h candle nu officieel verhandeld is
        if candle_ts > 0:
            symbol_last_traded_candles[symbol] = candle_ts

        active_trades[symbol] = {
            'side': 'long' if is_long else 'short', 
            'entry': price, 'peak': price, 'atr': atr,
            'sl': stop_loss_val, 'tp': take_profit_val,
            'trailing_mult': TRAILING_SL_MULT,
            'breakeven_secured': False,
        }
        push_trader_status_to_github()
    except Exception as e:
        print(f"   ❌ TWM orderfout voor {symbol}: {e}")


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    print("🚀 TWM (Reversal Sniper Mode) gestart op Render.")
    while True:
        try:
            pull_ranking_from_github()
            current_symbols = get_combined_trading_assets(top_n=5)
            manage_open_positions_and_trailing_stops(current_symbols)
            execute_live_trades_scan(current_symbols)
        except Exception as e:
            print(f"⚠️ TWM sniper lus fout: {e}")
        time.sleep(60)
