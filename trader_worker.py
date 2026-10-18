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
    """Haalt de meest recente ranking CSV op van GitHub zodat alle workers synchroon lopen."""
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
        print("📥 Verse ranking CSV succesvol opgehaald van GitHub voor trader.")
    except Exception as e:
        print(f"⚠️ Kon ranking CSV niet ophalen van GitHub voor trader ({e}), we gebruiken lokale versie.")


# --- CONFIGURATIE FUTURES SWING TRADING (V20 ZONE-BASED LIMITS) ---
def get_combined_trading_assets(top_n=5):
    """
    Combineert altijd de vaste kern met de geldige top N futures uit de ranking CSV.
    """
    core_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'LINK/USDT:USDT']
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
            print(f"📈 Dynamische top {top_n} ingeladen voor trader: {dynamic_symbols}")
        except Exception as e:
            print(f"⚠️ Fout bij lezen van ranking CSV in trader: {e}")
            
    combined = list(dict.fromkeys(core_symbols + dynamic_symbols))
    print(f"🎯 Actieve trader trading scope ({len(combined)} symbols): {combined}")
    return combined


TIMEFRAMES = ['4h', '1d']
TRADE_CHECK_INTERVAL_SECONDS = 3600
POSITION_CHECK_INTERVAL_SECONDS = 120

TRADE_AMOUNT_USDT = float(os.environ.get('TRADE_AMOUNT_USDT', 7.7))

ASSET_MARGIN_WEIGHTS = {
    'BTC': 1.5,
    'LINK': 1.5,
    'ETH': 1.0,
    'SOL': 1.0,
    'AVAX': 1.0
}

DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() in ('1', 'true', 'yes')
EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'kucoin')

MAX_CONCURRENT_POSITIONS = 2
MIN_RRR = 3.0
COOLDOWN_SECONDS = 7200
TP_ATR_mult_BASE = 5.0

MIN_BOSS_SCORE = float(os.environ.get('MIN_BOSS_SCORE', 84.0))
MAX_STATE_AGE_SECONDS = {'4h': 8 * 3600, '1d': 48 * 3600}

active_trades = {}
symbol_cooldowns = {}

exchange = getattr(ccxt, EXCHANGE_ID)({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
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


def load_pattern_state(asset):
    path = f'pattern_state_{asset.lower()}_4h.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def close_position(symbol, side, contracts):
    close_side = 'sell' if side == 'long' else 'buy'
    if DRY_RUN:
        return True
    try:
        execute_with_retry(lambda: exchange.create_order(
            symbol, 'market', close_side, contracts, None,
            {'reduceOnly': True, 'marginMode': DESIRED_MARGIN_MODE}
        ))
        return True
    except Exception:
        return False


def pnl_fraction(side, entry, price):
    if not entry:
        return 0.0
    move = (price - entry) / entry
    return move if side == 'long' else -move


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
            
        if stop_hit or tp_hit:
            if close_position(symbol, side, contracts):
                if pnl_fraction(side, entry, price) < 0:
                    symbol_cooldowns[symbol] = time.time()
                active_trades.pop(symbol, None)
                state_changed = True

    if state_changed or len(active_trades) > 0:
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
    weight = ASSET_MARGIN_WEIGHTS.get(asset, 1.0)
    target_margin = TRADE_AMOUNT_USDT * weight
    target = target_margin * leverage
    
    try:
        contract_size = float(exchange.market(symbol).get('contractSize') or 1.0)
    except Exception:
        return None
    per_contract = price * contract_size
    if per_contract <= 0:
        return None
    amount = math.floor(target / per_contract)
    return float(max(1, amount))


def execute_live_trades_scan(symbols):
    pull_latest_states_from_github(symbols)
    try:
        exchange.load_markets()
        positions = execute_with_retry(lambda: exchange.fetch_positions())
    except Exception:
        return

    active_symbols = set()
    for p in positions:
        if abs(float(p.get('contracts') or 0)) > 0:
            active_symbols.add(p.get('symbol'))

    if len(active_symbols) >= MAX_CONCURRENT_POSITIONS:
        return

    for symbol in symbols:
        if symbol not in exchange.markets or symbol in active_symbols or symbol in symbol_cooldowns:
            continue
        if time.time() - symbol_cooldowns.get(symbol, 0) < COOLDOWN_SECONDS:
            continue
        try:
            asset = symbol.split('/')[0].split(':')[0]
            evaluate_and_trade(asset, symbol)
        except Exception:
            pass


def evaluate_and_trade(asset, symbol):
    state_4h = load_state(asset, '4h')
    if state_4h is None:
        return

    signal = state_4h['signal']
    if signal == 'none':
        return
    is_long = signal == 'long'

    price = float(exchange.fetch_ticker(symbol)['last'])
    atr_pct = float(state_4h.get('atr_pct') or 0.0)
    if atr_pct <= 0:
        return
    atr = price * atr_pct / 100.0

    pattern_st = load_pattern_state(asset)
    if pattern_st:
        boss_signal = pattern_st.get('pattern_signal', 'none')
        boss_score = pattern_st.get('pattern_score', 50.0)
        boss_leverage = int(pattern_st.get('recommended_leverage', 2))
        boss_trailing_mult = float(pattern_st.get('recommended_trailing_mult', 2.5))
        
        if boss_signal != signal or boss_score < MIN_BOSS_SCORE:
            return
    else:
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
        return
        
    amount = compute_contract_amount(asset, symbol, limit_price, boss_leverage)
    if amount is None:
        return

    side = 'buy' if is_long else 'sell'
    if DRY_RUN:
        return

    params = {
        'stopLoss': {'triggerPrice': stop_loss_val},
        'takeProfit': {'triggerPrice': take_profit_val},
        'marginMode': DESIRED_MARGIN_MODE,
    }
    try:
        order = execute_with_retry(lambda: exchange.create_order(symbol, 'limit', side, amount, limit_price, params))
        active_trades[symbol] = {
            'side': 'long' if is_long else 'short', 
            'entry': limit_price, 'peak': limit_price, 'atr': atr,
            'sl': stop_loss_val, 'tp': take_profit_val,
            'trailing_mult': boss_trailing_mult,
            'breakeven_secured': False,
        }
        push_trader_status_to_github()
    except Exception as e:
        print(f"Fout bij plaatsen zone-based limit order voor {symbol}: {e}")


if __name__ == '__main__':
    threading.Thread(target=run_server, daemon=True).start()
    while True:
        try:
            pull_ranking_from_github()
            current_symbols = get_combined_trading_assets(top_n=5)

            manage_open_positions_and_trailing_stops(current_symbols)
            execute_live_trades_scan(current_symbols)
        except Exception as e:
            print(f"Fout in trader lus: {e}")
        time.sleep(60)
