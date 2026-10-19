"""Gedeelde order-helpers voor V20 en TWM (KuCoin Futures via ccxt).

Belangrijk:
- Untriggered stop-orders staan bij KuCoin Futures in een aparte lijst. Daarom wordt
  `fetch_open_orders` altijd ook met params={'stop': True} aangeroepen.
- SL is een 'market' order met `stopLossPrice` (ccxt zet de trigger-richting zelf).
  Type 'stop' bestaat niet in de KuCoin-API.
"""
import time


def _cid(order):
    cid = order.get('clientOrderId')
    if not cid:
        cid = (order.get('info') or {}).get('clientOid')
    return str(cid or '')


def fetch_all_open_orders(exchange, symbol=None):
    """Gewone open orders (fout = exception, zodat de caller niet 'geen orders' aanneemt)
    plus untriggered stop-orders (fout wordt gelogd en genegeerd). Stops krijgen `_stop=True`."""
    orders = list(exchange.fetch_open_orders(symbol))
    try:
        for o in exchange.fetch_open_orders(symbol, params={'stop': True}):
            o['_stop'] = True
            orders.append(o)
    except Exception as e:
        print(f"⚠️ Kon untriggered stop-orders niet ophalen ({symbol or 'alle'}): {e}")
    seen, unique = set(), []
    for o in orders:
        key = (o.get('id'), bool(o.get('_stop')))
        if key in seen:
            continue
        seen.add(key)
        unique.append(o)
    return unique


def cancel_orders(exchange, orders, prefix, keep_id=None):
    """Annuleert alle orders waarvan clientOrderId met `prefix` begint. Return aantal geannuleerd."""
    n = 0
    for o in orders:
        if not _cid(o).startswith(prefix):
            continue
        if keep_id is not None and o.get('id') == keep_id:
            continue
        try:
            params = {'stop': True} if o.get('_stop') else {}
            exchange.cancel_order(o['id'], o.get('symbol'), params)
            n += 1
        except Exception as e:
            print(f"⚠️ Annuleren van order {o.get('id')} ({_cid(o)}) mislukt: {e}")
    return n


def create_stop_loss(exchange, symbol, close_side, contracts, stop_price, prefix):
    cid = f"{prefix}sl_{int(time.time() * 1000)}"
    order = exchange.create_order(symbol, 'market', close_side, contracts, None, {
        'stopLossPrice': stop_price,
        'reduceOnly': True,
        'clientOrderId': cid,
        'marginMode': 'cross',  # Dwingt cross af voor de stop-order
    })
    return order.get('id') if order else None


def create_take_profit(exchange, symbol, close_side, contracts, tp_price, prefix):
    cid = f"{prefix}tp_{int(time.time() * 1000)}"
    order = exchange.create_order(symbol, 'limit', close_side, contracts, tp_price, {
        'reduceOnly': True,
        'clientOrderId': cid,
        'marginMode': 'cross',  # Dwingt cross af voor de take-profit
    })
    return order.get('id') if order else None


def replace_stop_loss(exchange, symbol, side, contracts, new_sl, prefix):
    """Nieuwe SL eerst plaatsen, dán pas de oude(n) annuleren (nooit een moment zonder stop).
    Return nieuwe order-id of None bij falen (oude SL blijft dan staan)."""
    close_side = 'sell' if side == 'long' else 'buy'
    try:
        new_id = create_stop_loss(exchange, symbol, close_side, contracts, new_sl, prefix)
    except Exception as e:
        print(f"⚠️ {prefix}: nieuwe stop-loss plaatsen voor {symbol} mislukt: {e}")
        return None
    try:
        cancel_orders(exchange, fetch_all_open_orders(exchange, symbol), f"{prefix}sl_", keep_id=new_id)
    except Exception as e:
        print(f"⚠️ {prefix}: oude stop-losses opruimen voor {symbol} mislukt: {e}")
    return new_id


def replace_protective_orders(exchange, symbol, side, contracts, sl, tp, prefix):
    """(Her)plaatst SL en TP op de volledige actuele positiegrootte `contracts`.
    Return True als de SL staat (zonder SL is de positie naakt). TP-falen wordt gelogd; de bot
    sluit software-matig op TP."""
    close_side = 'sell' if side == 'long' else 'buy'
    sl_id = replace_stop_loss(exchange, symbol, side, contracts, sl, prefix)
    if sl_id is None:
        return False
    if tp is not None:
        try:
            # Oude TP eerst weg: reduce-only limit orders kunnen elkaar blokkeren
            cancel_orders(exchange, fetch_all_open_orders(exchange, symbol), f"{prefix}tp_")
            create_take_profit(exchange, symbol, close_side, contracts, tp, prefix)
        except Exception as e:
            print(f"⚠️ {prefix}: take-profit plaatsen voor {symbol} mislukt (software-TP blijft actief): {e}")
    return True
