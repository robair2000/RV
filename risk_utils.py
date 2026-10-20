"""Gedeelde risico- en administratie-helpers voor de trader workers.

- log_trade: append-only CSV van elke order/exit (basis voor latere evaluatie).
- DailyLossGuard: kill-switch op basis van MAX_DAILY_LOSS_USDT (0 = uitgeschakeld).
"""
import csv
import os
import time
from datetime import datetime, timezone

from github_utils import github_pull, github_put

# Maximaal dagverlies in USDT (account-equity, incl. ongerealiseerd). 0 = uitgeschakeld.
MAX_DAILY_LOSS_USDT = float(os.environ.get('MAX_DAILY_LOSS_USDT', 0.0))
# Sluit bij het bereiken van de limiet ook de door deze bot beheerde posities?
DAILY_LOSS_CLOSE_POSITIONS = os.environ.get('DAILY_LOSS_CLOSE_POSITIONS', 'false').lower() in ('1', 'true', 'yes')
# Geschatte kosten per kant (taker fee + slippage) voor journal-schattingen, als fractie (0.0006 = 0.06%)
EST_FEE_PER_SIDE = float(os.environ.get('EST_FEE_PER_SIDE', 0.0006))

# Elke bot krijgt standaard een eigen journal (trade_journal_v20.csv / trade_journal_twm.csv), zodat
# twee services die allebei het hele bestand naar GitHub PUT-en elkaar niet overschrijven.
# TRADE_JOURNAL_FILE blijft als expliciete override werken.
_JOURNAL_OVERRIDE = os.environ.get('TRADE_JOURNAL_FILE', '').strip()


def journal_file_for(bot):
    if _JOURNAL_OVERRIDE:
        return _JOURNAL_OVERRIDE
    safe = ''.join(c for c in str(bot).lower() if c.isalnum()) or 'bot'
    return f'trade_journal_{safe}.csv'


JOURNAL_FIELDS = [
    'ts_iso', 'bot', 'event', 'symbol', 'side', 'price', 'amount',
    'entry', 'sl', 'tp', 'est_gross_pnl_pct', 'est_fees_pct', 'est_net_pnl_pct', 'note',
]


def log_trade(bot, event, symbol, side='', price=None, amount=None, entry=None, sl=None, tp=None, note=''):
    """Schrijft één regel naar het journal van deze bot (trade_journal_<bot>.csv) en pusht het bestand naar GitHub.

    Bij 'exit*'-events wordt een geschatte PnL berekend uit entry en exit-prijs
    (gross) minus 2x EST_FEE_PER_SIDE. Funding en echte fills zitten er NIET in.
    """
    try:
        journal = journal_file_for(bot)
        # Na een herstart van een ephemeral instance eerst de bestaande historie terughalen
        if not os.path.exists(journal):
            github_pull(journal, agent='rv-journal')

        gross = fees = net = ''
        if event.startswith('exit') and entry and price and side in ('long', 'short'):
            sgn = 1 if side == 'long' else -1
            g = sgn * (float(price) - float(entry)) / float(entry)
            f = 2 * EST_FEE_PER_SIDE
            gross, fees, net = round(g * 100, 4), round(f * 100, 4), round((g - f) * 100, 4)

        new_file = not os.path.exists(journal)
        with open(journal, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                'ts_iso': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'bot': bot, 'event': event, 'symbol': symbol, 'side': side,
                'price': price, 'amount': amount, 'entry': entry, 'sl': sl, 'tp': tp,
                'est_gross_pnl_pct': gross, 'est_fees_pct': fees, 'est_net_pnl_pct': net,
                'note': note,
            })
        with open(journal, 'rb') as f:
            github_put(journal, f.read(), agent='rv-journal')
    except Exception as e:
        print(f"⚠️ Journal schrijven mislukt: {e}")


class DailyLossGuard:
    """Kill-switch op dagverlies. State is JSON-serialiseerbaar (voor trader_status*.json)."""

    def __init__(self, exchange, max_loss_usdt=None, state=None):
        self.exchange = exchange
        self.max_loss = MAX_DAILY_LOSS_USDT if max_loss_usdt is None else float(max_loss_usdt)
        self.state = dict(state) if isinstance(state, dict) else {}
        self.dirty = False

    @property
    def enabled(self):
        return self.max_loss > 0

    def _equity(self):
        try:
            bal = self.exchange.fetch_balance()
            usdt = bal.get('USDT') or {}
            total = usdt.get('total')
            if total is None:
                total = (bal.get('total') or {}).get('USDT')
            return float(total) if total is not None else None
        except Exception as e:
            print(f"⚠️ DailyLossGuard: equity ophalen mislukt: {e}")
            return None

    def check(self):
        """True = nieuwe entries geblokkeerd voor vandaag (UTC-dag)."""
        if not self.enabled:
            return False
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        eq = self._equity()
        if self.state.get('date') != today:
            if eq is None:
                return False  # dagstart onbekend; niet blokkeren, fout is al gelogd
            self.state = {'date': today, 'start_equity': eq, 'halted': False}
            self.dirty = True
            return False
        if eq is None:
            return bool(self.state.get('halted', False))
        loss = float(self.state.get('start_equity', eq)) - eq
        if loss >= self.max_loss and not self.state.get('halted', False):
            self.state['halted'] = True
            self.state['halted_at'] = time.time()
            self.state['loss_at_halt'] = round(loss, 4)
            self.dirty = True
            print(f"🛑 MAX_DAILY_LOSS bereikt: verlies {loss:.2f} USDT >= {self.max_loss:.2f}. Geen nieuwe entries tot 00:00 UTC.")
        return bool(self.state.get('halted', False))
