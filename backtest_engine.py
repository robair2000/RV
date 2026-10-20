"""Backtest-engine voor V20 en TWM. Geen Streamlit-afhankelijkheid (testbaar).

Hergebruikt de LIVE logica uit agent.py (compute_primary_setups, event_features, meta-labeling)
en bootst de live trader-workers na:

V20 (trader_worker.py)
- Setup op sluiting van 4h-bar t; limit-entry 0.2 ATR gunstiger dan de close, geldig voor precies
  1 bar (bar t+1), gevuld als de prijs het limiet raakt (gap door het limiet = fill op de open).
- SL = trail_mult x ATR (2.5), TP = max(5, trail_mult x 3.2) x ATR (8.0).
- Break-even + trailing zodra de piek >= 2.4 ATR in winst staat: SL = piek - 0.8 x trail_mult x ATR.
- Optioneel: meta-gate met per-fold (walk-forward, purged) modellen, ook voor niet-gedeclusterde setups.

TWM (trader_worker_momentum.py)
- Fade van %B-extremen met RSI en volume; entry op de OPEN van bar t+1 (markt).
- SL = sl_mult x ATR (2.0), TP = SMA20, skip als reward < min_tp_atr x ATR of RRR < min_rrr.
- Break-even bij 1.5 ATR winst, geen trailing.

Intrabar-volgorde is pessimistisch: raken SL en TP in dezelfde bar, dan telt de SL. Een stop die
door de open heen gapt wordt op de open uitgevoerd. Kosten = 2 x fee_per_side (fractie van notional).
Funding, liquidatie in cross-margin en quant-gate zijn NIET gemodelleerd.
"""
import math

import numpy as np
import pandas as pd

TF4H_MS = 14_400_000


# ---------------------------------------------------------------------------
# Generieke trade-walker
# ---------------------------------------------------------------------------
def walk_trade(sgn, j0, entry, sl, tp, atr0, hi, lo, op, cl, be_atr, trail_dist=None):
    """Loopt bar voor bar vanaf j0 (entry-bar). Geeft (exit_idx, exit_px, reason)."""
    n = len(cl)
    sl0 = sl
    peak = entry
    moved = False
    for j in range(j0, n):
        if sgn == 1:
            if lo[j] <= sl:
                px = sl if j == j0 else min(sl, op[j])
                return j, px, ('trail_be' if moved else 'stop')
            if hi[j] >= tp:
                return j, tp, 'tp'
            peak = max(peak, hi[j])
            if (peak - entry) / atr0 >= be_atr:
                new_sl = max(sl, entry)
                if trail_dist is not None:
                    new_sl = max(new_sl, peak - trail_dist)
                if new_sl > sl:
                    sl, moved = new_sl, True
        else:
            if hi[j] >= sl:
                px = sl if j == j0 else max(sl, op[j])
                return j, px, ('trail_be' if moved else 'stop')
            if lo[j] <= tp:
                return j, tp, 'tp'
            peak = min(peak, lo[j])
            if (entry - peak) / atr0 >= be_atr:
                new_sl = min(sl, entry)
                if trail_dist is not None:
                    new_sl = min(new_sl, peak + trail_dist)
                if new_sl < sl:
                    sl, moved = new_sl, True
    return n - 1, cl[n - 1], 'open_end'


def _trade_row(asset, bot, sgn, t_sig, j0, ex, entry, exit_px, sl0, tp0, atr0, reason, ts, fee,
               margin, lev, extra=None):
    gross = sgn * (exit_px - entry) / entry
    net = gross - 2.0 * fee
    risk_pct = abs(entry - sl0) / entry
    row = {
        'asset': asset, 'bot': bot, 'side': 'long' if sgn == 1 else 'short',
        'signal_ts': int(ts[t_sig]), 'entry_ts': int(ts[j0]), 'exit_ts': int(ts[ex]) + TF4H_MS,
        'entry': entry, 'exit': exit_px, 'sl0': sl0, 'tp0': tp0, 'reason': reason,
        'gross_pct': gross * 100.0, 'net_pct': net * 100.0,
        'r_mult': net / risk_pct if risk_pct > 0 else np.nan,
        'bars': int(ex - j0 + 1), 'leverage': lev,
        'pnl_usd': margin * lev * net,
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# V20
# ---------------------------------------------------------------------------
def simulate_v20(asset, d4, f, p, gate=None, thr=None):
    """p: dict met trail_mult, tp_mult, entry_offset, be_atr, trail_factor, fee, margin, leverage,
    min_score, base_score (A.MIN_BOSS_SCORE). gate: {(asset, richting, ts): prob} of None (geen gate)."""
    hi, lo = d4['high'].values, d4['low'].values
    op, cl = d4['open'].values, d4['close'].values
    ts = d4['timestamp'].values
    atr = f['atr'].values
    sl_long, sl_short = f['setup_long'].values, f['setup_short'].values
    n = len(d4)
    trades = []
    busy_until = -1
    for t in range(n - 1):
        if t < busy_until:
            continue
        a = atr[t]
        if not (a > 0):
            continue
        if sl_long[t]:
            sgn = 1
        elif sl_short[t]:
            sgn = -1
        else:
            continue

        lev, prob, score = p['leverage'], None, None
        if gate is not None:
            prob = gate.get((asset, 'long' if sgn == 1 else 'short', int(ts[t])))
            if prob is None or thr is None or prob < thr:
                continue
            score = min(100.0, p['base_score'] + 200.0 * (prob - thr))
            if score < p['min_score']:
                continue
            lev = 3 if score >= 88.0 else (2 if score >= 78.0 else 1)

        limit = cl[t] - sgn * p['entry_offset'] * a
        j0 = t + 1
        if sgn == 1:
            if lo[j0] > limit:
                continue  # niet gevuld, order verloopt na 1 bar
            entry = min(limit, op[j0])
        else:
            if hi[j0] < limit:
                continue
            entry = max(limit, op[j0])

        sl0 = entry - sgn * p['trail_mult'] * a
        tp0 = entry + sgn * p['tp_mult'] * a
        ex, px, reason = walk_trade(sgn, j0, entry, sl0, tp0, a, hi, lo, op, cl, p['be_atr'],
                                    trail_dist=p['trail_factor'] * p['trail_mult'] * a)
        trades.append(_trade_row(asset, 'V20', sgn, t, j0, ex, entry, px, sl0, tp0, a, reason, ts,
                                 p['fee'], p['margin'], lev, {'meta_prob': prob, 'score': score}))
        busy_until = ex
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# TWM
# ---------------------------------------------------------------------------
def simulate_twm(asset, d4, f, p, mode='fade'):
    """p: sl_mult, be_atr, min_tp_atr, min_rrr, min_score, max_adx, fee, margin, leverage,
    cont_rr (alleen voor mode='continuation')."""
    hi, lo = d4['high'].values, d4['low'].values
    op, cl = d4['open'].values, d4['close'].values
    ts = d4['timestamp'].values
    atr, sma, adx = f['atr'].values, f['sma20'].values, f['adx'].values
    score = f['twm_score'].values
    tl, tsr = f['twm_long'].values, f['twm_short'].values
    n = len(d4)
    trades = []
    busy_until = -1
    for t in range(n - 1):
        if t < busy_until:
            continue
        a = atr[t]
        if not (a > 0) or np.isnan(sma[t]):
            continue
        if tl[t]:
            fade = 1
        elif tsr[t]:
            fade = -1
        else:
            continue
        if score[t] < p['min_score'] or not (adx[t] <= p['max_adx']):
            continue

        j0 = t + 1
        entry = op[j0]
        risk = p['sl_mult'] * a
        reward = fade * (sma[t] - entry)
        if reward < p['min_tp_atr'] * a or reward / risk < p['min_rrr']:
            continue

        if mode == 'fade':
            sgn = fade
            sl0 = entry - sgn * risk
            tp0 = float(sma[t])
            trail = None
            be = p['be_atr']
        else:  # continuation-test: zelfde signalen, omgekeerde richting, vaste RR
            sgn = -fade
            sl0 = entry - sgn * risk
            tp0 = entry + sgn * risk * p['cont_rr']
            trail = None
            be = 1e9  # geen break-even in de diagnostiek
        ex, px, reason = walk_trade(sgn, j0, entry, sl0, tp0, a, hi, lo, op, cl, be, trail_dist=trail)
        trades.append(_trade_row(asset, 'TWM-' + mode, sgn, t, j0, ex, entry, px, sl0, tp0, a, reason, ts,
                                 p['fee'], p['margin'], p['leverage'], {'twm_score': float(score[t])}))
        busy_until = ex
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Statistiek
# ---------------------------------------------------------------------------
def summarize(tr, capital):
    """Samenvatting van een trades-DataFrame. Equity = gerealiseerd (op exit-tijd)."""
    if tr is None or tr.empty:
        return None
    tr = tr.sort_values('exit_ts').reset_index(drop=True)
    n = len(tr)
    eq = capital + np.cumsum(tr['pnl_usd'].values)
    curve = np.concatenate([[capital], eq])
    peak = np.maximum.accumulate(curve)
    dd = float(((curve - peak) / peak).min() * 100.0)

    wins = tr['pnl_usd'] > 0
    gp = float(tr.loc[wins, 'pnl_usd'].sum())
    gl = float(-tr.loc[~wins, 'pnl_usd'].sum())
    mean = float(tr['net_pct'].mean())
    sd = float(tr['net_pct'].std(ddof=1)) if n > 1 else float('nan')
    tstat = mean / (sd / math.sqrt(n)) if (n > 1 and sd and sd > 0) else float('nan')

    ev = [(r, 1) for r in tr['entry_ts']] + [(r, -1) for r in tr['exit_ts']]
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = peak_conc = 0
    for _, d in ev:
        cur += d
        peak_conc = max(peak_conc, cur)

    return {
        'n': n, 'win_rate': float(wins.mean() * 100.0),
        'avg_net_pct': mean, 'avg_r': float(tr['r_mult'].mean()),
        'profit_factor': (gp / gl) if gl > 0 else float('inf'),
        'total_pnl': float(tr['pnl_usd'].sum()), 'final_equity': float(eq[-1]),
        'max_dd_pct': dd, 'tstat': tstat, 'peak_concurrent': peak_conc,
        'avg_bars': float(tr['bars'].mean()),
        'open_end': int((tr['reason'] == 'open_end').sum()),
        'equity': pd.DataFrame({'exit_ts': pd.to_datetime(tr['exit_ts'], unit='ms'), 'equity': eq}),
        'trades': tr,
    }


# ---------------------------------------------------------------------------
# Meta-gate: walk-forward waarschijnlijkheden voor ALLE setups (niet alleen gedeclusterde events)
# ---------------------------------------------------------------------------
def walk_forward_gate(A, histories):
    """A = agent-module. Per fold een model getraind op events met ts_end < cut_ts (purged), daarna
    P(TP) voorspeld voor elke setup-bar in dat fold-venster. Return (probs, info)."""
    info = {'ok': False, 'reason': 'geen events', 'n_events': 0, 'n_pos': 0, 'auc': None, 'z': None,
            'threshold': None, 'p_be': None, 'sel_n': 0, 'sel_mean_r': None, 'base_rate': None}
    events = A.build_meta_events(histories)
    if events is None or events.empty:
        return {}, info

    pack = A.train_meta_model(events)
    info = {k: pack.get(k) for k in info}
    thr = pack.get('threshold')
    if thr is None:
        return {}, info

    ev = events.sort_values('ts').reset_index(drop=True)
    n = len(ev)
    cuts = [int(n * q) for q in A.META_FOLD_CUTS] + [n]

    cand = []
    for asset, (d4, d1) in histories.items():
        f = A.compute_primary_setups(d4, d1)
        ts = d4['timestamp'].values
        atr = f['atr'].values
        for direction, sgn, col in (('long', 1, 'setup_long'), ('short', -1, 'setup_short')):
            idx = np.where(f[col].values & ~np.isnan(atr))[0]
            if len(idx) == 0:
                continue
            ft = A.event_features(f, idx.tolist(), sgn)
            ft['asset'] = asset
            ft['direction'] = direction
            ft['ts'] = ts[idx].astype('int64')
            cand.append(ft)
    if not cand:
        return {}, info
    cand = pd.concat(cand, ignore_index=True)

    probs = {}
    for k in range(len(cuts) - 1):
        test_ev = ev.iloc[cuts[k]:cuts[k + 1]]
        if test_ev.empty:
            continue
        cut_ts = int(test_ev['ts'].iloc[0])
        end_ts = int(ev['ts'].iloc[cuts[k + 1]]) if cuts[k + 1] < n else None
        train = ev[ev['ts_end'] < cut_ts]
        if len(train) < 60 or train['label'].nunique() < 2:
            continue
        m = A.make_meta_classifier()
        m.fit(train[A.META_FEATURES], train['label'])
        sel = cand['ts'] >= cut_ts
        if end_ts is not None:
            sel &= cand['ts'] < end_ts
        sub = cand[sel]
        if sub.empty:
            continue
        pr = m.predict_proba(sub[A.META_FEATURES])[:, 1]
        for a, d, t, pp in zip(sub['asset'], sub['direction'], sub['ts'], pr):
            probs[(a, d, int(t))] = float(pp)
    return probs, info
