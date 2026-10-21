"""
Backtest-engine voor het V20-systeem (zonder Streamlit-afhankelijkheid, dus los te testen).

Twee ketens:
  1. Agent + momentum-trader  : squeeze-setup uit agent.py -> market-entry zoals trader_worker_momentum.py
  2. Quant + quant-trader     : walk-forward modellen uit quant_worker.py -> limit-entry zoals trader_worker (quant-only)

Principes tegen lookahead:
  * Een signaal wordt genomen op de CLOSE van candle t; uitvoering begint op candle t+1.
  * Quant-modellen worden alleen getraind op data t/m candle t (met embargo, net als in quant_worker).
  * Als SL en TP in dezelfde candle geraakt worden, telt de SL (conservatief).
  * In de candle waarin een limit-order gevuld wordt, geldt alleen de SL-check (conservatief).
Niet gemodelleerd: funding-noodrem (geen historie), orderboek/liquiditeit, lot-afronding, API-fouten.
"""
import math
import os
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

H4 = 4 * 3600 * 1000
DAY = 24 * 3600 * 1000

CORE_SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
EXTRA_SYMBOLS = ['XRP/USDT:USDT', 'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'LTC/USDT:USDT', 'DOT/USDT:USDT',
                 'BNB/USDT:USDT', 'TRX/USDT:USDT']


# ============================================================== DATA
def fetch_ohlcv_range(exchange, symbol, bars, cache_dir='bt_cache', tf='4h'):
    """Haalt `bars` gesloten candles op (gepagineerd) en cachet ze als CSV. De lopende candle wordt weggelaten."""
    os.makedirs(cache_dir, exist_ok=True)
    step = H4 if tf == '4h' else DAY
    path = os.path.join(cache_dir, f"{symbol.split('/')[0]}_{tf}.csv")
    now = int(time.time() * 1000)
    if os.path.exists(path):
        old = pd.read_csv(path)
        if len(old) >= bars and now - int(old['timestamp'].iloc[-1]) < 2 * step:
            return old.tail(bars).reset_index(drop=True)
    since, rows, guard = now - bars * step, [], 0
    while since < now and guard < 500:
        guard += 1
        batch = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=200)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= since:
            break
        since = nxt
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df[df['timestamp'] + step <= now].reset_index(drop=True)
    df.to_csv(path, index=False)
    return df.tail(bars).reset_index(drop=True)


def load_kucoin_frames(symbols, bars, cache_dir='bt_cache'):
    import ccxt
    ex = ccxt.kucoinfutures({'enableRateLimit': True})
    ex.load_markets()
    frames, problems = {}, []
    for s in symbols:
        if s not in ex.markets:
            problems.append(f"{s}: niet beschikbaar op KuCoin Futures")
            continue
        try:
            frames[s] = fetch_ohlcv_range(ex, s, bars, cache_dir)
        except Exception as e:  # noqa
            problems.append(f"{s}: {e}")
    return frames, problems


def synthetic_frames(symbols, bars, seed=7):
    """Willekeurige koersen met volatiliteitsclusters en een marktfactor. Heeft GEEN edge: alleen om de code te testen."""
    rng = np.random.default_rng(seed)
    t0 = (int(time.time() * 1000) // H4) * H4 - bars * H4
    ts = t0 + np.arange(bars) * H4

    def vol_path(base):
        x, out = 0.0, np.empty(bars)
        for i in range(bars):
            x = 0.97 * x + rng.normal(0, 0.25)
            out[i] = base * math.exp(x)
        return out

    market = rng.normal(0, 1, bars) * vol_path(0.006)
    frames = {}
    for s in symbols:
        v = vol_path(0.008)
        ret = 0.6 * market + rng.normal(0, 1, bars) * v
        c = 100 * np.exp(np.cumsum(ret))
        o = np.r_[c[0], c[:-1]]
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.3, bars)) * v)
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.3, bars)) * v)
        vol = rng.lognormal(10, 0.3, bars) * (1 + 40 * np.abs(ret))
        frames[s] = pd.DataFrame({'timestamp': ts, 'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': vol})
    return frames


def true_range(df):
    pc = df['close'].shift(1)
    return pd.concat([df['high'] - df['low'], (df['high'] - pc).abs(), (df['low'] - pc).abs()], axis=1).max(axis=1)


def to_arrays(df):
    atr = true_range(df).rolling(14).mean().values
    return {'ts': df['timestamp'].values.astype('int64'), 'o': df['open'].values.astype(float),
            'h': df['high'].values.astype(float), 'l': df['low'].values.astype(float),
            'c': df['close'].values.astype(float), 'atr': atr.astype(float)}


def with_fixed_atr(a, pct):
    b = dict(a)
    b['atr'] = a['c'] * pct / 100.0
    return b


# ============================================================== AGENT (agent.py, huidige versie)
def agent_setup_masks(d4, p):
    """Vectoriseerde port van compute_indicators/compute_primary_setups uit de aangepaste agent.py.
    Origineel: squeeze (bb_width-percentiel <= 0.35 over 100 candles) + volume >= 1.3x; richting = close > sma20."""
    close, vol = d4['close'], d4['volume']
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bbw = (4 * std20) / sma20
    pctl = bbw.rolling(100).rank(pct=True)
    vol_ratio = vol / (vol.rolling(20).mean() + 1e-8)
    atr = true_range(d4).rolling(14).mean()
    valid = (pctl.notna() & sma20.notna() & atr.notna() & vol_ratio.notna()).values
    squeeze = (pctl <= p['squeeze_pctl']).values & (vol_ratio >= p['min_vol']).values & valid
    if p.get('direction', 'sma') == 'sma':
        up = (close > sma20).values
        long_m, short_m = squeeze & up, squeeze & ~up
    else:  # oude agent: uitbraak aan de bandrand
        lo, hi = sma20 - 2 * std20, sma20 + 2 * std20
        pct_b = ((close - lo) / ((hi - lo) + 1e-8)).values
        long_m = squeeze & (pct_b > 0.92) & (close > sma20).values
        short_m = squeeze & (pct_b < 0.08) & (close < sma20).values
    first = int(np.argmax(valid)) + 50 if valid.any() else len(d4)  # agent: range(50, ...) na dropna
    long_m[:first] = False
    short_m[:first] = False
    return long_m, short_m, first


def agent_signals(d4, p):
    """Geeft [(t, side)] terug (side +1 long / -1 short) en een funnel.
    mode 'corrected': alleen een setup op de laatste candle geeft een signaal.
    mode 'as_is'    : zoals agent.py nu werkt: de LAATSTE setup uit de historie blijft elke ronde als signaal staan
                      (latest_signal = setups.iloc[-1]), met `lag` candles vertraging door de dubbele candle-afkap."""
    long_m, short_m, first = agent_setup_masks(d4, p)
    n = len(d4)
    side = np.where(long_m, 1, np.where(short_m, -1, 0))
    sigs = []
    if p.get('mode', 'corrected') == 'corrected':
        for t in np.where(side != 0)[0]:
            if t < n - 1:
                sigs.append((int(t), int(side[t])))
    else:
        lag = int(p.get('lag', 2))
        last_idx = np.maximum.accumulate(np.where(side != 0, np.arange(n), -1))
        for t in range(first + lag, n - 1):
            j = last_idx[t - lag]
            if j >= 0:
                sigs.append((t, int(side[j])))
    funnel = {'candles': n, 'setups_long': int(long_m.sum()), 'setups_short': int(short_m.sum()),
              'signalen': len(sigs)}
    return sigs, funnel, first


# ============================================================== QUANT (quant_worker.py v8)
BASE_FEATURES = ['return_1', 'return_3', 'RSI', 'atr_ratio', 'bb_position', 'vol_ratio']
MODEL_NAMES = ['xgb', 'cat', 'rf']
LEARNING_RATE = 0.03
LABEL_HORIZON = 3
EMBARGO = 3
OOS_FRACTION = 0.50
WF_FOLDS = 4
MIN_FOLD_SIZE = 25
MIN_TRAIN_ROWS = 120
MIN_FEATURE_ROWS = 100
MIN_LOOKBACK = 300
QUANT_LOOKBACK = 1200
QUANT_FETCH_LIMIT = 1500


def model_backend():
    try:
        import xgboost  # noqa
        import catboost  # noqa
        return 'xgboost + catboost + random forest (zoals live)'
    except Exception:
        return 'sklearn-vervangers (xgboost/catboost ontbreken: resultaten wijken licht af van live)'


def make_regressors():
    from sklearn.ensemble import RandomForestRegressor
    try:
        import xgboost as xgb
        from catboost import CatBoostRegressor
        return {
            'xgb': xgb.XGBRegressor(learning_rate=LEARNING_RATE, max_depth=3, n_estimators=40, random_state=42,
                                    verbosity=0, n_jobs=1),
            'cat': CatBoostRegressor(learning_rate=LEARNING_RATE, depth=3, iterations=40, random_seed=42,
                                     verbose=0, thread_count=1),
            'rf': RandomForestRegressor(n_estimators=25, max_depth=3, random_state=42, n_jobs=1),
        }
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
        return {
            'xgb': GradientBoostingRegressor(learning_rate=LEARNING_RATE, max_depth=3, n_estimators=40, random_state=42),
            'cat': HistGradientBoostingRegressor(learning_rate=LEARNING_RATE, max_depth=3, max_iter=40, random_state=42),
            'rf': RandomForestRegressor(n_estimators=25, max_depth=3, random_state=42, n_jobs=1),
        }


def make_classifier():
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(learning_rate=LEARNING_RATE, n_estimators=25, max_depth=2, random_state=42,
                                 eval_metric='logloss', verbosity=0, n_jobs=1)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(learning_rate=LEARNING_RATE, n_estimators=25, max_depth=2, random_state=42)


def compute_features(df):
    df = df.copy()
    close, high, low = df['close'], df['high'], df['low']
    sma20 = close.rolling(20).mean()
    df['return_1'] = close.pct_change()
    df['return_3'] = close.pct_change(3)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    atr = true_range(df).rolling(14).mean()
    df['atr_ratio'] = (atr / (close + 1e-8)) * 100.0
    bb_std = close.rolling(20).std()
    bb_up, bb_lo = sma20 + 2 * bb_std, sma20 - 2 * bb_std
    df['bb_position'] = (close - bb_lo) / ((bb_up - bb_lo) + 1e-8)
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    df['ATR'] = atr
    return df.replace([np.inf, -np.inf], np.nan)


def usable_features(df, feats):
    return [f for f in feats if f in df.columns and df[f].notna().sum() >= MIN_FEATURE_ROWS]


def build_dataset(df, feats, lookback):
    d = df.dropna(subset=feats + ['close', 'ATR']).reset_index(drop=True)
    lb = min(int(lookback), len(d) - LABEL_HORIZON - 10)
    if lb < MIN_LOOKBACK:
        return None
    X = d[feats].iloc[:-LABEL_HORIZON]
    y = (d['close'].shift(-LABEL_HORIZON) / d['close'] - 1.0).iloc[:-LABEL_HORIZON]
    return X.iloc[-lb:], y.iloc[-lb:]


def z_vs_half(acc, n_obs):
    n_eff = n_obs / float(LABEL_HORIZON)
    if acc is None or n_eff <= 0:
        return None
    return float((acc - 0.5) / math.sqrt(0.25 / n_eff))


def walk_forward_eval(X, y):
    n = len(X)
    oos_n = int(n * OOS_FRACTION)
    fold = oos_n // WF_FOLDS
    if fold < MIN_FOLD_SIZE or (n - oos_n - EMBARGO) < MIN_TRAIN_ROWS:
        return None
    starts = [n - oos_n + k * fold for k in range(WF_FOLDS)]
    ends = starts[1:] + [n]
    preds = {name: [] for name in MODEL_NAMES}
    ys, ps = [], []
    for s, e in zip(starts, ends):
        tr_end = s - EMBARGO
        Xtr, ytr, Xte, yte = X.iloc[:tr_end], y.iloc[:tr_end], X.iloc[s:e], y.iloc[s:e]
        for name, model in make_regressors().items():
            model.fit(Xtr, ytr)
            preds[name].append(np.asarray(model.predict(Xte), dtype=float))
        ytr_c = (ytr > 0).astype(int)
        if ytr_c.nunique() == 2:
            clf = make_classifier()
            clf.fit(Xtr, ytr_c)
            ps.append(clf.predict_proba(Xte)[:, 1])
        else:
            ps.append(np.full(len(Xte), np.nan))
        ys.append(np.asarray(yte, dtype=float))
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    maes = {k: float(np.mean(np.abs(v - y_all))) for k, v in preds.items()}
    inv = {k: 1.0 / (v + 1e-9) for k, v in maes.items()}
    tot = sum(inv.values())
    weights = {k: float(v / tot) for k, v in inv.items()}
    ens = sum(weights[k] * preds[k] for k in preds)
    skill = float(np.mean(np.abs(ens - y_all))) / (float(np.mean(np.abs(y_all))) + 1e-12)
    dir_acc = float(np.mean(np.sign(ens) == np.sign(y_all)))
    sel = ((ens > 0) & (p_all >= 0.52)) | ((ens < 0) & (p_all <= 0.48))
    joint_n = int(sel.sum())
    joint_acc = float(np.mean(np.sign(ens[sel]) == np.sign(y_all[sel]))) if joint_n else None
    return {'weights': weights, 'skill': skill, 'dir_acc': dir_acc, 'joint_n': joint_n, 'joint_acc': joint_acc,
            'joint_z': z_vs_half(joint_acc, joint_n) if joint_n else None}


def quant_signals_for_coin(df, p, progress=None):
    """Loopt de quant-worker na op elk hertrainmoment (elke `retrain_every` candles) over het testvenster.
    Tussen hertrainmomenten worden de bevroren modellen gebruikt om elke nieuwe candle te voorspellen."""
    feat = compute_features(df)
    n = len(df)
    R = int(p['retrain_every'])
    start = max(n - int(p['test_candles']), MIN_LOOKBACK + 100)
    ts = df['timestamp'].values.astype('int64')
    sigs, stats = [], []
    t = start
    total = max(1, (n - 1 - start) // R + 1)
    done = 0
    while t < n - 1:
        end = min(t + R, n - 1)
        sub = feat.iloc[max(0, t + 1 - QUANT_FETCH_LIMIT):t + 1]
        feats = usable_features(sub, BASE_FEATURES)
        res = None
        if feats:
            ds = build_dataset(sub, feats, QUANT_LOOKBACK)
            if ds is not None:
                res = walk_forward_eval(*ds)
        row = {'t': t, 'ts': int(ts[t]), 'evaluated': res is not None, 'has_edge': False,
               'joint_z': None, 'joint_n': 0, 'skill': None, 'long': 0, 'short': 0}
        if res is not None:
            row.update(joint_z=res['joint_z'], joint_n=res['joint_n'], skill=res['skill'])
            has_edge = (res['skill'] < 1.0 and res['joint_n'] >= p['min_joint_n']
                        and res['joint_z'] is not None and res['joint_z'] >= p['min_z'])
            row['has_edge'] = bool(has_edge)
            if has_edge:
                X, y = build_dataset(sub, feats, QUANT_LOOKBACK)
                models = make_regressors()
                for m in models.values():
                    m.fit(X, y)
                clf = None
                y_cls = (y > 0).astype(int)
                if y_cls.nunique() == 2:
                    clf = make_classifier()
                    clf.fit(X, y_cls)
                rows = feat.iloc[t:end][feats]
                ok = rows.notna().all(axis=1).values
                if ok.any() and clf is not None:
                    Xp = rows[ok]
                    ens = sum(res['weights'][k] * np.asarray(models[k].predict(Xp), dtype=float) for k in models)
                    prob_up = clf.predict_proba(Xp)[:, 1] * 100.0
                    idx = np.arange(t, end)[ok]
                    for j, e, pu in zip(idx, ens, prob_up):
                        if e > 0 and pu >= 52.0:
                            sigs.append((int(j), 1)); row['long'] += 1
                        elif e <= 0 and pu <= 48.0:
                            sigs.append((int(j), -1)); row['short'] += 1
        stats.append(row)
        t = end
        done += 1
        if progress:
            progress(done / total)
    return sigs, stats, start


def quant_signals_all(frames, p, progress=None, workers=2):
    out, done = {}, {}
    syms = list(frames)

    def run(sym):
        return sym, quant_signals_for_coin(frames[sym], p)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(syms)))) as pool:
        futs = [pool.submit(run, s) for s in syms]
        for k, fu in enumerate(as_completed(futs), 1):
            sym, res = fu.result()
            out[sym] = res
            if progress:
                progress(k / len(syms), sym)
    return out


# ============================================================== UITVOERING
def simulate_trade(a, t, side, ex):
    """Simuleert 1 trade vanaf signaal-candle t. Geeft None (ongeldig), {'filled': False, ...} of trade-dict."""
    ts, o, h, l, c, atrv = a['ts'], a['o'], a['h'], a['l'], a['c'], a['atr']
    n = len(c)
    if t + 1 >= n:
        return None
    atr, ref = atrv[t], c[t]
    if not (atr > 0):
        return None
    slip = ex['slip']
    if ex['mode'] == 'market':
        k = t + 1
        entry = o[k] * (1 + side * slip)
        base, entry_fee, full = entry, ex['taker'], True
    else:
        limit = ref - side * ex['limit_atr'] * atr
        last = min(t + int(ex['ttl']), n - 1)
        k = -1
        for j in range(t + 1, last + 1):
            if (side == 1 and l[j] <= limit) or (side == -1 and h[j] >= limit):
                k = j
                break
        if k < 0:
            return {'filled': False, 'expiry': int(ts[last]) + H4}
        entry = limit
        if (side == 1 and o[k] < limit) or (side == -1 and o[k] > limit):
            entry = o[k]  # gap door de limiet: beter gevuld
        base = limit if ex['sl_from_limit'] else ref
        entry_fee, full = ex['maker'], False
    sl = base - side * ex['sl_atr'] * atr
    tp = base + side * ex['tp_atr'] * atr
    sl0 = sl
    be_dist = ex['be_atr'] * atr
    mh = int(ex['max_hold'])
    secured = False
    exit_idx, exit_px, reason = -1, 0.0, ''
    for j in range(k, n):
        if j == k and not full:
            if (side == 1 and l[j] <= sl) or (side == -1 and h[j] >= sl):
                exit_idx, exit_px, reason = j, sl, 'stop'
                break
            continue
        if side == 1:
            if l[j] <= sl:
                exit_px = o[j] if o[j] < sl else sl
                exit_idx, reason = j, ('breakeven' if secured else 'stop')
                break
            if h[j] >= tp:
                exit_px, exit_idx, reason = max(o[j], tp), j, 'tp'
                break
            if not secured and be_dist > 0 and h[j] - entry >= be_dist:
                sl, secured = entry, True
        else:
            if h[j] >= sl:
                exit_px = o[j] if o[j] > sl else sl
                exit_idx, reason = j, ('breakeven' if secured else 'stop')
                break
            if l[j] <= tp:
                exit_px, exit_idx, reason = min(o[j], tp), j, 'tp'
                break
            if not secured and be_dist > 0 and entry - l[j] >= be_dist:
                sl, secured = entry, True
        if mh > 0 and (j - k + 1) >= mh:
            exit_idx, exit_px, reason = j, c[j], 'time'
            break
    if exit_idx < 0:
        exit_idx, exit_px, reason = n - 1, c[n - 1], 'open_end'
    if reason != 'tp':
        exit_px *= (1 - side * slip)
    exit_fee = ex['maker'] if (reason == 'tp' and ex['tp_maker']) else ex['taker']
    bars = exit_idx - k + 1
    funding = ex['funding8h'] * side * (bars * 4 / 8.0)
    pnl_pct = side * (exit_px - entry) / entry - entry_fee - exit_fee - funding
    risk_frac = abs(entry - sl0) / entry
    if risk_frac <= 0:
        return None
    return {'filled': True, 'entry_ts': int(ts[k]), 'exit_ts': int(ts[exit_idx]) + H4, 'entry': float(entry),
            'exit': float(exit_px), 'reason': reason, 'bars': int(bars), 'side': int(side),
            'pnl_pct': float(pnl_pct), 'R': float(pnl_pct / risk_frac), 'sl0': float(sl0), 'tp': float(tp)}


def run_portfolio(signals, arrays, ex, pf, notional):
    """signals: [(sym, t, side)]. Past max posities, cooldown, richting-limiet en dagverlies-limiet toe."""
    sigs = sorted(signals, key=lambda s: (arrays[s[0]]['ts'][s[1]], s[0]))
    busy, active, trades = {}, [], []
    f = defaultdict(int)
    for sym, t, side in sigs:
        a = arrays[sym]
        dec = int(a['ts'][t]) + H4
        f['signalen in portfolio-check'] += 1
        if busy.get(sym, 0) > dec:
            f['geblokkeerd: symbool bezet/cooldown'] += 1
            continue
        active = [x for x in active if x[0] > dec]
        if len(active) >= pf['max_pos']:
            f['geblokkeerd: max posities'] += 1
            continue
        if pf['max_same_dir'] > 0 and sum(1 for x in active if x[1] == side) >= pf['max_same_dir']:
            f['geblokkeerd: max zelfde richting'] += 1
            continue
        if pf['daily_loss'] > 0:
            day = dec // DAY
            loss = -sum(tr['pnl_usdt'] for tr in trades if tr['exit_ts'] <= dec and tr['exit_ts'] // DAY == day)
            if loss >= pf['daily_loss']:
                f['geblokkeerd: dagverlies-limiet'] += 1
                continue
        res = simulate_trade(a, t, side, ex)
        if res is None:
            f['ongeldig'] += 1
            continue
        if not res['filled']:
            f['limit niet gevuld (TTL)'] += 1
            busy[sym] = res['expiry']
            active.append((res['expiry'], side))
            continue
        res.update(sym=sym, sig_ts=int(a['ts'][t]), pnl_usdt=res['pnl_pct'] * notional)
        busy[sym] = res['exit_ts'] + int(pf['cooldown']) * H4
        active.append((res['exit_ts'], side))
        trades.append(res)
        f['uitgevoerde trades'] += 1
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values('exit_ts').reset_index(drop=True)
    return df, dict(f)


def metrics(tr, capital):
    if tr is None or tr.empty:
        return {'n': 0}
    r = tr['R'].values
    n = len(r)
    sd = r.std(ddof=1) if n > 1 else float('nan')
    t = r.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float('nan')
    gains, losses = tr.loc[tr['pnl_usdt'] > 0, 'pnl_usdt'].sum(), -tr.loc[tr['pnl_usdt'] < 0, 'pnl_usdt'].sum()
    eq = tr['pnl_usdt'].cumsum().values
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    dd = float((eq - peak).min()) if n else 0.0
    half = n // 2
    return {'n': n, 'winrate': float((tr['pnl_usdt'] > 0).mean() * 100), 'mean_R': float(r.mean()), 't': float(t),
            'pf': float(gains / losses) if losses > 0 else float('inf'),
            'total_usdt': float(tr['pnl_usdt'].sum()), 'return_pct': float(tr['pnl_usdt'].sum() / capital * 100),
            'maxdd_usdt': dd, 'maxdd_pct': float(dd / capital * 100),
            'R_first': float(r[:half].mean()) if half else float('nan'),
            'R_second': float(r[half:].mean()) if n - half else float('nan'),
            'avg_bars': float(tr['bars'].mean())}


def run_controls(signals, arrays, ex, pf, notional, n_runs, ranges, seed=1):
    """Random-entry-controle: zelfde aantal signalen per coin, willekeurige candle en richting, zelfde uitvoering."""
    syms = [s[0] for s in signals]
    out = []
    for k in range(n_runs):
        rng = np.random.default_rng(seed + k)
        rs = []
        for sym in syms:
            lo, hi = ranges[sym]
            rs.append((sym, int(rng.integers(lo, hi + 1)), int(rng.choice([-1, 1]))))
        tr, _ = run_portfolio(rs, arrays, ex, pf, notional)
        if not tr.empty:
            out.append(float(tr['R'].mean()))
    return out
