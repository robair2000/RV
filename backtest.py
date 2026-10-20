import os
import json
import ccxt
import pandas as pd
import plotly.express as px
import streamlit as st

from github_utils import github_pull
from market_utils import TF_MS, fetch_ohlcv_paginated
from backtest_engine import simulate_twm, summarize, walk_forward_gate

st.set_page_config(page_title="Squeeze & Breakout Backtest", page_icon="🚀", layout="wide")

st.title("🚀 Squeeze & Breakout + 4h Markov Backtest")
st.markdown(
    "Draait op de échte live-logica uit `agent.py` en `markov_worker.py`. Test volatiliteits-compressie (Bollinger Squeeze), "
    "het 4h Markov-kleurrichtingsfilter en ATR-gebaseerde uitbraken. **Er worden nooit synthetische data gebruikt.**"
)

try:
    import agent as A  # Live-strategie (setups, meta-labeling)
except Exception as e:
    st.error(f"Kon `agent.py` niet importeren ({e}). Installeer de dependencies uit requirements (xgboost, scikit-learn, ccxt).")
    st.stop()

CORE_ASSETS = ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK']
NEEDED = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


# ==========================================
# DATA FUNCTIES
# ==========================================
def cache_path(asset, tf):
    return f"shared_cache_{tf}_{asset.lower()}.json"


def _clean(df):
    if df is None or len(df) == 0 or not all(c in df.columns for c in NEEDED):
        return None
    df = df[NEEDED].copy()
    for c in NEEDED:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    if df.empty:
        return None
    df['timestamp'] = df['timestamp'].astype('int64')
    return df


@st.cache_data(show_spinner=False)
def _read_cache(path, mtime):
    with open(path, 'r', encoding='utf-8') as f:
        c = json.load(f)
    return pd.DataFrame(c.get('data', []))


def read_local(asset, tf):
    p = cache_path(asset, tf)
    if not os.path.exists(p):
        return None
    try:
        return _clean(_read_cache(p, os.path.getmtime(p)))
    except Exception:
        return None


@st.cache_data(ttl=4 * 3600, show_spinner="Candles ophalen van KuCoin Futures...")
def fetch_live(asset, tf, bars):
    ex = ccxt.kucoinfutures({'enableRateLimit': True})
    return _clean(fetch_ohlcv_paginated(ex, f"{asset}/USDT:USDT", tf, total_bars=bars, pause=0.6))


def resample_daily(d4):
    day = TF_MS['1d']
    g = d4.assign(day=(d4['timestamp'] // day) * day).groupby('day')
    agg = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                close=('close', 'last'), volume=('volume', 'sum'), n=('open', 'size'))
    agg = agg[agg['n'] == 6].drop(columns='n').reset_index().rename(columns={'day': 'timestamp'})
    return _clean(agg)


def get_data(asset, allow_live, bars4h, bars1d):
    d4, src4 = read_local(asset, '4h'), 'lokale cache'
    if d4 is None and allow_live:
        d4, src4 = fetch_live(asset, '4h', bars4h), 'live KuCoin'
    if d4 is None or len(d4) < 250:
        return None
    d1, src1 = read_local(asset, '1d'), 'lokale cache'
    if d1 is None and allow_live:
        d1, src1 = fetch_live(asset, '1d', bars1d), 'live KuCoin'
    if d1 is None or len(d1) < 60:
        d1, src1 = resample_daily(d4), 'geresampled uit 4h'
    if d1 is None or len(d1) < 60:
        return None
    return d4, d1, src4, src1


# ==========================================
# SIDEBAR
# ==========================================
local_assets = sorted({f[len('shared_cache_4h_'):-len('.json')].upper()
                       for f in os.listdir('.') if f.startswith('shared_cache_4h_') and f.endswith('.json')})
all_assets = list(dict.fromkeys(local_assets + CORE_ASSETS))

st.sidebar.header("📁 Data")
allow_live = st.sidebar.checkbox("Ontbrekende candles live ophalen (KuCoin)", value=True)
bars4h = st.sidebar.number_input("4h-bars historie", 500, 6000, 2000, 250)
bars1d = st.sidebar.number_input("1d-bars historie", 100, 1500, 450, 50)
selected = st.sidebar.multiselect("Assets", all_assets, default=(local_assets or CORE_ASSETS))

if st.sidebar.button("🔄 Cache-bestanden van GitHub halen"):
    got = 0
    for a in selected:
        for tf in ('4h', '1d'):
            if github_pull(cache_path(a, tf), agent='backtest'):
                got += 1
    st.sidebar.success(f"{got} cache-bestand(en) bijgewerkt.")
    st.cache_data.clear()

st.sidebar.header("🚀 Squeeze & Breakout Parameters")
min_vol_ratio = st.sidebar.slider("Min Volume Ratio", 1.0, 3.0, float(getattr(A, 'MIN_VOL_RATIO', 1.3)), 0.1)
squeeze_pctl = st.sidebar.slider("Max BB Width Percentile (Squeeze)", 0.10, 0.50, 0.35, 0.05)
trail_mult = st.sidebar.slider("Stop Loss (ATR)", 1.0, 4.0, float(getattr(A, 'META_SL_ATR', 1.5)), 0.1)
tp_mult = st.sidebar.slider("Take Profit (ATR)", 2.0, 8.0, float(getattr(A, 'META_TP_ATR', 3.0)), 0.5)

st.sidebar.header("📊 Markov Filter Parameters")
min_markov_prob = st.sidebar.slider("Min 4h Markov Kleurkans (%)", 50.0, 65.0, 52.0, 1.0)

st.sidebar.header("🛡️ Meta-Model Gate")
use_gate = st.sidebar.checkbox("XGBoost Meta-gate inschakelen", value=True)
min_score = st.sidebar.slider("Min Boss Score", 60.0, 100.0, float(getattr(A, 'MIN_BOSS_SCORE', 75.0)), 1.0)

st.sidebar.header("💰 Kosten & Kapitaal")
fee_side = st.sidebar.number_input("Kosten per kant (%) (fee + slippage)", 0.0, 0.5, 0.06, 0.01) / 100.0
margin = st.sidebar.number_input("Margin per trade (USDT)", 1.0, 100000.0, 20.0, 5.0)
capital = st.sidebar.number_input("Startkapitaal (USDT)", 10.0, 10_000_000.0, 1000.0, 100.0)

if not selected:
    st.info("Kies minstens één asset in de zijbalk.")
    st.stop()

# ==========================================
# DATA LADEN + BEREKENINGEN
# ==========================================
A.MIN_VOL_RATIO = float(min_vol_ratio)
A.META_SL_ATR = float(trail_mult)
A.META_TP_ATR = float(tp_mult)


@st.cache_data(show_spinner=False)
def compute_setups(sig, _d4, _d1):
    m_data = {'probabilities_next_4h': {'green': min_markov_prob, 'red': min_markov_prob}}
    return A.compute_primary_setups(_d4, _d1, markov_data=m_data)


@st.cache_data(show_spinner="Meta-model walk-forward trainen (per fold)...")
def cached_gate(sigs, sl_, tp_, _histories):
    A.META_SL_ATR, A.META_TP_ATR = float(sl_), float(tp_)
    return walk_forward_gate(A)


data, rows = {}, []
for a in selected:
    got = get_data(a, allow_live, int(bars4h), int(bars1d))
    if got is None:
        rows.append({'Asset': a, '4h-bars': 0, 'Periode': '—', '1d-bars': 0, 'Bron 4h': 'geen data', 'Bron 1d': '—'})
        continue
    d4, d1, s4, s1 = got
    data[a] = (d4, d1)
    rows.append({
        'Asset': a, '4h-bars': len(d4),
        'Periode': f"{pd.to_datetime(d4['timestamp'].iloc[0], unit='ms'):%Y-%m-%d} → {pd.to_datetime(d4['timestamp'].iloc[-1], unit='ms'):%Y-%m-%d}",
        '1d-bars': len(d1), 'Bron 4h': s4, 'Bron 1d': s1,
    })

with st.expander("📦 Gebruikte data", expanded=not data):
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

if not data:
    st.error("Geen bruikbare data gevonden. Zet 'live ophalen' aan of koppel KuCoin.")
    st.stop()

sigs = tuple(sorted((a, len(d4), int(d4['timestamp'].iloc[-1]), len(d1), int(d1['timestamp'].iloc[-1]))
                    for a, (d4, d1) in data.items()))
setups = {a: compute_setups(s, d4, d1) for (a, (d4, d1)), s in zip(sorted(data.items()), sigs)}

gate_probs, gate_info, thr = None, None, None
if use_gate:
    gate_probs, gate_info = cached_gate(sigs, trail_mult, tp_mult, data)
    thr = gate_info.get('threshold') if gate_info else 0.52

# ==========================================
# SIMULATIE RUNNER
# ==========================================
sqz_params = dict(
    sl_mult=trail_mult,
    be_atr=1.5,
    min_tp_atr=tp_mult,
    min_rrr=tp_mult / trail_mult,
    min_score=min_score,
    max_adx=100.0,
    fee=fee_side,
    margin=margin,
    leverage=2
)


def run_all(fn, **kw):
    parts = [fn(a, data[a][0], setups[a], **kw) for a in data]
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


tr_sqz = run_all(simulate_twm, p=sqz_params, mode='fade')


def show(tr, title):
    s = summarize(tr, capital)
    if s is None:
        st.warning("Geen trades gevonden met deze instellingen en data.")
        return
    c = st.columns(4)
    c[0].metric("Trades", s['n'], help=f"waarvan {s['open_end']} nog open op het einde")
    c[1].metric("Win rate", f"{s['win_rate']:.1f}%")
    c[2].metric("Gem. netto/trade", f"{s['avg_net_pct']:+.3f}%")
    c[3].metric("Gem. R", f"{s['avg_r']:+.2f}")
    c = st.columns(4)
    c[0].metric("Profit factor", "∞" if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}")
    c[1].metric("t-stat rendement", "n.v.t." if s['tstat'] != s['tstat'] else f"{s['tstat']:.2f}")
    c[2].metric("Eindkapitaal", f"${s['final_equity']:,.2f}", f"{s['total_pnl']:+,.2f}")
    c[3].metric("Max drawdown", f"{s['max_dd_pct']:.2f}%")

    st.plotly_chart(px.line(s['equity'], x='exit_ts', y='equity', title=f"{title} — Equity Curve"), width="stretch")

    per = s['trades'].groupby('asset').agg(
        trades=('net_pct', 'size'),
        win_pct=('net_pct', lambda x: (x > 0).mean() * 100),
        gem_netto_pct=('net_pct', 'mean'),
        pnl_usd=('pnl_usd', 'sum')
    ).round(3)
    
    exits = s['trades']['reason'].value_counts().rename('aantal')
    c1, c2 = st.columns(2)
    c1.markdown("**Per asset**")
    c1.dataframe(per, width="stretch")
    c2.markdown("**Exit-redenen**")
    c2.dataframe(exits, width="stretch")

    with st.expander("Alle trades"):
        out = s['trades'].copy()
        for col in ('signal_ts', 'entry_ts', 'exit_ts'):
            out[col] = pd.to_datetime(out[col], unit='ms')
        st.dataframe(out, width="stretch")
        st.download_button(
            "📥 Download trades (CSV)", 
            out.to_csv(index=False).encode('utf-8'),
            file_name=f"trades_{title.split()[0].lower()}.csv", 
            mime="text/csv",
            key=f"download_btn_{title.split()[0].lower()}"
        )


tab_sqz, tab_notes = st.tabs(["🚀 Squeeze & Breakout Resultaten", "ℹ️ Model & Aannames"])

with tab_sqz:
    if use_gate and gate_info is not None:
        ok = gate_info.get('ok')
        auc, z, tt = gate_info.get('auc'), gate_info.get('z'), gate_info.get('threshold')
        msg = (f"XGBoost Meta-gate status: {'**OPEN**' if ok else '**DICHT**'} · "
               f"Events: {gate_info.get('n_events')} · AUC: {auc if auc is None else round(auc, 3)} · "
               f"drempel P(TP): {tt if tt is None else round(tt, 3)}")
        (st.success if ok else st.warning)(msg)
    show(tr_sqz, "Squeeze Breakout Backtest")

with tab_notes:
    st.markdown("""
**Backtest Logica & Aannames:**
- **Squeeze Filter:** Bollinger Bands percentiel $\le$ 35% met volume-expansie $\ge$ 1.3.
- **Risk Management:** 1.5 ATR Stop Loss, 3.0 ATR Take Profit, break-even geactiveerd vanaf 1.5 ATR winst.
- **Kapitaal:** Getest op een vaste marge van $20 per trade vermenigvuldigd met de gekozen hefboom.
""")
