import json
import os

import ccxt
import pandas as pd
import plotly.express as px
import streamlit as st

from backtest_engine import simulate_twm, simulate_v20, summarize, walk_forward_gate
from github_utils import github_pull
from market_utils import TF_MS, fetch_ohlcv_paginated

st.set_page_config(page_title="V20 & TWM Backtest (live-logica)", page_icon="⚡", layout="wide")

st.title("⚡ V20 & TWM Backtest — draait op de échte live-logica")
st.markdown(
    "Gebruikt `compute_primary_setups` en de meta-labeling uit `agent.py`, en bootst de entry/SL/TP/trailing "
    "van de trader-workers na (limit-entry, ATR-stops, break-even, SMA20-target). Intrabar is pessimistisch: "
    "raken SL en TP in dezelfde bar, dan telt de SL. **Er worden nooit synthetische data gebruikt.**"
)

try:
    import agent as A  # live-strategie (setups, meta-labeling)
except Exception as e:  # bv. xgboost/sklearn ontbreekt in deze omgeving
    st.error(f"Kon `agent.py` niet importeren ({e}). Installeer de dependencies uit requirements (xgboost, scikit-learn, ccxt).")
    st.stop()

CORE_ASSETS = ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK']
NEEDED = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


# ==========================================
# DATA
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
allow_live = st.sidebar.checkbox("Ontbrekende candles live ophalen (KuCoin)", value=True,
                                 help="De agent pusht de 4h-cache niet naar GitHub; zonder lokale cache haalt de backtest "
                                      "de historie live op (traag bij de eerste keer, daarna 4u gecached).")
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

st.sidebar.header("🛡️ V20 parameters")
min_adx = st.sidebar.slider("Min ADX (4h)", 10.0, 35.0, float(A.MIN_ADX_TREND), 0.5)
trail_mult = st.sidebar.slider("SL (ATR)", 1.0, 5.0, float(A.TRAIL_SL_ATR), 0.1)
tp_mult = st.sidebar.slider("TP (ATR)", 2.0, 15.0, float(A.EFFECTIVE_TP_ATR), 0.5)
entry_offset = st.sidebar.slider("Limit-offset (ATR)", 0.0, 0.6, 0.2, 0.05)
use_gate = st.sidebar.checkbox("Meta-gate (walk-forward OOS)", value=True,
                               help="Aan: alleen setups waarvan het per-fold model P(TP) >= drempel geeft, en score >= min score. "
                                    "Uit: alle ruwe setups (laat zien of de setup zelf iets waard is).")
min_score = st.sidebar.slider("Min boss-score (trader)", 60.0, 100.0, 84.0, 1.0,
                              help="Trader-default MIN_BOSS_SCORE = 84; de agent zelf eist 78.")
lev_ungated = st.sidebar.selectbox("Hefboom zonder gate", [1, 2, 3], index=1)

st.sidebar.header("🚀 TWM parameters")
min_twm = st.sidebar.slider("Min TWM-score", 60.0, 100.0, 75.0, 1.0)
max_adx_twm = st.sidebar.slider("Max ADX (4h)", 15.0, 40.0, 28.0, 1.0)
twm_sl = st.sidebar.slider("SL (ATR)", 1.0, 4.0, 2.0, 0.1, key='twm_sl')
twm_be = st.sidebar.slider("Break-even bij (ATR winst)", 0.5, 3.0, 1.5, 0.1)
twm_min_tp = st.sidebar.slider("Min afstand SMA20-target (ATR)", 0.5, 3.0, 1.0, 0.1)
twm_min_rrr = st.sidebar.slider("Min RRR", 0.5, 3.0, 1.0, 0.1)
twm_cont = st.sidebar.checkbox("Test ook continuation (omgekeerde richting, RR 1:1)", value=True)

st.sidebar.header("💰 Kosten & grootte")
fee_side = st.sidebar.number_input("Kosten per kant (%) (fee + slippage)", 0.0, 0.5, 0.06, 0.01) / 100.0
margin = st.sidebar.number_input("Margin per trade (USDT)", 1.0, 100000.0, 20.0, 5.0)
capital = st.sidebar.number_input("Startkapitaal (USDT)", 10.0, 10_000_000.0, 1000.0, 100.0)

if not selected:
    st.info("Kies minstens één asset in de zijbalk.")
    st.stop()

# ==========================================
# DATA LADEN + SETUPS
# ==========================================
A.MIN_ADX_TREND = float(min_adx)
A.META_SL_ATR = float(trail_mult)
A.META_TP_ATR = float(tp_mult)


@st.cache_data(show_spinner=False)
def compute_setups(sig, min_adx_, _d4, _d1):
    A.MIN_ADX_TREND = float(min_adx_)
    return A.compute_primary_setups(_d4, _d1)


@st.cache_data(show_spinner="Meta-model walk-forward trainen (per fold)...")
def cached_gate(sigs, min_adx_, sl_, tp_, _histories):
    A.MIN_ADX_TREND, A.META_SL_ATR, A.META_TP_ATR = float(min_adx_), float(sl_), float(tp_)
    return walk_forward_gate(A, _histories)


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
    st.error("Geen bruikbare data. Zet 'live ophalen' aan, of draai eerst de agent/quant worker zodat er cache-bestanden zijn. "
             "De backtest valt bewust NIET terug op nepdata.")
    st.stop()

sigs = tuple(sorted((a, len(d4), int(d4['timestamp'].iloc[-1]), len(d1), int(d1['timestamp'].iloc[-1]))
                    for a, (d4, d1) in data.items()))
setups = {a: compute_setups(s, min_adx, d4, d1) for (a, (d4, d1)), s in zip(sorted(data.items()), sigs)}

gate_probs, gate_info, thr = None, None, None
if use_gate:
    gate_probs, gate_info = cached_gate(sigs, min_adx, trail_mult, tp_mult, data)
    thr = gate_info.get('threshold')

# ==========================================
# SIMULATIE
# ==========================================
v20_p = dict(trail_mult=trail_mult, tp_mult=tp_mult, entry_offset=entry_offset, be_atr=2.4, trail_factor=0.8,
             fee=fee_side, margin=margin, leverage=lev_ungated, min_score=min_score, base_score=float(A.MIN_BOSS_SCORE))
twm_p = dict(sl_mult=twm_sl, be_atr=twm_be, min_tp_atr=twm_min_tp, min_rrr=twm_min_rrr, min_score=min_twm,
             max_adx=max_adx_twm, fee=fee_side, margin=margin, leverage=2, cont_rr=1.0)


def run_all(fn, **kw):
    parts = [fn(a, data[a][0], setups[a], **kw) for a in data]
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


tr_v20 = run_all(simulate_v20, p=v20_p, gate=gate_probs, thr=thr) if (not use_gate or gate_probs) else pd.DataFrame()
tr_twm = run_all(simulate_twm, p=twm_p, mode='fade')
tr_cont = run_all(simulate_twm, p=twm_p, mode='continuation') if twm_cont else pd.DataFrame()


def show(tr, title):
    s = summarize(tr, capital)
    if s is None:
        st.warning("Geen trades met deze instellingen en data.")
        return
    c = st.columns(4)
    c[0].metric("Trades", s['n'], help=f"waarvan {s['open_end']} nog open aan het einde van de data")
    c[1].metric("Win rate", f"{s['win_rate']:.1f}%")
    c[2].metric("Gem. netto/trade", f"{s['avg_net_pct']:+.3f}%", help="Na kosten, op notional")
    c[3].metric("Gem. R", f"{s['avg_r']:+.2f}")
    c = st.columns(4)
    c[0].metric("Profit factor", "∞" if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}")
    c[1].metric("t-stat gem. rendement", "n.v.t." if s['tstat'] != s['tstat'] else f"{s['tstat']:.2f}",
                help="Gem. netto/trade gedeeld door standaardfout. Onder ~1.3 is het niet te onderscheiden van ruis.")
    c[2].metric("Eindkapitaal (gerealiseerd)", f"${s['final_equity']:,.2f}", f"{s['total_pnl']:+,.2f}")
    c[3].metric("Max drawdown", f"{s['max_dd_pct']:.2f}%")
    st.caption(f"Piek gelijktijdige posities: {s['peak_concurrent']} · gem. duur {s['avg_bars']:.1f} bars "
               f"({s['avg_bars'] * 4:.0f}u) · gerealiseerde equity, zonder funding/liquidatie.")

    st.plotly_chart(px.line(s['equity'], x='exit_ts', y='equity', title=f"{title} — equity (op exit-tijd)"), width="stretch")

    per = s['trades'].groupby('asset').agg(trades=('net_pct', 'size'), win_pct=('net_pct', lambda x: (x > 0).mean() * 100),
                                          gem_netto_pct=('net_pct', 'mean'), pnl_usd=('pnl_usd', 'sum')).round(3)
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
        st.download_button("⬇️ Download trades (CSV)", out.to_csv(index=False).encode('utf-8'),
                           file_name=f"trades_{title.split()[0].lower()}.csv", mime="text/csv")


tab_v20, tab_twm, tab_notes = st.tabs(["🛡️ V20 Trend-Pullback", "🚀 TWM Reversal", "ℹ️ Aannames & beperkingen"])

with tab_v20:
    if use_gate and gate_info is not None:
        ok = gate_info.get('ok')
        auc, z, tt = gate_info.get('auc'), gate_info.get('z'), gate_info.get('threshold')
        msg = (f"Meta-gate op **volledige historie** is nu {'**OPEN**' if ok else '**DICHT**'}: {gate_info.get('reason')} · "
               f"events {gate_info.get('n_events')} · AUC {auc if auc is None else round(auc, 3)} · "
               f"z {z if z is None else round(z, 2)} · drempel P(TP) {tt if tt is None else round(tt, 3)}")
        (st.success if ok else st.warning)(msg)
        st.caption("De backtest toont alleen het OOS-deel (per-fold modellen, eerste ~40% van de events is trainingsdata). "
                   "Als de gate live dicht is, handelt V20 live niet, ook al toont de backtest historische trades.")
    elif not use_gate:
        st.info("Meta-gate staat uit: dit zijn alle ruwe pullback-setups (zonder ML-filter).")
    show(tr_v20, "V20 Trend-Pullback")

with tab_twm:
    st.markdown("#### Fade (zoals live: terug naar SMA20)")
    show(tr_twm, "TWM fade")
    if twm_cont:
        st.markdown("#### Diagnostiek: continuation (zelfde signalen, omgekeerde richting, RR 1:1)")
        st.caption("Als dit beter scoort dan de fade, zijn %B-extremen met volume in deze data eerder momentum dan uitputting.")
        show(tr_cont, "TWM continuation")

with tab_notes:
    st.markdown("""
**Wat is gemodelleerd**
- V20: limit-entry 0.2 ATR gunstiger dan de close, 1 bar geldig; SL/TP in ATR; break-even + trailing vanaf 2.4 ATR piekwinst; meta-gate met per-fold (purged walk-forward) modellen.
- TWM: market-entry op de open na het signaal; SL 2 ATR; TP = SMA20; break-even bij 1.5 ATR; dezelfde filters (score, ADX, min afstand, RRR) als live.
- Per asset maximaal één positie tegelijk, zoals live. Kosten = 2 × kosten per kant.

**Wat NIET is gemodelleerd**
- De **quant-gate** (3-bar ensemble, `quant_worker.py`): live moet V20 hier ook mee akkoord zijn, dus live-trades ⊂ de trades hier. Het aantal trades is een bovengrens.
- Funding, liquidatie/margin-calls in cross-margin, orderboek-slippage bij grotere sizes, en minimale contractgrootte (BTC wordt live mogelijk overgeslagen bij $20 margin).
- Concurrency-limiet (MAX_CONCURRENT_POSITIONS) wordt niet afgedwongen; de piek gelijktijdige posities staat onder de statistieken.
- Intrabar-volgorde is onbekend; bij een dubbele hit telt de SL. De trailing gebruikt de bar-high/low en werkt pas vanaf de volgende bar.

**Interpretatie**
- Met ~2000 4h-bars per asset (≈11 maanden) en OOS pas vanaf ~40% zijn de aantallen klein. Kijk naar de t-stat en de spreiding per asset, niet naar de eindcurve.
- Alle assets zijn sterk gecorreleerd: 5 trades tegelijk is grotendeels één weddenschap.
""")
