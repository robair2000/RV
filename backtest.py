"""
V20 Backtest Dashboard (Streamlit)

Start:   streamlit run backtest_app.py
Nodig:   pip install streamlit plotly ccxt pandas numpy scikit-learn xgboost catboost
Naast dit bestand moet backtest_engine.py staan.

Tab 1: Agent + TWM  -> squeeze-signalen uit de (aangepaste) agent.py + momentum-trader (market entry)
Tab 2: Quant + Trader -> walk-forward quant_worker v8 + quant-trader (limit entry)
Alle cijfers komen uit een echte simulatie op historische candles (geen vaste teksten of willekeurige curves).
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

import backtest_engine as be

st.set_page_config(page_title="V20 Backtest", page_icon="📊", layout="wide")

try:
    _VER = tuple(int(x) for x in st.__version__.split('.')[:2])
except Exception:  # noqa
    _VER = (1, 0)


def stretch(fn, *args, **kw):
    """Compatibel met oude (use_container_width) en nieuwe (width='stretch') Streamlit-versies."""
    if _VER >= (1, 50):
        try:
            return fn(*args, width='stretch', **kw)
        except Exception:  # noqa
            pass
    return fn(*args, use_container_width=True, **kw)


def fmt_ts(ms):
    return pd.to_datetime(ms, unit='ms').strftime('%Y-%m-%d')


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False, ttl=3600)
def get_frames(source, symbols, bars):
    if source == 'synthetisch':
        return be.synthetic_frames(list(symbols), bars), []
    return be.load_kucoin_frames(list(symbols), bars)


# ------------------------------------------------------------------ weergave
def verdict(m, ctrl):
    if m.get('n', 0) == 0:
        return 'error', 'Geen trades: er kwam geen enkel signaal door de filters.'
    if m['n'] < 100:
        return 'warning', f"Slechts {m['n']} trades: te weinig om iets uit te concluderen, hoe mooi de cijfers ook zijn."
    p95 = float(np.percentile(ctrl, 95)) if ctrl else None
    checks = {
        'gemiddelde netto R > 0': m['mean_R'] > 0,
        'totaalresultaat in USDT > 0': m['total_usdt'] > 0,
        't-waarde > 2': m['t'] > 2,
        'beide helften van de periode positief': m['R_first'] > 0 and m['R_second'] > 0,
        'duidelijk beter dan random entries (p95)': (p95 is not None and m['mean_R'] > p95),
    }
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        return 'success', ('Alle vier de toetsen gehaald. Dat is interessant, maar nog geen bewijs: test op een andere '
                           'periode en eerst in DRY_RUN voordat je er geld op zet.')
    return 'error', 'Geen aantoonbaar voordeel. Niet gehaald: ' + '; '.join(failed) + '.'


def render_results(res):
    import plotly.graph_objects as go
    tr, m, ctrl = res['trades'], res['metrics'], res['controls']
    kind, msg = verdict(m, ctrl)
    getattr(st, kind)(msg)
    if m.get('n', 0) == 0:
        st.markdown("#### Trechter")
        stretch(st.dataframe, pd.DataFrame(list(res['funnel'].items()), columns=['stap', 'aantal']), hide_index=True)
        return

    st.caption(f"Periode: {fmt_ts(tr['entry_ts'].min())} t/m {fmt_ts(tr['exit_ts'].max())} · "
               f"notional per trade {res['notional']:.0f} USDT · {len(res['per_coin'])} coins")
    c = st.columns(6)
    c[0].metric("Trades", m['n'])
    c[1].metric("Winrate", f"{m['winrate']:.1f}%")
    c[2].metric("Gem. netto R", f"{m['mean_R']:+.3f}", f"t = {m['t']:.2f}", delta_color='off')
    c[3].metric("Resultaat", f"{m['total_usdt']:+.2f} USDT", f"{m['return_pct']:+.2f}% van kapitaal", delta_color='off')
    c[4].metric("Max drawdown", f"{m['maxdd_usdt']:.2f} USDT", f"{m['maxdd_pct']:.2f}%", delta_color='off')
    c[5].metric("Profit factor", "∞" if m['pf'] == float('inf') else f"{m['pf']:.2f}",
                f"gem. {m['avg_bars']:.1f} candles in trade", delta_color='off')

    st.caption("R = resultaat gedeeld door het initiële risico (afstand tot de stop) van die trade. De inzet in USDT is per trade "
               "vast, dus het USDT-resultaat kan afwijken van de gemiddelde R als de ATR per trade verschilt.")
    left, right = st.columns([3, 2])
    with left:
        eq = tr['pnl_usdt'].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(tr['exit_ts'], unit='ms'), y=eq, mode='lines', name='Equity (cumulatief)'))
        fig.add_trace(go.Scatter(x=pd.to_datetime(tr['exit_ts'], unit='ms'), y=eq - np.maximum.accumulate(np.r_[0.0, eq.values])[1:],
                                 mode='lines', name='Drawdown', line=dict(width=1), fill='tozeroy'))
        fig.update_layout(title="Cumulatief resultaat (USDT, vaste inzet per trade, niet samengesteld)",
                          height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h'))
        stretch(st.plotly_chart, fig)
    with right:
        fig2 = go.Figure()
        if ctrl:
            fig2.add_trace(go.Histogram(x=ctrl, nbinsx=25, name='Random entries'))
            fig2.add_vline(x=m['mean_R'], line_color='red', annotation_text='strategie')
            p = (1 + sum(1 for x in ctrl if x >= m['mean_R'])) / (len(ctrl) + 1)
            fig2.update_layout(title=f"Gem. R: strategie vs {len(ctrl)} random-runs (p ≈ {p:.3f})")
        else:
            fig2.add_trace(go.Histogram(x=tr['R'], nbinsx=30, name='R per trade'))
            fig2.update_layout(title="Verdeling van R per trade (geen controle-runs gedraaid)")
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        stretch(st.plotly_chart, fig2)

    a, b, c3 = st.columns(3)
    with a:
        st.markdown("**Trechter**")
        stretch(st.dataframe, pd.DataFrame(list(res['funnel'].items()), columns=['stap', 'aantal']), hide_index=True)
    with b:
        st.markdown("**Uitstapredenen**")
        er = tr.groupby('reason').agg(trades=('R', 'size'), gem_R=('R', 'mean')).round(3).reset_index()
        stretch(st.dataframe, er, hide_index=True)
        st.caption(f"Eerste helft gem. R {m['R_first']:+.3f} · tweede helft {m['R_second']:+.3f}")
    with c3:
        st.markdown("**Per coin**")
        pc = tr.groupby('sym').agg(trades=('R', 'size'), gem_R=('R', 'mean'), resultaat_usdt=('pnl_usdt', 'sum')).round(3).reset_index()
        stretch(st.dataframe, pc, hide_index=True)

    for title, df in res.get('extra_tables', []):
        st.markdown(f"**{title}**")
        stretch(st.dataframe, df, hide_index=True)

    with st.expander("Alle trades"):
        show = tr.copy()
        for col in ('sig_ts', 'entry_ts', 'exit_ts'):
            show[col] = pd.to_datetime(show[col], unit='ms')
        show['side'] = np.where(show['side'] == 1, 'long', 'short')
        show = show[['sym', 'side', 'sig_ts', 'entry_ts', 'exit_ts', 'entry', 'exit', 'reason', 'bars', 'pnl_pct', 'R', 'pnl_usdt']]
        stretch(st.dataframe, show, hide_index=True)
        st.download_button("Download trades (CSV)", show.to_csv(index=False).encode('utf-8'),
                           file_name=f"{res['key']}_trades.csv", mime='text/csv')


def finish(key, signals, arrays, ranges, ex, pf, notional, capital, n_controls, funnel_head, per_coin, extra=None):
    trades, pf_funnel = be.run_portfolio(signals, arrays, ex, pf, notional)
    funnel = dict(funnel_head)
    funnel.update(pf_funnel)
    metrics = be.metrics(trades, capital)
    ctrl = []
    if n_controls > 0 and signals:
        ctrl = be.run_controls(signals, arrays, ex, pf, notional, n_controls, ranges)
    return {'key': key, 'trades': trades, 'metrics': metrics, 'controls': ctrl, 'funnel': funnel,
            'per_coin': per_coin, 'notional': notional, 'extra_tables': extra or []}


# ------------------------------------------------------------------ sidebar
def sidebar():
    st.sidebar.header("⚙️ Algemeen")
    source = st.sidebar.radio("Databron", ['KuCoin Futures', 'synthetisch'],
                              help="'synthetisch' is willekeurige data zonder edge: alleen om de app te testen.")
    symbols = st.sidebar.multiselect("Coins", be.CORE_SYMBOLS + be.EXTRA_SYMBOLS, default=be.CORE_SYMBOLS)
    bars = st.sidebar.slider("Aantal 4h-candles historie", 1500, 6000, 3000, 250,
                             help="3000 candles ≈ 500 dagen. De quant-tab heeft minstens ~1500 nodig.")
    capital = st.sidebar.number_input("Startkapitaal (USDT)", value=1000.0, step=100.0)
    margin = st.sidebar.number_input("Marge per trade (USDT)", value=20.0, step=5.0)
    leverage = st.sidebar.slider("Hefboom", 1, 5, 2)
    st.sidebar.subheader("Kosten")
    taker = st.sidebar.number_input("Taker-fee per kant (%)", value=0.06, step=0.01, format="%.3f") / 100
    maker = st.sidebar.number_input("Maker-fee per kant (%)", value=0.02, step=0.01, format="%.3f") / 100
    slip = st.sidebar.number_input("Slippage bij market/stop (%)", value=0.03, step=0.01, format="%.3f") / 100
    funding = st.sidebar.number_input("Funding per 8 uur (%)", value=0.01, step=0.005, format="%.3f") / 100
    tp_maker = st.sidebar.checkbox("Take-profit als maker (anders taker)", value=False)
    st.sidebar.subheader("Portfolio-regels")
    max_pos = st.sidebar.slider("Max gelijktijdige posities", 1, 10, 5)
    max_same = st.sidebar.slider("Max zelfde richting (0 = uit)", 0, 10, 0)
    cooldown = st.sidebar.slider("Cooldown na uitstap (candles)", 0, 6, 1)
    daily = st.sidebar.number_input("Dagverlies-limiet (USDT, 0 = uit)", value=0.0, step=5.0)
    n_controls = st.sidebar.slider("Random-controle runs", 0, 200, 50,
                                   help="Zelfde uitvoering en kosten, maar willekeurige entries. 0 = overslaan.")
    base = dict(taker=taker, maker=maker, slip=slip, funding8h=funding, tp_maker=tp_maker)
    pf = dict(max_pos=max_pos, max_same_dir=max_same, cooldown=cooldown, daily_loss=daily)
    return dict(source=source, symbols=tuple(symbols), bars=bars, capital=capital, notional=margin * leverage,
                n_controls=n_controls, base=base, pf=pf)


def load(cfg):
    if not cfg['symbols']:
        st.warning("Kies minstens één coin.")
        return None
    with st.spinner("Data ophalen (bij KuCoin kan dit even duren; daarna staat het in de cache)..."):
        frames, problems = get_frames(cfg['source'], cfg['symbols'], cfg['bars'])
    for pr in problems:
        st.warning(pr)
    return frames or None


# ------------------------------------------------------------------ tab 1
def tab_agent(cfg):
    st.subheader("Agent + TWM → momentum-trader")
    st.caption("Signaal: squeeze (bb-breedte-percentiel ≤ drempel over 100 candles) + volume ≥ drempel, richting = close t.o.v. SMA20. "
               "Uitvoering: market-entry, SL/TP in ATR, breakeven (zoals trader_worker_momentum.py).")
    with st.expander("Instellingen agent + trader", expanded=True):
        c1, c2, c3 = st.columns(3)
        mode = c1.selectbox("Agent-versie", ['corrected', 'as_is'], index=0, format_func=lambda x: {
            'corrected': 'Huidige agent.py: alleen een setup op de laatste candle telt',
            'as_is': 'Vorige agent.py (met stale-bug), alleen ter vergelijking'}[x])
        direction = c1.selectbox("Richtingsregel", ['sma', 'breakout'], format_func=lambda x: {
            'sma': 'close > SMA20 (huidige agent)', 'breakout': '%B-uitbraak (oude agent)'}[x])
        sq = c2.slider("Squeeze-percentiel ≤", 0.05, 0.60, 0.35, 0.05)
        vol = c2.slider("Min. volume-ratio", 1.0, 3.0, 1.3, 0.1)
        lag = c3.selectbox("Vertraging vorige versie (candles)", [1, 2], index=1,
                           help="agent.py kapt de lopende candle af (iloc[:-1]) en gebruikt range(..., len-1). Als "
                                "fetch_ohlcv_paginated de lopende candle al weglaat, is de vertraging 2; anders 1.")
        fixed_atr = c3.checkbox("ATR vast op 2% (vorige versie)", value=True,
                                help="agent.py leest 'atr_pct' uit d4, maar die kolom bestaat daar niet: het wordt altijd 2.0.")
        c4, c5, c6, c7 = st.columns(4)
        sl = c4.number_input("SL (ATR)", value=1.5, step=0.25)
        tp = c5.number_input("TP (ATR)", value=3.0, step=0.25)
        be_atr = c6.number_input("Breakeven bij (ATR, 0 = uit)", value=1.5, step=0.25)
        mh = c7.number_input("Max. candles in trade (0 = onbeperkt)", value=0, step=1,
                             help="De live trader heeft geen tijdlimiet; 0 is dus gelijk aan live.")
    if mode == 'as_is':
        st.warning("In deze modus draait de backtest de VORIGE agent.py (voor de fix), alleen ter vergelijking. Let op: in de live trader wijst "
                   "MAX_CANDLE_AGE_HOURS (5,5 u) signalen met deze vertraging waarschijnlijk af. Gebruik je een hogere waarde, "
                   "dan krijg je wat hier getoond wordt.")
    if st.button("▶ Draai backtest: Agent + TWM", type='primary', key='run_agent'):
        frames = load(cfg)
        if frames is None:
            return
        p = dict(squeeze_pctl=sq, min_vol=vol, direction=direction, mode=mode, lag=lag)
        ex = dict(cfg['base'], mode='market', limit_atr=0.0, ttl=1, sl_atr=sl, tp_atr=tp, be_atr=be_atr,
                  max_hold=mh, sl_from_limit=False)
        arrays, signals, ranges, rows = {}, [], {}, []
        fh = defaultdict(int)
        for sym, df in frames.items():
            a = be.to_arrays(df)
            sigs, fun, first = be.agent_signals(df, p)
            if mode == 'as_is' and fixed_atr:
                a = be.with_fixed_atr(a, 2.0)
            arrays[sym] = a
            ranges[sym] = (first, len(df) - 2)
            signals += [(sym, t, s) for t, s in sigs]
            fh['candles geëvalueerd'] += fun['candles']
            fh['echte squeeze-setups (long)'] += fun['setups_long']
            fh['echte squeeze-setups (short)'] += fun['setups_short']
            rows.append({'coin': sym.split('/')[0], 'candles': fun['candles'], 'setups': fun['setups_long'] + fun['setups_short'],
                         'signalen': fun['signalen']})
        fh['signalen naar trader'] = len(signals)
        with st.spinner("Simuleren..."):
            res = finish('agent', signals, arrays, ranges, ex, cfg['pf'], cfg['notional'], cfg['capital'],
                         cfg['n_controls'], fh, rows)
        st.session_state['res_agent'] = res
    if 'res_agent' in st.session_state:
        render_results(st.session_state['res_agent'])


# ------------------------------------------------------------------ tab 2
def tab_quant(cfg):
    st.subheader("Quant (walk-forward) → quant-trader")
    st.caption("Signaal: quant_worker v8 op elk hertrainmoment (expanding walk-forward met embargo, ensemble xgb/cat/rf, "
               "poort op z-score). Uitvoering: limit-order op 0,2 ATR, SL/TP uit de state (zoals de quant-trader).")
    st.info(f"Modellen: {be.model_backend()}. Funding-noodrem is niet gemodelleerd (geen historische funding opgehaald).")
    with st.expander("Instellingen quant + trader", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        min_z = c1.slider("Min. z-score (QUANT_MIN_Z)", 1.0, 3.0, 1.28, 0.05)
        min_n = c2.slider("Min. gezamenlijke signalen (QUANT_MIN_JOINT_N)", 10, 200, 30, 5)
        retrain = c3.selectbox("Hertrainen elke … candles", [1, 3, 6, 12, 24], index=3,
                               help="Live: elke 4h-candle (=1). Hogere waarde = sneller, iets minder exact.")
        test_c = c4.slider("Testvenster (laatste N candles)", 240, 2400, 1000, 120,
                           help="Elke hertraining kost ~1 s per coin. Reken: (N / hertrain-interval) × aantal coins ÷ 2 seconden.")
        c5, c6, c7, c8, c9 = st.columns(5)
        limit_atr = c5.number_input("Limit-afstand (ATR)", value=0.2, step=0.05)
        ttl = c6.number_input("Order geldig (candles)", value=1, step=1, min_value=1)
        sl = c7.number_input("SL (ATR)", value=2.5, step=0.25)
        tp = c8.number_input("TP (ATR)", value=5.0, step=0.25)
        be_atr = c9.number_input("Breakeven bij (ATR)", value=2.0, step=0.25)
        c10, c11 = st.columns(2)
        sl_from_limit = c10.checkbox("SL/TP herberekenen vanaf de limietprijs", value=False,
                                     help="Live gebruikt de quant-trader SL/TP uit de state (gebaseerd op de slotkoers). Aanvinken = voorgestelde verbetering.")
        mh = c11.number_input("Max. candles in trade (0 = onbeperkt)", value=0, step=1)
    n_c = len(cfg['symbols'])
    est = (test_c / retrain) * n_c / 2
    st.caption(f"Geschatte rekentijd: ± {est / 60:.1f} minuut (2 threads).")
    if st.button("▶ Draai backtest: Quant + Trader", type='primary', key='run_quant'):
        frames = load(cfg)
        if frames is None:
            return
        short = [s.split('/')[0] for s, d in frames.items() if len(d) < be.MIN_LOOKBACK + 400]
        if short:
            st.warning(f"Te weinig candles voor {', '.join(short)}: verhoog het aantal candles historie.")
        frames = {s: d for s, d in frames.items() if len(d) >= be.MIN_LOOKBACK + 400}
        if not frames:
            return
        p = dict(retrain_every=retrain, test_candles=test_c, min_joint_n=min_n, min_z=min_z)
        bar = st.progress(0.0, text="Modellen trainen (walk-forward)...")
        out = be.quant_signals_all(frames, p, progress=lambda f, s=None: bar.progress(min(1.0, f), text=f"Klaar: {s}" if s else None))
        bar.empty()
        ex = dict(cfg['base'], mode='limit', limit_atr=limit_atr, ttl=ttl, sl_atr=sl, tp_atr=tp, be_atr=be_atr,
                  max_hold=mh, sl_from_limit=sl_from_limit)
        arrays, signals, ranges, rows = {}, [], {}, []
        fh = defaultdict(int)
        for sym, (sigs, stats, start) in out.items():
            arrays[sym] = be.to_arrays(frames[sym])
            ranges[sym] = (start, len(frames[sym]) - 2)
            signals += [(sym, t, s) for t, s in sigs]
            ev = [r for r in stats if r['evaluated']]
            edge = [r for r in stats if r['has_edge']]
            zs = [r['joint_z'] for r in ev if r['joint_z'] is not None]
            fh['hertrainmomenten'] += len(stats)
            fh['model kon geëvalueerd'] += len(ev)
            fh['edge-poort open'] += len(edge)
            rows.append({'coin': sym.split('/')[0], 'hertrainingen': len(stats), 'poort open': len(edge),
                         '% open': round(100 * len(edge) / max(1, len(stats)), 1),
                         'mediaan z': round(float(np.median(zs)), 2) if zs else None,
                         'long': sum(r['long'] for r in stats), 'short': sum(r['short'] for r in stats)})
        fh['signalen naar trader'] = len(signals)
        with st.spinner("Simuleren..."):
            res = finish('quant', signals, arrays, ranges, ex, cfg['pf'], cfg['notional'], cfg['capital'],
                         cfg['n_controls'], fh, rows, extra=[("Edge-poort per coin (hoe vaak is de quant-poort open?)", pd.DataFrame(rows))])
        st.session_state['res_quant'] = res
    if 'res_quant' in st.session_state:
        render_results(st.session_state['res_quant'])


# ------------------------------------------------------------------ main
def main():
    st.title("📊 V20 Backtest")
    st.caption("Echte simulatie op historische candles, met kosten, portfolio-regels en een random-entry-controle. "
               "Een mooie backtest is geen garantie: let op het aantal trades, de twee helften van de periode en de controle.")
    cfg = sidebar()
    if cfg['source'] == 'synthetisch':
        st.warning("Databron is synthetisch (willekeurige koersen zonder edge). Dit is alleen om de app te testen.")
    t1, t2 = st.tabs(["🤖 Agent + TWM", "⚡ Trader + Quant"])
    with t1:
        tab_agent(cfg)
    with t2:
        tab_quant(cfg)


if __name__ == '__main__':
    main()
