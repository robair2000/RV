import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import subprocess
import plotly.express as px

# --- PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="V20 & TWM Super Quant Backtest & Tuner",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ V20 & TWM Super Quant Backtest & Parameter Tuner")
st.markdown("Automatische synchronisatie met GitHub, multi-asset selectie en live parameter-tuning voor je V20- en TWM-strategieën.")

# ==========================================
# 0. GITHUB SYNC & ASSET SELECTIE
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🔄 GitHub & Data Sync")

def auto_git_pull():
    try:
        subprocess.run(["git", "pull", "--rebase"], capture_output=True, text=True)
    except Exception:
        pass

# Automatische pull bij laden
auto_git_pull()

if st.sidebar.button("🔄 Ververs Data van GitHub"):
    try:
        res = subprocess.run(["git", "pull", "--rebase"], capture_output=True, text=True)
        if res.returncode == 0:
            st.sidebar.success("✅ Lokale data succesvol bijgewerkt!")
        else:
            st.sidebar.warning(f"⚠️ Git melding: {res.stdout.strip()}")
    except Exception as e:
        st.sidebar.error(f"❌ Sync fout: {e}")

# Scan automatisch naar alle aanwezige shared_cache_4h bestanden
available_files = [f for f in os.listdir('.') if f.startswith('shared_cache_4h_') and f.endswith('.json')]
available_assets = [f.replace('shared_cache_4h_', '').replace('.json', '').upper() for f in available_files]

st.sidebar.header("📁 Asset & Bestand Selectie")
if available_assets:
    selected_asset = st.sidebar.selectbox("Kies een Munt (4h Cache)", available_assets)
    target_cache_file = f"shared_cache_4h_{selected_asset.lower()}.json"
else:
    selected_asset = None
    target_cache_file = None
    st.sidebar.warning("⚠️ Geen shared_cache_4h bestanden gevonden. Zorg dat de agent gedraaid heeft.")

uploaded_file = st.sidebar.file_uploader("Of upload handmatig een JSON/CSV bestand", type=["json", "csv"])

# ==========================================
# 1. PARAMETERS & SCHUIFREGELAARS
# ==========================================
params = {
    "min_adx_trend": 18.0,
    "min_boss_score": 78.0,
    "min_twm_score": 75.0,
    "meta_sl_atr": 2.5,
    "meta_tp_atr": 8.0,
    "trailing_mult": 2.0,
    "initial_capital": 10000.0,
}

df_candles = pd.DataFrame()
active_source_name = "Geen data"

try:
    if uploaded_file is not None:
        content = json.load(uploaded_file) if uploaded_file.name.endswith(".json") else pd.read_csv(uploaded_file)
        active_source_name = f"Upload: {uploaded_file.name}"
    elif target_cache_file and os.path.exists(target_cache_file):
        with open(target_cache_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        active_source_name = f"Cache: {target_cache_file}"
    else:
        content = None

    if content is not None:
        if isinstance(content, dict) and 'data' in content:
            df_candles = pd.DataFrame(content['data'])
        elif isinstance(content, list):
            df_candles = pd.DataFrame(content)
        elif isinstance(content, pd.DataFrame):
            df_candles = content

        for k in params.keys():
            if isinstance(content, dict) and k in content:
                params[k] = content[k]
except Exception as e:
    st.sidebar.error(f"⚠️ Fout bij inlezen data: {e}")

st.sidebar.info(f"📂 Actieve bron: {active_source_name}")

st.sidebar.header("🎛️ Strategie Schuifregelaars")
st.sidebar.subheader("V20 Trend-Pullback")
min_adx = st.sidebar.slider("Minimale ADX Trend", 10.0, 35.0, float(params["min_adx_trend"]), 0.5)
min_boss = st.sidebar.slider("Minimale Boss Score", 60.0, 95.0, float(params["min_boss_score"]), 1.0)
meta_sl = st.sidebar.slider("Meta Stop Loss (ATR)", 1.0, 5.0, float(params["meta_sl_atr"]), 0.1)
meta_tp = st.sidebar.slider("Meta Take Profit (ATR)", 2.0, 15.0, float(params["meta_tp_atr"]), 0.5)

st.sidebar.subheader("TWM Mean-Reversion")
min_twm = st.sidebar.slider("Minimale TWM Score", 60.0, 90.0, float(params["min_twm_score"]), 1.0)
trailing_mult = st.sidebar.slider("Trailing Stop Multiplier", 1.0, 4.0, float(params["trailing_mult"]), 0.1)

capital = st.sidebar.number_input("Startkapitaal ($)", value=float(params["initial_capital"]), step=1000.0)

# ==========================================
# 2. DATA PREPARATIE & INDICATOREN
# ==========================================
def process_data(df_in):
    if df_in is None or df_in.empty or not all(col in df_in.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="4h")
        price = 60000 + np.cumsum(np.random.randn(1000) * 400)
        df = pd.DataFrame({"timestamp": dates, "open": price, "high": price+200, "low": price-200, "close": price, "volume": np.random.rand(1000)*100})
    else:
        df = df_in.copy()
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start="2024-01-01", periods=len(df), freq="4h")

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean().fillna(close * 0.02)
    
    up = high.diff()
    down = -low.diff()
    tr_s = tr.ewm(alpha=1.0/14, adjust=False).mean()
    plus_di = 100.0 * up.where((up > down) & (up > 0), 0.0).ewm(alpha=1.0/14).mean() / (tr_s + 1e-8)
    minus_di = 100.0 * down.where((down > up) & (down > 0), 0.0).ewm(alpha=1.0/14).mean() / (tr_s + 1e-8)
    df['adx'] = (100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)).ewm(alpha=1.0/14).mean().fillna(20.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['sma20'] = sma20
    df['pct_b'] = (close - (sma20 - 2 * std20)) / ((4 * std20) + 1e-8)
    df['vol_ratio'] = (volume / volume.rolling(20).mean().where(lambda x: x > 0)).fillna(1.0)
    df['ema9'] = close.ewm(span=9, adjust=False).mean()

    return df.dropna().reset_index(drop=True)

df_clean = process_data(df_candles)

# ==========================================
# 3. TABBLADEN & BACKTEST ENGINE
# ==========================================
tab_v20, tab_twm = st.tabs(["🛡️ V20 Trend-Pullback Backtest", "🚀 TWM Mean-Reversion Backtest"])

with tab_v20:
    st.subheader(f"V20 Strategie Resultaten {f'({selected_asset})' if selected_asset else ''}")
    
    def run_v20(df, adx_min, sl_mult, tp_mult, cap):
        d = df.copy()
        long_cond = (d['adx'] >= adx_min) & (d['macd_hist'] > 0) & (d['close'] >= d['ema9'] * 0.992)
        short_cond = (d['adx'] >= adx_min) & (d['macd_hist'] < 0) & (d['close'] <= d['ema9'] * 1.008)
        
        d['signal'] = 0
        d.loc[long_cond & ~short_cond, 'signal'] = 1
        d.loc[short_cond & ~long_cond, 'signal'] = -1

        equity = [cap]
        position, entry, sl, tp = 0, 0, 0, 0
        trades = []

        for i in range(1, len(d)):
            price = d['close'].iloc[i]
            atr = d['atr'].iloc[i]
            sig = d['signal'].iloc[i-1]

            if position == 0 and sig != 0:
                position = sig
                entry = price
                sl = entry - position * sl_mult * atr
                tp = entry + position * tp_mult * atr
            elif position != 0:
                hit_sl = (position == 1 and price <= sl) or (position == -1 and price >= sl)
                hit_tp = (position == 1 and price >= tp) or (position == -1 and price <= tp)
                if hit_sl or hit_tp:
                    trades.append({'pnl': (price - entry)/entry * position})
                    position = 0

            eq = equity[-1]
            if position != 0:
                eq += eq * (((price - d['close'].iloc[i-1]) / d['close'].iloc[i-1]) * position * 0.02)
            equity.append(eq)
        d['equity'] = equity[:len(d)]
        return d, trades

    res_v20, trades_v20 = run_v20(df_clean, min_adx, meta_sl, meta_tp, capital)
    
    col1, col2, col3, col4 = st.columns(4)
    final_eq = res_v20['equity'].iloc[-1]
    ret_pct = ((final_eq - capital) / capital) * 100
    win_rate = (len([t for t in trades_v20 if t['pnl'] > 0]) / len(trades_v20) * 100) if trades_v20 else 0.0
    max_dd = (res_v20['equity'] / res_v20['equity'].cummax() - 1).min() * 100

    col1.metric("Eindkapitaal", f"${final_eq:,.2f}", f"{ret_pct:.2f}%")
    col2.metric("Totaal Trades", len(trades_v20))
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Max Drawdown", f"{max_dd:.2f}%")
    
    st.plotly_chart(px.line(res_v20, x="timestamp", y="equity", title="V20 Vermogensverloop (Equity Curve)"), use_container_width=True)

with tab_twm:
    st.subheader(f"TWM Sniper Resultaten {f'({selected_asset})' if selected_asset else ''}")
    
    def run_twm(df, trail_m, cap):
        d = df.copy()
        twm_long = (d['pct_b'] < -0.02) & (d['rsi'] < 25.0) & (d['vol_ratio'] > 1.4)
        twm_short = (d['pct_b'] > 1.02) & (d['rsi'] > 75.0) & (d['vol_ratio'] > 1.4)
        
        d['signal'] = 0
        d.loc[twm_long, 'signal'] = 1
        d.loc[twm_short, 'signal'] = -1

        equity = [cap]
        position, entry, target, sl = 0, 0, 0, 0
        trades = []

        for i in range(1, len(d)):
            price = d['close'].iloc[i]
            atr = d['atr'].iloc[i]
            sma = d['sma20'].iloc[i]
            sig = d['signal'].iloc[i-1]

            if position == 0 and sig != 0:
                position = sig
                entry = price
                target = sma
                sl = entry - position * (trail_m * atr)
            elif position != 0:
                hit_sl = (position == 1 and price <= sl) or (position == -1 and price >= sl)
                hit_tp = (position == 1 and price >= target) or (position == -1 and price <= target)
                if hit_sl or hit_tp:
                    trades.append({'pnl': (price - entry)/entry * position})
                    position = 0

            eq = equity[-1]
            if position != 0:
                eq += eq * (((price - d['close'].iloc[i-1]) / d['close'].iloc[i-1]) * position * 0.02)
            equity.append(eq)
        d['equity'] = equity[:len(d)]
        return d, trades

    res_twm, trades_twm = run_twm(df_clean, trailing_mult, capital)
    
    col1, col2, col3, col4 = st.columns(4)
    final_eq_twm = res_twm['equity'].iloc[-1]
    ret_twm = ((final_eq_twm - capital) / capital) * 100
    win_rate_twm = (len([t for t in trades_twm if t['pnl'] > 0]) / len(trades_twm) * 100) if trades_twm else 0.0
    max_dd_twm = (res_twm['equity'] / res_twm['equity'].cummax() - 1).min() * 100

    col1.metric("Eindkapitaal", f"${final_eq_twm:,.2f}", f"{ret_twm:.2f}%")
    col2.metric("Totaal Trades", len(trades_twm))
    col3.metric("Win Rate", f"{win_rate_twm:.1f}%")
    col4.metric("Max Drawdown", f"{max_dd_twm:.2f}%")
    
    st.plotly_chart(px.line(res_twm, x="timestamp", y="equity", title="TWM Vermogensverloop (Equity Curve)"), use_container_width=True)
