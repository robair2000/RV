import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import ccxt

# Pagina configuratie
st.set_page_config(
    page_title="RV Institutional Quant & Trader Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RV Institutional Quant & Trader Hub")
st.markdown("*Real-time ML Model Monitoring, Final Boss Gatekeeper & Signal Synchronization*")

# Sidebar: Asset keuze & Tijdinterval
asset_options = {
    'Bitcoin (BTC)': 'BTC',
    'Ethereum (ETH)': 'ETH',
    'Chainlink (LINK)': 'LINK',
}

interval_options = {
    '15 Minuten': {'interval': '15m', 'limit': 500, 'code': '15m'},
    '1 Uur': {'interval': '1h', 'limit': 500, 'code': '1h'},
    '1 Dag': {'interval': '1d', 'limit': 365, 'code': '1d'},
    '1 Week': {'interval': '1w', 'limit': 100, 'code': '1w'},
    '1 Maand': {'interval': '1M', 'limit': 60, 'code': '1m'}
}

selected_asset_name = st.sidebar.selectbox("Selecteer Asset", list(asset_options.keys()))
selected_interval_name = st.sidebar.selectbox("Selecteer Tijdinterval", list(interval_options.keys()))

asset_code = asset_options[selected_asset_name]
interval_info = interval_options[selected_interval_name]
interval_code = interval_info['code']

@st.cache_data(ttl=300)
def load_dashboard_data(symbol, tf_info):
    """Haalt de marktdata rechtstreeks op via KuCoin (CCXT) voor technische context."""
    try:
        exchange = ccxt.kucoin({
            'enableRateLimit': True,
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_info['interval'], limit=tf_info['limit'])
        if not ohlcv:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop(columns=['timestamp'], inplace=True)
        df.set_index('Date', inplace=True)
        
        # Technische RSI indicator berekening
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    except Exception as e:
        st.error(f"Fout bij ophalen KuCoin data: {e}")
        return None

# Haal live data op voor actuele koers en RSI via KuCoin
symbol = f"{asset_code}/USDT"
df = load_dashboard_data(symbol, interval_info)

# State bestanden
state_file = f"model_state_{asset_code.lower()}_{interval_code}.json"
pattern_file = f"pattern_state_{asset_code.lower()}_15m.json"

if df is not None and not df.empty:
    current_price = float(df['close'].iloc[-1])
    
    # --- LAAD FINAL BOSS STATE ---
    boss_state = {}
    if os.path.exists(pattern_file):
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                boss_state = json.load(f)
        except Exception:
            pass

    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            predicted_price = state.get('last_prediction')
            lr = state.get('learning_rate', 0.05)
            prob = state.get('direction_probability', 50.0)
            lookback = state.get('lookback_window', 150)
            sl = state.get('stop_loss', 0.0)
            tp = state.get('take_profit', 0.0)
            regime = state.get('current_regime', 'sideways')
            
            if predicted_price is not None:
                signal = state.get('signal')
                last_close = state.get('last_close') or current_price
                pred_ret_pct = state.get('predicted_return_pct')
                if pred_ret_pct is None:
                    pred_ret_pct = (predicted_price - last_close) / last_close * 100
                acc = state.get('holdout_direction_accuracy')
                skill = state.get('baseline_error')

                # --- FINAL BOSS DATA EXTRACTIE ---
                boss_signal = boss_state.get('pattern_signal', 'none')
                boss_score = boss_state.get('pattern_score', 50.0)
                min_score_req = boss_state.get('min_score_required', 75.0)
                strictness = boss_state.get('strictness_mode', 'strict')
                rec_leverage = boss_state.get('recommended_leverage', 1)
                is_squeeze = boss_state.get('bollinger_squeeze', False)

                # --- SAMENVATTING & TRADER-SYNCHRONISATIE STATUS ---
                st.markdown("### 👑 Final Boss & Trader Status")
                if boss_signal == signal and boss_score >= min_score_req:
                    st.success(f"✅ **GOEDKEURING:** Final Boss en Quant Model zijn in lijn ({signal.upper()}). Score: {boss_score} (Vereist: {min_score_req}). De trader mag handelen met {rec_leverage}x hefboom!")
                elif boss_signal != signal:
                    st.warning(f"⚠️ **VETO (Richting):** Final Boss ziet '{boss_signal.upper()}' terwijl Quant model '{signal.upper()}' aangeeft. De trader houdt zich koest.")
                else:
                    st.info(f"⏸ **VETO (Score):** Signaal komt overeen, maar de score ({boss_score}) is te laag voor de huidige '{strictness}' modus (vereist: {min_score_req}).")

                # --- RIJ 1: Kernkoersen & Voorspelling ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Huidige Koers (KuCoin, live)", f"${current_price:,.2f}")
                col2.metric(f"Voorspelling volgende {interval_code}-candle", f"${predicted_price:,.2f}",
                            delta=f"{pred_ret_pct:+.3f}% t.o.v. laatste close", delta_color="normal")
                col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
                col4.metric("Final Boss Score", f"{boss_score} / 100", delta=f"Min. vereist: {min_score_req}")

                # --- RIJ 2: Final Boss Regimes & Hefboom ---
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Boss Modus", strictness.capitalize())
                col6.metric("Aanbevolen Hefboom", f"{rec_leverage}x")
                col7.metric("Bollinger Squeeze", "🔥 Actief" if is_squeeze else "Rustig")
                col8.metric("Kans op stijging (Quant)", f"{prob:.1f}%")

                # --- RIJ 3: Stop-Loss, Take-Profit & Model Details ---
                show_levels = signal in ('long', 'short')
                sl_txt = f"${sl:,.2f}" if (show_levels and sl > 0) else "N.v.t. (Geen signaal)"
                tp_txt = f"${tp:,.2f}" if (show_levels and tp > 0) else "N.v.t. (Geen signaal)"

                col9, col10, col11, col12 = st.columns(4)
                col9.metric("Stop-Loss (SL - 1.5x ATR)", sl_txt)
                col10.metric("Take-Profit (TP - 2.5x ATR)", tp_txt)
                col11.metric("Marktregime", regime.capitalize())
                col12.metric("Skill (<1 = beter dan baseline)", f"{skill:.2f}" if isinstance(skill, (int, float)) else "n.v.t.")
                
                if isinstance(skill, (int, float)) and skill >= 1:
                    st.caption("⚠️ Let op: Skill ≥ 1 betekent dat het model op de out-of-sample testperiode niet beter presteert dan 'geen beweging'.")
            else:
                st.warning("Model state gevonden, maar nog geen voorspelling beschikbaar.")
                
            st.markdown("---")
            
           # --- UITKLAPMENU: Uitgebreide Model Details & Marktcontext ---
            with st.expander("📊 Bekijk Model State, Final Boss JSON & Ensemble Gewichten"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Final Boss State (Pattern JSON)")
                    st.json(boss_state)
                with col_b:
                    st.subheader("Quant Model State (JSON)")
                    st.json(state)
                
                st.subheader("Ensemble Model Gewichten (XGBoost / CatBoost / Random Forest)")
                weights = state.get('model_weights', {})
                if weights:
                    df_weights = pd.DataFrame(list(weights.items()), columns=['Model', 'Gewicht'])
                    st.bar_chart(df_weights.set_index('Model'))
                
        except Exception as e:
            st.error(f"Fout bij het inlezen van state-bestanden: {e}")
    else:
        st.warning(f"Model state-bestand `{state_file}` nog niet gevonden. Wacht op de eerste run van de quant_worker.")
else:
    st.error("Kon geen marktdata ophalen via KuCoin.")
