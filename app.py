import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import ccxt

# Pagina configuratie
st.set_page_config(
    page_title="RV Quant & Trader Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RV Institutional Quant Dashboard")
st.markdown("*Adaptive Machine Learning Monitor & Live Model State*")

# Sidebar: Asset keuze & Tijdinterval (aangepast op quant_worker timeframes)
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

@st.cache_data(ttl=60)
def load_dashboard_data(symbol, tf_info):
    """Haalt de actuele marktdata rechtstreeks op via KuCoin (CCXT) voor de RSI en koerscontext."""
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

# Haal live marktdata op
symbol = f"{asset_code}/USDT"
df = load_dashboard_data(symbol, interval_info)

# State bestand naam volgt het exacte formaat van quant_worker: model_state_{asset}_{timescale}.json
state_file = f"model_state_{asset_code.lower()}_{interval_code}.json"

if df is not None and not df.empty:
    current_price = float(df['close'].iloc[-1])
    
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
            signal = state.get('signal', 'none')
            
            # Richtingslogica & correcte SL/TP afstemming op basis van het model-signaal
            is_long = signal == 'long' or (predicted_price is not None and predicted_price > current_price)
            
            # Automatische validatie tegen tegenspraak:
            # Als signaal 'long' is, moet TP boven koers en SL onder koers liggen. Bij 'short' andersom.
            if tp > 0 and sl > 0:
                if is_long and tp < current_price:
                    tp, sl = sl, tp
                elif not is_long and tp > current_price:
                    tp, sl = sl, tp

            if predicted_price is not None:
                pct_diff = ((predicted_price - current_price) / current_price) * 100
                
                # --- RIJ 1: Kernkoersen & Modelvoorspelling ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Huidige Koers (KuCoin)", f"${current_price:,.2f}")
                col2.metric(f"Ensemble Voorspelling ({interval_code})", f"${predicted_price:,.2f}", delta=f"{pct_diff:+.2f}%")
                col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
                col4.metric("Adaptieve LR", f"{lr:.4f}" if isinstance(lr, float) else f"{lr}")

                # --- RIJ 2: Kansen & Risicobeheer (ATR gebaseerd vanuit workers) ---
                col5, col6, col7, col8 = st.columns(4)
                prob_color = "📈" if prob >= 50 else "📉"
                col5.metric(f"{prob_color} Kans op stijging", f"{prob:.1f}%")
                col6.metric("Lookback-venster", f"{lookback} candles")
                col7.metric("Stop-Loss (SL)", f"${sl:,.2f}" if sl > 0 else "N.v.t.")
                col8.metric("Take-Profit (TP)", f"${tp:,.2f}" if tp > 0 else "N.v.t.")

                # --- RIJ 3: Handelsstatistieken & Out-of-sample Holdout Validatie ---
                acc = state.get('holdout_direction_accuracy')
                skill = state.get('baseline_error')
                regime = state.get('current_regime', 'sideways')
                
                col9, col10, col11, col12 = st.columns(4)
                signal_display = {'long': '📈 LONG', 'short': '📉 SHORT'}.get(signal, '⏸ Geen')
                col9.metric("Model Signaal", signal_display)
                col10.metric("Marktregime", regime.capitalize())
                col11.metric("Holdout Richtingsnauwkeurigheid", f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else "n.v.t.")
                col12.metric("Baseline Fout (Skill)", f"{skill:.2f}" if isinstance(skill, (int, float)) else "n.v.t.")
            else:
                st.warning("Model state gevonden, maar nog geen voorspelling beschikbaar.")
                
            st.markdown("---")
            
            # Model details & ensemble gewichten uitklapmenu
            with st.expander("📊 Bekijk Model & State Details (Weging, Fouten & Parameters)"):
                st.subheader("Ensemble Model Gewichten (XGBoost / CatBoost / Random Forest)")
                weights = state.get('model_weights', {})
                if weights:
                    df_weights = pd.DataFrame(list(weights.items()), columns=['Model', 'Gewicht'])
                    st.bar_chart(df_weights.set_index('Model'))
                st.json(state)
                
        except Exception as e:
            st.error(f"Fout bij het inlezen van state-bestand: {e}")
    else:
        st.warning(f"Model state-bestand `{state_file}` nog niet gevonden. Wacht op de eerste run van de quant_worker.")
else:
    st.error("Kon geen marktdata ophalen via KuCoin.")
