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
st.markdown("*Real-time ML Model Monitoring, Marktcontext & Signal Synchronization*")

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

# State bestand naam volgt het exacte formaat van de workers: model_state_{asset}_{timescale}.json
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
            regime = state.get('current_regime', 'sideways')
            
            if predicted_price is not None:
                signal = state.get('signal')
                last_close = state.get('last_close') or current_price
                pred_ret_pct = state.get('predicted_return_pct')
                if pred_ret_pct is None:
                    pred_ret_pct = (predicted_price - last_close) / last_close * 100
                acc = state.get('holdout_direction_accuracy')
                skill = state.get('baseline_error')

                # --- SAMENVATTING & TRADER-SYNCHRONISATIE STATUS ---
                direction_txt = "stijging" if pred_ret_pct > 0 else "daling"
                if signal == 'long':
                    st.success(f"📈 LONG-signaal actief: prijsmodel ({pred_ret_pct:+.3f}%) en kansmodel ({prob:.1f}% stijging) zijn in lijn. De trader-worker zoekt naar instapmogelijkheden.")
                elif signal == 'short':
                    st.error(f"📉 SHORT-signaal actief: prijsmodel ({pred_ret_pct:+.3f}%) en kansmodel ({100 - prob:.1f}% daling) wijzen omlaag. De trader-worker zoekt naar short-posities.")
                elif signal == 'none':
                    st.warning(f"⏸ Geen handelssignaal: het prijsmodel verwacht een {direction_txt} ({pred_ret_pct:+.3f}%), "
                               f"maar het kansmodel geeft {prob:.1f}% kans. Regressie en classifier spreken elkaar tegen (Trader slaat deze ronde over).")
                else:
                    st.info("State in oud formaat: wacht op de eerstvolgende run van de quant_worker.")

                # --- RIJ 1: Kernkoersen & Voorspelling ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Huidige Koers (KuCoin, live)", f"${current_price:,.2f}")
                col2.metric(f"Voorspelling volgende {interval_code}-candle", f"${predicted_price:,.2f}",
                            delta=f"{pred_ret_pct:+.3f}% t.o.v. laatste close", delta_color="normal")
                col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
                col4.metric("Adaptieve LR", f"{lr:.4f}" if isinstance(lr, float) else f"{lr}")
                st.caption(f"Voorspelling is berekend t.o.v. de laatste gesloten candle (${last_close:,.2f}), niet de live tick-koers.")

                # SL/TP weergave controleren op basis van het actieve signaal
                show_levels = signal in ('long', 'short')
                sl_txt = f"${sl:,.2f}" if (show_levels and sl > 0) else "N.v.t. (Geen signaal)"
                tp_txt = f"${tp:,.2f}" if (show_levels and tp > 0) else "N.v.t. (Geen signaal)"

                # --- RIJ 2: Kansen & ATR gebaseerde Stop-Loss / Take-Profit ---
                col5, col6, col7, col8 = st.columns(4)
                prob_color = "📈" if prob >= 50 else "📉"
                col5.metric(f"{prob_color} Kans op stijging", f"{prob:.1f}%")
                col6.metric("Lookback-venster", f"{lookback} candles")
                col7.metric("Stop-Loss (SL - 1.5x ATR)", sl_txt)
                col8.metric("Take-Profit (TP - 2.5x ATR)", tp_txt)

                # --- RIJ 3: Validatie, Regime & Model Skill ---
                col9, col10, col11, col12 = st.columns(4)
                col9.metric("Signaal Status", {'long': '📈 LONG', 'short': '📉 SHORT'}.get(signal, '⏸ Geen'))
                col10.metric("Marktregime", regime.capitalize())
                col11.metric("Holdout Nauwkeurigheid", f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else "n.v.t.")
                col12.metric("Skill (<1 = beter dan baseline)", f"{skill:.2f}" if isinstance(skill, (int, float)) else "n.v.t.")
                
                if isinstance(skill, (int, float)) and skill >= 1:
                    st.caption("⚠️ Let op: Skill ≥ 1 betekent dat het model op de out-of-sample testperiode niet beter presteert dan 'geen beweging'.")
            else:
                st.warning("Model state gevonden, maar nog geen voorspelling beschikbaar.")
                
            st.markdown("---")
            
           # --- UITKLAPMENU: Uitgebreide Model Details & Marktcontext ---
            with st.expander("📊 Bekijk Model State, Ensemble Gewichten & Fouten (JSON)"):
                st.subheader("Ensemble Model Gewichten (XGBoost / CatBoost / Random Forest)")
                weights = state.get('model_weights', {})
                if weights:
                    df_weights = pd.DataFrame(list(weights.items()), columns=['Model', 'Gewicht'])
                    st.bar_chart(df_weights.set_index('Model'))
                
                st.subheader("Volledige State JSON")
                st.json(state)
                
        except Exception as e:
            st.error(f"Fout bij het inlezen van state-bestand: {e}")
    else:
        st.warning(f"Model state-bestand `{state_file}` nog niet gevonden. Wacht op de eerste run van de quant_worker.")
else:
    st.error("Kon geen marktdata ophalen via KuCoin.")
