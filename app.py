import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import ccxt

# Pagina configuratie
st.set_page_config(
    page_title="RV Predictor",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RV Predictor")
st.markdown("*Its magic*")

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
    """Haalt de marktdata rechtstreeks op via KuCoin (CCXT)."""
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
        
        # Technische indicatoren voor de RSI metric
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

# State bestand naam volgt het formaat: model_state_{asset}_{timescale}.json
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
            
            if predicted_price is not None:
                signal = state.get('signal')                 # None = state in oud formaat
                last_close = state.get('last_close') or current_price
                pred_ret_pct = state.get('predicted_return_pct')
                if pred_ret_pct is None:
                    pred_ret_pct = (predicted_price - last_close) / last_close * 100
                acc = state.get('holdout_direction_accuracy')
                skill = state.get('baseline_error')

                # --- Eén duidelijke conclusie bovenaan ---
                direction_txt = "stijging" if pred_ret_pct > 0 else "daling"
                if signal == 'long':
                    st.success(f"📈 LONG-signaal: prijsmodel ({pred_ret_pct:+.3f}%) en kansmodel ({prob:.1f}% stijging) wijzen allebei omhoog.")
                elif signal == 'short':
                    st.error(f"📉 SHORT-signaal: prijsmodel ({pred_ret_pct:+.3f}%) en kansmodel ({100 - prob:.1f}% daling) wijzen allebei omlaag.")
                elif signal == 'none':
                    st.warning(f"⏸ Geen eenduidig signaal: het prijsmodel verwacht een {direction_txt} ({pred_ret_pct:+.3f}%), "
                               f"maar het kansmodel geeft {prob:.1f}% kans op stijging. De twee zijn het oneens, dus er wordt niet gehandeld.")
                else:
                    st.info("State in oud formaat: wacht op de eerstvolgende run van de worker.")

                # Hoofdmetrics rij 1
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Huidige Koers (KuCoin, live)", f"${current_price:,.2f}")
                col2.metric(f"Voorspelling volgende {interval_code}-candle", f"${predicted_price:,.2f}",
                            delta=f"{pred_ret_pct:+.3f}% t.o.v. laatste close", delta_color="normal")
                col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
                col4.metric("Adaptieve LR", f"{lr:.4f}" if isinstance(lr, float) else f"{lr}")
                st.caption(f"De voorspelling geldt t.o.v. de laatste gesloten candle (${last_close:,.2f}), niet t.o.v. de live koers.")

                # SL/TP alleen tonen als er een signaal is; anders zouden ze een richting suggereren die er niet is
                show_levels = signal in ('long', 'short') or signal is None
                sl_txt = f"${sl:,.2f}" if (show_levels and sl > 0) else "N.v.t."
                tp_txt = f"${tp:,.2f}" if (show_levels and tp > 0) else "N.v.t."

                # Extra metrics rij 2
                col5, col6, col7, col8 = st.columns(4)
                prob_color = "📈" if prob >= 50 else "📉"
                col5.metric(f"{prob_color} Kans op stijging (kansmodel)", f"{prob:.1f}%",
                            help="Ruwe uitkomst van de classifier. Niet gekalibreerd: vergelijk met de holdout-nauwkeurigheid hieronder.")
                col6.metric("Lookback-venster", f"{lookback} candles")
                col7.metric("Stop-Loss (SL)", sl_txt)
                col8.metric("Take-Profit (TP)", tp_txt)

                # Validatie & handelssignaal (out-of-sample holdout)
                col9, col10, col11 = st.columns(3)
                col9.metric("Signaal", {'long': '📈 LONG', 'short': '📉 SHORT'}.get(signal, '⏸ Geen'))
                col10.metric("Holdout richtingsnauwkeurigheid", f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else "n.v.t.")
                col11.metric("Skill (<1 = beter dan 'geen beweging')", f"{skill:.2f}" if isinstance(skill, (int, float)) else "n.v.t.")
                if isinstance(skill, (int, float)) and skill >= 1:
                    st.caption("⚠️ Skill ≥ 1: op de testperiode voorspelt het model niet beter dan 'de prijs blijft gelijk'.")
            else:
                st.warning("Model state gevonden, maar nog geen voorspelling beschikbaar.")
                
            st.markdown("---")
            
            # Model details uitklapmenu
            with st.expander("📊 Bekijk Model & State Details (Weging, Fouten & Parameters)"):
                st.json(state)
                
        except Exception as e:
            st.error(f"Fout bij het inlezen van state-bestand: {e}")
    else:
        st.warning(f"Model state-bestand `{state_file}` nog niet gevonden. Wacht op de eerste run van de achtergrond-worker.")
else:
    st.error("Kon geen marktdata ophalen via KuCoin.")
