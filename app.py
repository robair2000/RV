import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
import urllib.request

# Pagina configuratie
st.set_page_config(
    page_title="RV Institutional Quant & Trader Dashboard (V20 & TWM)",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RV Institutional S&R / S&D Swing Hub (V20 & TWM Sniper)")
st.markdown("*Real-time ML Meta-Labeling, S&R Confluence Gatekeeper & Live Reversal Sniper Status*")

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

def load_json_from_github_or_local(filename):
    """Probeert het JSON-bestand direct vers van GitHub op te halen, valt anders terug op lokaal."""
    token = os.environ.get('GITHUB_TOKEN', '')
    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
    headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'v20-dashboard'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

# --- DYNAMISCHE ASSET SELECTIE UIT RANKING CSV ---
def get_dashboard_trading_assets(top_n=5):
    core_assets = ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK']
    ranking_file = 'kucoin_best_coins_ranking.csv'
    dynamic_assets = []
    
    try:
        url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{ranking_file}?ref={GITHUB_BRANCH}'
        token = os.environ.get('GITHUB_TOKEN', '')
        headers = {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'v20-dashboard'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            with open(ranking_file, 'wb') as f:
                f.write(resp.read())
    except Exception:
        pass

    if os.path.exists(ranking_file):
        try:
            df = pd.read_csv(ranking_file)
            symbols = df['Symbol'].head(top_n).tolist()
            for s in symbols:
                asset_name = s.split('/')[0].split(':')[0]
                if asset_name not in dynamic_assets:
                    dynamic_assets.append(asset_name)
        except Exception:
            pass
            
    combined = list(dict.fromkeys(core_assets + dynamic_assets))
    options = {}
    for asset in combined:
        label = f"{asset} (Kern)" if asset in core_assets else f"{asset} (Scanner Top {top_n})"
        options[label] = asset
    return options

asset_options = get_dashboard_trading_assets(top_n=5)

interval_options = {
    '4 Uur (Macro Swing)': {'interval': '4h', 'limit': 500, 'code': '4h'},
    '1 Dag (Trend)': {'interval': '1d', 'limit': 200, 'code': '1d'},
}

selected_label = st.sidebar.selectbox("Selecteer Asset", list(asset_options.keys()))
selected_interval_name = st.sidebar.selectbox("Selecteer Tijdinterval", list(interval_options.keys()))

asset_code = asset_options[selected_label]
interval_info = interval_options[selected_interval_name]
interval_code = interval_info['code']

@st.cache_data(ttl=300)
def load_dashboard_data(symbol, tf_info):
    try:
        exchange_client = ccxt.kucoin({'enableRateLimit': True})
        ohlcv = exchange_client.fetch_ohlcv(symbol, timeframe=tf_info['interval'], limit=tf_info['limit'])
        if not ohlcv:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop(columns=['timestamp'], inplace=True)
        df.set_index('Date', inplace=True)
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    except Exception as e:
        st.error(f"Fout bij ophalen KuCoin data: {e}")
        return None

@st.cache_resource
def get_exchange_client():
    exch = ccxt.kucoin({
        'apiKey': os.getenv('KUCOIN_API_KEY', ''),
        'secret': os.getenv('KUCOIN_API_SECRET', ''),
        'password': os.getenv('KUCOIN_API_PASSWORD', ''),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    try:
        exch.load_markets()
    except Exception:
        pass
    return exch

exchange = get_exchange_client()

# --- MAAK DRIE TABS AAN ---
tab1, tab2, tab3 = st.tabs([
    "🚀 Dashboard & Live Trades", 
    "⚙️ Systeem & Worker Status", 
    "🎯 TWM Reversal Sniper"
])

with tab1:
    st.markdown("### 💼 Live KuCoin Futures Posities (Handmatig & Automatisch)")
    try:
        positions = exchange.fetch_positions()
        active_positions = [p for p in positions if abs(float(p.get('contracts') or 0)) > 0]
        
        if active_positions:
            pos_data = []
            for p in active_positions:
                sym = p.get('symbol')
                side = str(p.get('side')).upper()
                contracts = float(p.get('contracts', 0))
                entry_price = float(p.get('entryPrice', 0))
                mark_price = float(p.get('markPrice', 0))
                pnl = float(p.get('unrealizedPnl', 0))
                pnl_pct = float(p.get('percentage', 0)) if 'percentage' in p and p.get('percentage') is not None else 0.0
                leverage = p.get('leverage', 1)
                margin_mode = p.get('marginMode', 'cross')
                
                pos_data.append({
                    'Symbool': sym,
                    'Richting': side,
                    'Contracten': contracts,
                    'Entry Prijs ($)': round(entry_price, 4),
                    'Markt Prijs ($)': round(mark_price, 4),
                    'Hefboom': f"{leverage}x",
                    'Modus': margin_mode,
                    'Ongerealiseerde PnL ($)': round(pnl, 2),
                    'PnL (%)': f"{pnl_pct:+.2f}%"
                })
            
            df_pos = pd.DataFrame(pos_data)
            st.dataframe(df_pos, width="stretch")
        else:
            st.info("ℹ️ Geen actieve openstaande posities gevonden op je KuCoin Futures account op dit moment.")
    except Exception as e:
        st.warning(f"⚠️ Kon live posities niet ophalen van KuCoin: {e}")

    st.markdown("---")

    symbol = f"{asset_code}/USDT"
    df = load_dashboard_data(symbol, interval_info)

    state_file = f"model_state_{asset_code.lower()}_{interval_code}.json"
    pattern_file = f"pattern_state_{asset_code.lower()}_4h.json"

    if df is not None and not df.empty:
        current_price = float(df['close'].iloc[-1])
        
        boss_state = load_json_from_github_or_local(pattern_file)
        state = load_json_from_github_or_local(state_file)

        predicted_price = state.get('last_prediction')
        prob = state.get('direction_probability', 50.0)
        sl = state.get('stop_loss', 0.0)
        tp = state.get('take_profit', 0.0)
        regime = state.get('current_regime', 'trending')
        
        if predicted_price is not None:
            signal = state.get('signal')
            pred_ret_pct = ((predicted_price - current_price) / current_price) * 100
            skill = state.get('baseline_error')

            boss_signal = boss_state.get('pattern_signal', 'none')
            boss_score = boss_state.get('pattern_score', 50.0)
            min_score_req = boss_state.get('min_score_required', 75.0)
            strictness = boss_state.get('strictness_mode', 'v20_sr_sd_confluence')
            rec_leverage = boss_state.get('recommended_leverage', 2)

            st.markdown(f"### 👑 V20 S&R Confluence & Model Status voor {asset_code}")
            if boss_signal == signal and boss_score >= min_score_req:
                st.success(f"✅ **GOEDKEURING:** S&R Confluence Agent en Quant Model zijn in lijn ({signal.upper()}). Score: {boss_score} (Vereist: {min_score_req}).")
            elif boss_signal != signal:
                st.warning(f"⚠️ **VETO (Richting):** S&R Agent ziet '{boss_signal.upper()}' terwijl Quant model '{signal.upper()}' aangeeft.")
            else:
                st.info(f"⏸ **VETO (Score):** Signaal komt overeen, maar de score ({boss_score}) is te laag (vereist: {min_score_req}).")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Huidige Koers (KuCoin, live)", f"${current_price:,.2f}")
            col2.metric(f"Voorspelling volgende {interval_code}-candle", f"${predicted_price:,.2f}",
                        delta=f"{pred_ret_pct:+.3f}% t.o.v. live koers", delta_color="normal")
            col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
            col4.metric("V20 Confluence Score", f"{boss_score} / 100", delta=f"Min. vereist: {min_score_req}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Strategie Modus", strictness.replace('_', ' ').capitalize())
            col6.metric("Aanbevolen Hefboom", f"{rec_leverage}x")
            col7.metric("S&R Trend Richting", boss_signal.upper())
            col8.metric("Kans op stijging (Quant)", f"{prob:.1f}%")

            show_levels = signal in ('long', 'short')
            sl_txt = f"${sl:,.2f}" if (show_levels and sl > 0) else "N.v.t. (Geen signaal)"
            tp_txt = f"${tp:,.2f}" if (show_levels and tp > 0) else "N.v.t. (Geen signaal)"

            col9, col10, col11, col12 = st.columns(4)
            col9.metric("Model SL (2.5x ATR)", sl_txt)
            col10.metric("Model TP (5.0x ATR)", tp_txt)
            col11.metric("Marktregime", regime.capitalize())
            col12.metric("Skill (<1 = beter dan baseline)", f"{skill:.2f}" if isinstance(skill, (int, float)) else "n.v.t.")

        else:
            st.warning(f"Model state gevonden voor `{asset_code}`, maar wacht op de eerste geldige voorspelling...")
            
        st.markdown("---")
        
        with st.expander("📊 Bekijk Model State, S&R Pattern JSON & Ensemble Gewichten"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("S&R & S&D Pattern State (JSON)")
                st.json(boss_state)
            with col_b:
                st.subheader("Quant Model State (JSON)")
                st.json(state)
            
            st.subheader("Ensemble Model Gewichten (XGBoost / CatBoost / Random Forest)")
            weights = state.get('model_weights', {})
            if weights:
                df_weights = pd.DataFrame(list(weights.items()), columns=['Model', 'Gewicht'])
                st.bar_chart(df_weights.set_index('Model'))
    else:
        st.error("Kon geen marktdata ophalen via KuCoin.")

with tab2:
    st.markdown("### ⚙️ Systeem Status & Worker Logs (GitHub & Bot Sync)")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("#### 🤖 Trader Worker Status (`trader_status.json`)")
        t_status = load_json_from_github_or_local("trader_status.json")
        if t_status:
            up_time = t_status.get('updated_at', 0)
            if up_time > 0:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(up_time))
                st.caption(f"Laatst bijgewerkt: {time_str}")
            
            active_t = t_status.get('active_trades', {})
            if active_t:
                st.warning(f"🚨 **Bot Beheerde Trades / Limit Orders:** {len(active_t)} openstaand")
                st.json(active_t)
            else:
                st.info("ℹ️ Geen actieve bot-trades of open zone-limit orders op dit moment.")
            
            cooldowns = t_status.get('cooldowns', {})
            if cooldowns:
                st.markdown("**Actieve Cooldowns:**")
                st.json(cooldowns)
        else:
            st.info("⏳ Nog geen trader status bestand beschikbaar.")

    with col_s2:
        st.markdown("#### 🏆 Markt Scanner Ranking (`kucoin_best_coins_ranking.csv`)")
        ranking_file = 'kucoin_best_coins_ranking.csv'
        if os.path.exists(ranking_file):
            try:
                df_ranking = pd.read_csv(ranking_file)
                st.dataframe(df_ranking, width="stretch")
            except Exception as e:
                st.error(f"Fout bij inlezen ranking CSV: {e}")
        else:
            st.warning("⏳ Ranking CSV nog niet beschikbaar.")

    st.markdown("---")
    st.markdown("#### 🔄 Worker Synchronisatie & GitHub Status")
    st.info("Alle workers synchroniseren direct via GitHub. Het dashboard haalt nu altijd de meest recente data op.")

with tab3:
    st.markdown("### 🎯 TWM Reversal Sniper Status (`trader_status_momentum.json`)")
    st.markdown("*Real-time monitor voor de Explosive Reversal Hunter (Volume-expansie & Bollinger Squeeze S&R Bounces)*")
    
    twm_status = load_json_from_github_or_local("trader_status_momentum.json")
    if twm_status:
        up_time = twm_status.get('updated_at', 0)
        if up_time > 0:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(up_time))
            st.caption(f"TWM Laatst bijgewerkt: {time_str}")
        
        twm_active = twm_status.get('active_trades', {})
        if twm_active:
            st.success(f"🚀 **Actieve TWM Reversal Sniper Posities:** {len(twm_active)}")
            
            # Converteer de active_trades dictionary naar een mooie overzichtstabel met SL en TP
            sniper_rows = []
            for sym, details in twm_active.items():
                sniper_rows.append({
                    'Symbool': sym,
                    'Richting': str(details.get('side', '')).upper(),
                    'Entry Prijs ($)': round(float(details.get('entry', 0)), 4),
                    'Stop Loss (SL) ($)': round(float(details.get('sl', 0)), 4),
                    'Take Profit (TP) ($)': round(float(details.get('tp', 0)), 4),
                    'Peak Prijs ($)': round(float(details.get('peak', 0)), 4),
                    'Breakeven Beveiligd': 'Ja' if details.get('breakeven_secured', False) else 'Nee'
                })
            
            df_sniper = pd.DataFrame(sniper_rows)
            st.dataframe(df_sniper, width="stretch")
        else:
            st.info("ℹ️ Geen actieve TWM sniper posities op dit moment. De bot scant actief op volume-uitbraken en S&R squeezes.")
        
        twm_cooldowns = twm_status.get('cooldowns', {})
        if twm_cooldowns:
            st.markdown("**Actieve TWM Cooldowns:**")
            st.json(twm_cooldowns)
    else:
        st.warning("⏳ Nog geen TWM status bestand (`trader_status_momentum.json`) gevonden op GitHub of lokaal.")
