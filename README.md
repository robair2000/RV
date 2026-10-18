# ⚡ RV Institutional Quant & Trader Dashboard (V20)

Een autonoom, institutioneel-grade V20 swing-trading en machine-learning systeem op **KuCoin Futures**, gebouwd met een multi-service architectuur op Render, een centrale GitHub Single Source of Truth, en een live Streamlit-dashboard.

---

## 🏗️ Systeemarchitectuur & Componenten

Het systeem draait als een gedecentraliseerde multi-service op Render, waarbij **GitHub** fungeert als de centrale databron (`Single Source of Truth`) voor alle synchronisatie:

1. **`getcoins.py` (De Marktscanner):**
   * Scant alle actieve lineaire USDT Futures op KuCoin.
   * Berekent ATR, trendscores en volume om de sterkste momentummunten te identificeren.
   * Genereert en pusht de `kucoin_best_coins_ranking.csv` naar GitHub.

2. **`quant_worker.py` (De Machine Learning Trainer):**
   * Draait automatisch de scanner en haalt de top 5 dynamische munten op.
   * Combineert deze naadloos met de vaste kern (**BTC, ETH, LINK**) voor een actieve trading scope van 8 assets.
   * Traint ensemble-modellen (XGBoost, CatBoost, Random Forest) en pusht de `model_state_*.json` bestanden naar GitHub.

3. **`agent.py` (De S&R & S&D Confluence Agent):**
   * Trekt periodiek de actieve ranking van GitHub.
   * Analyseert 4h en 1d candles op Support/Resistance, Supply/Demand zones en past XGBoost meta-labeling toe.
   * Pusht de `pattern_state_*.json` bestanden naar GitHub.

4. **`trader_worker.py` (De Uitvoerder):**
   * Bewaakt openstaande posities, berekent dynamische trailing stops en beheert breakeven-beveiliging.
   * Plaatst intelligente zone-based limit orders op KuCoin Futures op basis van de goedgekeurde V20 signalen.
   * Bevat een ingebouwde webserver (`HealthCheckHandler`) met live HTML-weergave van de ranking-tabel op `/ranking`.

5. **`app.py` (Het Streamlit Dashboard):**
   * Een interactieve gebruikersinterface om in real-time koersen, ML-voorspellingen, V20-confluencescores en **live openstaande KuCoin Futures posities** te monitoren.

---

## 🎯 Dynamische Asset Selectie
Het systeem gebruikt een hybride portfolio-strategie:
* **Vaste Kern:** `BTC`, `ETH`, `LINK` (altijd actief voor stabiele liquiditeit).
* **Dynamische Top 5:** Wordt automatisch aangevuld vanuit de live marktscan (`kucoin_best_coins_ranking.csv`).

---

## 🚀 Deployment & Configuratie (Render)

### 1. Environment Variables (Instellen in Render Dashboard)
Zorg ervoor dat de volgende omgevingsvariabelen zijn ingesteld voor je services:
* `KUCOIN_API_KEY`: Jouw KuCoin API Key
* `KUCOIN_API_SECRET`: Jouw KuCoin API Secret
* `KUCOIN_API_PASSWORD`: Jouw KuCoin API Passphrase / Password
* `GITHUB_TOKEN`: Een GitHub Personal Access Token (met `repo` permissies)
* `GITHUB_REPO`: `robair2000/RV` (of jouw repository)
* `GITHUB_BRANCH`: `main`
* `DRY_RUN`: `false` (of `true` als je wilt testen zonder echt geld)
* `TRADE_AMOUNT_USDT`: `20.0` (Inleg per trade in USDT)

### 2. Lokale Dashboard Start (Streamlit)
Om het dashboard lokaal te draaien:
```bash
pip install -r requirements.txt
streamlit run app.py

Streamlit App (app.py): Biedt volledig inzicht in live koersen, modelvoorspellingen, ensemble-gewichten en je actuele beursposities via CCXT.

Trader Status Page: Bezoek de web-URL van je trader service op /ranking om de live HTML-tabel van de actieve marktscan direct in je browser te bekijken.
