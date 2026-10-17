# ⚡ RV Institutional Multi-Agent Crypto Trading Ecosystem

Een geavanceerd, hiërarchisch multi-agent kwantitatief handelssysteem voor **BTC**, **ETH** en **LINK** op KuCoin Futures, aangedreven door adaptieve Machine Learning modellen en een strategische "Final Boss" marktregime-agent.

> ⚠️ **Disclaimer:** Dit systeem is experimenteel en bedoeld voor kwantitatief onderzoek en geautomatiseerde uitvoering. Geen financieel advies. Test altijd eerst uitgebreid met `DRY_RUN=true`.

---

## 🏗️ Systeemarchitectuur

Het ecosysteem is opgebouwd uit een hiërarchie van autonome agenten die via GitHub state-payloads asynchroon met elkaar communiceren:

+-----------------------------+
                  |     Final Boss Agent        |
                  |   (agent.py op Render)      |
                  +--------------+--------------+
                                 |
                                 | Pusht pattern_state_*.json
                                 v
                  +-----------------------------+
                  |      GitHub Repository      | <--- Pusht model_state_*.json
                  +--------------+--------------+
                                 ^                               ^
                                 |                               |

+-----------------------------------+--+         +------------------+---------------+
|       Trader Worker                  |         |         Quant Worker             |
|   (trader_worker.py op achtergrond)  |         |   (quant_worker.py via Cron/Server)|
+--------------------------------------+         +----------------------------------+
     

### Bestandsleutel & Rollen

| Bestand | Rol | Beschrijving |
|---|---|---|
| **`agent.py`** | 👑 **Final Boss Agent** | Draait 24/7 als Render Web Service. Analyseert marktvolatiliteit, Bollinger Squeezes, RSI-momentum en stelt dynamisch het risicoregime, de minimale score-eisen, hefboomwerking (*leverage*) en trailing-stop multipliers vast. |
| **`trader_worker.py`** | ⚙️ **Trader Worker** | De fouttolerante uitvoerder op KuCoin Futures. Haalt de nieuwste states op, voert een strikte **Veto-check** uit op de instructies van de Final Boss, bewaakt de totale portfolio-exposure en beheert open posities met trailing stops. |
| **`quant_worker.py`** | 🧠 **Quant ML Worker** | Haalt marktdata en orderboek/funding-data op, traint een ensemble van XGBoost, CatBoost en RandomForest modellen, en voert out-of-sample holdout-validaties uit. |
| **`app.py`** | 📊 **Streamlit Dashboard** | Real-time monitoring hub voor het inzien van modelvoorspellingen, ensemblegewichten, marktcontext en actieve signalen. |

---

## 👑 Strategie & Final Boss Gatekeeper

De Trader Worker opent **alleen** een positie als voldaan is aan de volgende voorwaarden:
1. **Richting Overeenstemming:** Het signaal van het ML-model (`long`/`short`) moet exact overeenkomen met het patroonsignaal van de Final Boss.
2. **Dynamische Score Drempel:** De patroonscore van de Final Boss moet hoger zijn dan de op dat moment door de Boss geëiste minimumscore (`min_score_required`).
3. **Portfolio Exposure Limit:** Er mogen maximaal **2 gelijktijdige posities** openstaan over het hele portfolio (`MAX_CONCURRENT_POSITIONS = 2`).
4. **Harde & Zachte Risicofilters:** Voldoet aan eisen m.b.t. RRR (Risk-Reward Ratio ≥ 1.4), Holdout-nauwkeurigheid, Orderboek-imbalans en Funding-rates.

---

## ⚙️ Environment Variables

Stel de volgende omgevingsvariabelen in op je Render Web Service of lokale omgeving:

### Voor de Final Boss (`agent.py`) & Dashboard (`app.py`):
* `GITHUB_TOKEN`: GitHub Personal Access Token (met `contents: write` rechten).
* `GITHUB_REPO`: `robair2000/RV` (of jouw eigen repository).
* `GITHUB_BRANCH`: `main`.
* `EXCHANGE_ID`: `kucoin` (standaard).
* `PORT`: `10000` (automatisch door Render ingesteld).

### Voor de Trader Worker (`trader_worker.py`):
* `KUCOIN_API_KEY`: API Sleutel van KuCoin Futures.
* `KUCOIN_API_SECRET`: Secret Sleutel van KuCoin Futures.
* `KUCOIN_API_PASSWORD`: Passphrase van KuCoin API.
* `DRY_RUN`: `true` (voor simulatie) of `false` (voor daadwerkelijke handel).
* `TRADE_AMOUNT_USDT`: Marge per positie in USDT (bijv. `20.0`).
* `MARGIN_MODE`: `cross` of `isolated`.

---

## 🚀 Lokaal & Server Starten

1. Afhankelijkheden installeren
bash
pip install -r requirements.txt

2. Final Boss Agent starten
Bash
python agent.py

3. Quant Worker uitvoeren (Trainingsronde)
Bash
python quant_worker.py

4. Trader Worker starten (Live / Dry-run)
Bash
DRY_RUN=true python trader_worker.py

5. Streamlit Dashboard openen
Bash
streamlit run app.py


