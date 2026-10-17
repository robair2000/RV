# ⚡ RV Multi-Timeframe Ensemble Predictor & Trader

Zelflerend ensemble-systeem voor BTC, ETH en LINK met een Streamlit-dashboard en een automatische futures-trader (KuCoin).

> ⚠️ Experimenteel. Geen financieel advies. Draai eerst met `DRY_RUN=true` en controleer de out-of-sample cijfers in het dashboard voordat je echt geld inzet.

## Onderdelen
| Bestand | Rol |
|---|---|
| `quant_worker.py` | Haalt candles op (KuCoin via ccxt), berekent indicatoren, traint XGBoost + CatBoost + RandomForest, schrijft `model_state_<asset>_<tf>.json` en synchroniseert via de GitHub-API. |
| `server.py` | Kleine webserver; `GET /run-worker` (cron) start één trainingsronde (met lock tegen dubbele runs). |
| `trader_worker.py` | Leest de state-bestanden, filtert signalen, opent/beheert futures-posities (hard stop, trailing stop, take-profit). |
| `app.py` | Streamlit-dashboard. |

Timeframes: `15m`, `1h`, `1d`, `1w`, `1m`. De trader gebruikt `15m` en `1h`.

## Hoe de voorspelling werkt
- **Target = rendement van de volgende candle** (niet de absolute prijs). `last_prediction = laatste_close × (1 + voorspeld_rendement)`.
- De nog lopende candle wordt weggelaten; alleen afgesloten candles worden gebruikt.
- **Out-of-sample validatie:** per run wordt op de eerste 75% getraind en op de laatste 25% getest. Daaruit komen:
  - modelgewichten (omgekeerd evenredig met de holdout-fout),
  - `holdout_direction_accuracy` (hoe vaak de richting klopte),
  - `baseline_error` = *skill*: ensemble-MAE gedeeld door MAE van "geen beweging". **Onder 1 = beter dan niets.**
- **Zelfoptimalisatie:** elke run wordt één kleine mutatie (learning rate, lookback, features) voorgesteld en alleen overgenomen als hij op dezelfde holdout beter scoort.
- **Signaal:** `long`/`short` alleen als regressie en classifier dezelfde kant op wijzen, anders `none`. `signal_probability` is de kans op de gekozen richting.

## Trader-regels
- Richting en kans komen uit `signal` / `signal_probability`.
- Harde filters: ATR-band (echte ATR uit de state), RRR, holdout-nauwkeurigheid ≥ `MIN_HOLDOUT_ACCURACY`, kans ≥ `MIN_DIRECTION_PROBABILITY`, voorspelde move ≥ `MIN_PREDICTED_RETURN_PCT`, state niet ouder dan 40 min (15m) / 2,5 uur (1h).
- Zacht (min. 2 van 4): 1h bevestigt, niet tegen BTC in, orderboek, funding. Onbekend = geen punt.
- Positiegrootte: `floor(margin × hefboom / contractwaarde)`; één contract alleen als dat ≤ 1,5× de gewenste notional is.
- Stops worden **elke minuut** bewaakt en sluiten altijd (ook als het model nog "met je mee" is). Verlies → cooldown van 1 uur.

## Environment variables
`KUCOIN_API_KEY`, `KUCOIN_API_SECRET`, `KUCOIN_API_PASSWORD`, `GITHUB_TOKEN` (contents: write), optioneel: `GITHUB_REPO`, `DRY_RUN`, `TRADE_AMOUNT_USDT`, `LEVERAGE`, `MARGIN_MODE`, `EXCHANGE_ID` (standaard `kucoin`), `MIN_DIRECTION_PROBABILITY`, `MIN_PREDICTED_RETURN_PCT`, `MIN_HOLDOUT_ACCURACY`.

## Draaien
```bash
pip install -r requirements.txt
python quant_worker.py            # één trainingsronde
DRY_RUN=true python trader_worker.py
streamlit run app.py
```
Procfile: `web: python server.py` en `trader: python trader_worker.py`.
