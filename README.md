# Crude Oil Supply Chain Risk Intelligence

A deep learning system that predicts supply chain fragility for crude oil markets by reading four signal layers simultaneously. Built to answer a simple question: can you tell how close a supply chain is to breaking before it actually breaks?

The model doesn't predict black swans. It identifies when conditions are fragile enough that any shock, predictable or not, will cause outsized damage. Think of it less as earthquake prediction and more as identifying which buildings haven't been retrofitted.

<img width="3256" height="1062" alt="hero_timeline" src="https://github.com/user-attachments/assets/ad229364-50e0-4a25-9df0-d9d87efa47b9" />


The chart above shows the model's risk score from 2005 to 2025. Shaded regions are known disruptions. The model was trained on data through 2020 and had never seen the Russia-Ukraine war or Houthi Red Sea attacks. It flagged both.

---

## How It Works

The system reads 47 engineered features across four signal layers, every trading day:

**Physical supply chain** // US crude inventory levels vs seasonal norms, refinery utilization drops, import trend shifts. These tell you whether the physical infrastructure has buffer or is running tight.

**Financial signals** // Crude price momentum, realized and implied volatility, futures curve shape (contango vs backwardation), trading volume anomalies. Markets price in information fast but not instantly.

**Geopolitical tension** // Production-weighted tension scores across seven oil-producing regions (Saudi Arabia, Russia, Iraq, Libya, Iran, Nigeria, Venezuela). Built from GDELT event data. Tension escalates before markets react, usually by days or weeks.

**Environmental** // Hurricane intensity in the Gulf of Mexico, winter severity affecting pipelines and refineries.

The raw 17 signals get transformed into 47 features through rolling statistics, cross-signal interactions, and regime indicators. Things like "is inventory falling while tension is rising" or "how many oil-producing regions are stressed at the same time."

---

## Model

XGBoost regressor trained on 5,000+ trading days. The target is a continuous fragility score (0 to 1) constructed from 8 labeled disruption events with severity weighting and a 21-day lookahead ramp. The model learns what conditions look like in the weeks before a disruption, not during one.

**Performance:**

| Metric | Validation | Test |
|---|---|---|
| AUC-ROC | 0.90 | 0.87 |
| Avg Precision | 0.73 | 0.72 |

Train split: through Dec 2020 (includes GFC, Arab Spring, OPEC war, Harvey, Aramco attack, COVID). Validation: 2021-2022 (includes Russia-Ukraine). Test: 2023+ (includes Houthi Red Sea attacks). No data leakage between splits.

### Why XGBoost and not a neural net?

I started with XGBoost as a baseline and it performed well enough that adding an LSTM didn't justify the complexity for this dataset size. With only 6 training disruptions, a simpler model generalizes more reliably. The interpretability is also better: SHAP values give exact feature contributions for every prediction, which matters when the output needs to inform business decisions rather than just produce a number.

---

## What the Model Sees

### Feature Importance

!<img width="2847" height="1063" alt="feature_importance" src="https://github.com/user-attachments/assets/945f163b-d65e-4c62-a72c-d741afb18674" />


The signal layer split is balanced: Financial 29%, Market Structure 26%, Geopolitical 22%, Physical Supply 13%, Environmental 9%. The model is genuinely using all four layers, not just watching crude prices. The top feature (contango/backwardation flag) makes sense because futures curve shape is one of the most reliable single indicators of supply chain tightness.

### Signal Decomposition (SHAP)

<img width="3256" height="1062" alt="signal_decomposition" src="https://github.com/user-attachments/assets/8ce441ca-115e-4c50-84c8-f7195e623dcc" />


This breaks down what's driving the risk score at each point in time. Different disruptions have different drivers. The geopolitical layer spikes around Russia-Ukraine (2022). Environmental signals dominate during hurricane seasons. Financial and physical supply signals compound during broad market crises like COVID. The model adapts rather than applying one pattern to everything.

### Score Distribution

<img width="3060" height="907" alt="score_distribution" src="https://github.com/user-attachments/assets/9769476b-1b28-436b-b4ea-9ea7c3735230" />


Clean separation between calm and disruption periods on the training set. On val and test sets, the distributions overlap more in the 0.2-0.4 range. That's expected with only 6 training events. The model is conservative on unseen disruptions: it elevates the score in the right direction but doesn't push as high as it does for events it trained on. With real market data and more labeled events, calibration would improve.

---

## Business Impact: Airlines

Jet fuel is refined from crude oil, so crude price spikes flow through to airline operating costs with a short lag (typically 1.2x amplification from the crack spread). For a mid-size carrier spending $3B/year on fuel with an 8% operating margin, even moderate supply disruptions can erase profitability.

I computed what actually happens to crude prices during disruption windows at each risk score band, using the model's own predictions on the synthetic data. This isn't a theoretical exercise. The numbers below reflect what the model saw and how prices moved during the 8 labeled disruption events.

![Airline Scenario](outputs/airline_scenario.png)

The key finding: signal-triggered hedging (increasing fuel hedge coverage to 80% when risk score exceeds 0.3, dropping to 30% during calm periods) consistently outperforms the flat quarterly hedge that most carriers use. The savings come from two places. First, you lock in more volume at pre-spike prices before disruptions hit. Second, you hedge less during calm periods, avoiding unnecessary premium costs.

The margin erosion ranges are wide, and that's honest. With 6 training events and 2 held-out events, this is directionally correct but not a precise forecast. An airline's fuel desk would want to calibrate this against their specific route mix, existing hedge book, and contract structure. The model provides the timing signal. The client decides how aggressively to act on it.

A conservative carrier (thin margins, limited hedging infrastructure) might set their action trigger at risk score 0.3. A carrier with a sophisticated fuel desk might wait until 0.5. Both are reasonable. The recommendation adapts to the client's risk appetite.

---

## Business Impact: Shipping and Logistics

Shipping companies face a different version of the same problem. When supply disruptions close or threaten chokepoints (Suez Canal, Strait of Hormuz, Strait of Malacca), vessels must reroute around the Cape of Good Hope. That detour adds roughly 10 days per voyage at $32,500/day in operating costs, plus 15% additional fuel burn.

The question isn't whether to reroute. It's when. Preemptive rerouting (planning the detour when the risk signal is elevated but before the disruption is confirmed) costs significantly less than reactive rerouting (scrambling after the disruption hits, with congestion surcharges, last-minute fuel stops, and crew overtime adding a 1.35x premium).

![Shipping Scenario](outputs/shipping_scenario.png)

I computed the actual freight rate proxy behavior during disruption windows at each risk band, then modeled the fleet-wide cost difference between preemptive and reactive rerouting for a 45-vessel fleet.

The per-voyage savings from acting on the early warning signal (preemptive vs. reactive) are consistent across risk bands. The fleet-wide annual value scales with how much time the model spends at elevated risk levels and how many voyages are affected.

The affected voyage fractions (how many routes actually need rerouting at each risk level) are estimates. A regional disruption at risk 0.3-0.5 might only affect 15-25% of trade lanes. A systemic event at risk >0.7 could affect 60-80%. The actual number depends on the fleet's specific trade lane mix, which is client-specific information. The model tells you when to start planning. Fleet operations decides which routes to divert.

Revenue opportunity cost matters too. Ships on longer routes are tied up for 10 extra days per voyage, meaning fewer available voyages per year. For high-value cargo routes, that lost capacity can exceed the direct rerouting cost.

---

## Limitations

Honest about what this can and can't do:

The model trains on 8 disruption events. Six in training, two held-out. That's enough to learn directional patterns but not enough for precise calibration. The risk score is conservative on out-of-sample events: Russia-Ukraine (severity 4/5) only pushed the score to ~0.3-0.4. The model gets the direction right and correctly identifies the driving signal layer, but underestimates magnitude for events with novel characteristics.

The data is synthetic. Statistical properties are calibrated to real crude markets (price volatility, inventory ranges, seasonal patterns), and disruption events are injected at actual historical dates with realistic signal staggering (geopolitical tension leads price by ~25 days, freight leads by ~14 days). But real GDELT tension data, real EIA inventory pulls, and real futures data would improve signal fidelity. The code includes real API functions that can be swapped in by setting `USE_SYNTHETIC = False`.

The consulting scenarios use reasonable industry assumptions (jet fuel crack spread multiplier, rerouting costs, operating cost per vessel day) but these vary significantly by company. The model provides the timing signal. The dollar impact numbers are illustrative, not prescriptive.

---

## Data Sources

| Layer | Source | Frequency | Coverage |
|---|---|---|---|
| Crude inventory & refinery utilization | EIA API v2 | Weekly | 2005-2025 |
| Crude futures & implied volatility | Yahoo Finance (CL=F, OVX) | Daily | 2005-2025 |
| Geopolitical tension | GDELT event database | Daily | 7 oil-producing regions |
| Freight rates | Frontline (FRO) as VLCC proxy | Daily | 2005-2025 |
| Hurricane / weather | NOAA (seasonal indicators) | Daily | 2005-2025 |

Currently running on synthetic data.

---

## Disruptions in the Dataset

| Event | Period | Severity | Split |
|---|---|---|---|
| Global Financial Crisis | Jul 2008 - Dec 2008 | 5/5 | Train |
| Arab Spring / Libya | Feb 2011 - Oct 2011 | 4/5 | Train |
| OPEC price war | Jun 2014 - Feb 2016 | 4/5 | Train |
| Hurricane Harvey | Aug 2017 - Sep 2017 | 3/5 | Train |
| Saudi Aramco drone attack | Sep 2019 - Oct 2019 | 3/5 | Train |
| COVID + Russia-Saudi price war | Mar 2020 - Jun 2020 | 5/5 | Train |
| Russia-Ukraine war | Feb 2022 - Jul 2022 | 4/5 | Validation |
| Houthi Red Sea attacks | Oct 2023 - Mar 2024 | 3/5 | Test |

---

