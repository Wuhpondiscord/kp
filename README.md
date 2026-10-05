---
title: BetCheck
emoji: 🌦️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BetCheck — real Kalshi markets, simulated money

Browse markets, inspect experimental weather probabilities, choose WeatherSignal,
MarketGuard or a custom ConsensusBlend, and evaluate or continue training saved models.
**No proven trading edge. No real-money orders.**

The local [BTC data and cost upgrade](outputs/kalshi-helper/BTC-DATA-AND-COST-UPGRADE.md)
adds archived July markets, aggressor-volume and cross-exchange features, a
market-relative challenger and a sealed evaluation cohort. Its positive validation
result involves only three trades and has not been deployed.

The [expanded BTC flow evaluation](outputs/kalshi-helper/BTC-FLOW-FOLLOWUP.md)
tests fixed feature ablations across four chronological windows. The full model
loses $0.46 on two simulated trades and does not beat market accuracy overall;
the earlier small positive result is not confirmed. The holdout remains sealed.

The [fixed market anchor study](outputs/kalshi-helper/BTC-FIXED-ANCHOR-STUDY.md)
removes learned market recalibration and slightly improves development forecast
scores, but generates no trades after costs. Delayed inputs remove that small
advantage. It remains a local research candidate.

The [execution feasibility study](outputs/kalshi-helper/BTC-EXECUTION-FRONTIER.md)
replays frozen models under explicit costs and separates cost increases from
trade-selection changes. Narrow slippage tolerances and unstable selected-trade
results do not support model promotion.

The [training-day stability study](outputs/kalshi-helper/BTC-STABILITY-STUDY.md)
tests a fixed-anchor ensemble and a disagreement guard. The ensemble is worse
overall than the single model; the guard abstains. Neither is promoted.

The [ensemble diagnosis and repair](outputs/kalshi-helper/BTC-STABILITY-REPAIR.md)
corrects unequal boundary-day weighting and moving feature origins. It recovers
part of the ensemble's forecast regression, but the single model remains better
and trading evidence remains insufficient.

[Two further BTC batches](outputs/kalshi-helper/BTC-CONTEXT-BATCHES.md) test
training-only regularization, price/volatility context and directional constraints.
One candidate slightly improves Brier but worsens log loss and primary trading
results. All six candidates remain unpromoted research.

The [research review and tree benchmark](outputs/kalshi-helper/BTC-RESEARCH-AND-TREES.md)
connects tabular/market-model research to our data, tests market-initialized trees,
and adds a hashed experiment ledger. Trees underperform; no SOTA or edge claim
is supported.

The [pinned TabPFN benchmark and repair](outputs/kalshi-helper/BTC-TABPFN-STUDY.md)
now tests pretrained local inference rather than only discussing it. Results
reproduce exactly but underperform the baseline; a bounded market correction
reduces the regression without establishing an improvement or trading edge.

The [one-second data experiment](outputs/kalshi-helper/BTC-ONE-SECOND-DATA.md)
returns to custom models with richer timestamped inputs and a passing
second-to-minute aggregation audit. Results vary by period and do not beat the
existing custom baseline overall.

The **BTC paper lab** at `/btc` now includes three additional **EXPERIMENTAL**
replays: FixedAnchor (lowest log loss), DirectionalAnchor Joint (lowest Brier),
and Delayed MicroFlow (positive but sparse timing-stress P&L). Read the detailed
[pipelines, benchmarks and limitations](outputs/kalshi-helper/BTC-EXPERIMENTAL-MODELS.md).
These use a common 527-contract June/July cohort and shared trading rules. Legacy
September models remain separately labeled. No proven edge, live BTC trading or
automatic promotion; all results are historical simulations.

A separate [BTC 15-minute research pilot](outputs/kalshi-helper/BTC-PILOT.md)
now compares Coinbase-based forecasts with Kalshi prices on real September 2026
contracts. It does not establish an edge and is not a live model selection.
The [calibration follow-up](outputs/kalshi-helper/BTC-NEXT-STEP.md) tests settlement
averaging and temporal stability, and documents the next data-source investigation.
The optional [local BRTI capture tool](outputs/kalshi-helper/BRTI-SETUP.md) is ready
for an entitled Kalshi account; live access and tick import remain unverified.
The [forecast and trading comparison](outputs/kalshi-helper/BTC-FORECAST-AND-TRADING.md)
evaluates nonlinear BTC forecasts and a separately trained trade filter. These
remain development experiments, with no demonstrated profitable strategy.
The [source and edge audit](outputs/kalshi-helper/BTC-DIAGNOSIS.md) measures price-source
disagreements and the gap between predicted and realized trading returns.
The [prospective BTC runner](outputs/kalshi-helper/BTC-PROSPECTIVE.md) now freezes
models and records public data for a 30-day paper comparison with depth-limited replay.
For research without waiting, the [historical BTC simulation](outputs/kalshi-helper/BTC-HISTORICAL-SIMULATION.md)
trains on six months of public Binance data and replays an existing month of Kalshi
contracts. This is the current historical-first workflow; no forward run is required.
The [BTC loss audit](outputs/kalshi-helper/BTC-LOSS-AUDIT.md) explains selected-trade
overconfidence, reconciles trading costs, and corrects the forecast/execution timing
comparison before further model development.

The hosted Space is a **shared demo**: visitors share model selections, job slots and
paper sessions. Do not upload private information. Runtime data and new checkpoints
may disappear on restart. Run locally for private, durable research.

## Run locally

Use Python 3.13. From `outputs/kalshi-helper`:

```sh
pip install -r requirements.txt
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m helper serve
```

Copy `seed/single-run-archive` to `data/single-run-archive` for offline model training
and evaluation. The hosted entrypoint does this automatically. Open http://127.0.0.1:8765.
No Kalshi account or API token is required.

## Deployment and verification

Pushes to main run Python and frontend tests, load/evaluate all three models and
complete a one-epoch training/checkpoint round trip. GitHub Actions then uploads to
[wuhp/kp](https://huggingface.co/spaces/wuhp/kp), using the repository Actions secret
`HF_TOKEN` with write access to that Space. The token is never included in the image.

After upload the workflow waits for the exact GitHub revision at `/healthz` and
checks pages, custom weights, evaluation, one-epoch training, the synthetic paper
engine, Kalshi connectivity and live weather-market discovery. An upload alone is
not a successful deployment. Check Actions for failures; provider outages may fail
live checks even when the container is healthy.

The Docker setup follows [Hugging Face documentation](https://huggingface.co/docs/hub/spaces-sdks-docker).
Port: 7860. Allowed public origin: `https://wuhp-kp.hf.space`. Change both the
`BETCHECK_PUBLIC_ORIGIN` environment variable and smoke-test URL if changing domains.

## Models, sources and evidence

- [Model differences and results](outputs/kalshi-helper/MODEL-DISPARITY-LOG.md)
- [Model Room](outputs/kalshi-helper/NAMED-MODEL-UI.md)
- [Full audit](outputs/kalshi-helper/FULL-CODE-AND-RESEARCH-AUDIT.md)
- [Independent weather pilot](outputs/kalshi-helper/INDEPENDENT-WEATHER-PILOT.md)
- [Categorical challenger](outputs/kalshi-helper/EVENT-FUSION-BENEFITS.md)
- [Forecast provenance](outputs/kalshi-helper/FORECAST-ARCHIVE-SETUP.md)

Saved models are hash-verified. Bundled development rows and archived GFS/ECMWF
forecasts support offline evaluation; reused validation is not an untouched test.
The reserved test partition and local databases are excluded. Historical research
documents may refer to local work directories or packages outside this hosted snapshot.

Live sources: Kalshi public API, Open-Meteo GFS/ECMWF and Iowa Environmental Mesonet
station observations. Polymarket is an optional research input; all simulated trades
are Kalshi contracts. Current named models cover NYC, Chicago, Miami and Denver daily
highs near their tested horizon. Unsupported markets display a reason instead of
fabricated probabilities. The independent weather pilot is research, not a replacement
for the saved market models.
