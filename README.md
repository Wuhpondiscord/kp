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

The new **BTC paper lab** at `/btc` lets you pick one of three experimental models,
set a play-money balance and replay September with bundled real historical data.
See the [lab guide](outputs/kalshi-helper/BTC-LAB.md). All three standard scenarios
lose after costs. BTC is available for historical experimentation only; live BTC
automation and automatic model promotion remain disabled.

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
