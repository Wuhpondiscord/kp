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
