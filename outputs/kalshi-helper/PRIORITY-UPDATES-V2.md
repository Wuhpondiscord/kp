# Prediction and strategy updates

## What is now implemented

1. **Weather inputs:** prospective sessions archive NWS hourly forecasts, recent station observations, grid cloud cover, and current GFS/ECMWF hourly forecasts through Open-Meteo. Features include remaining-day maxima, forecast revision at a common decision time, humidity, wind speed, cloud cover, model disagreement and missing-data indicators. Sources retain receipt times and raw hashes. Optional source failures are explicit warnings. Open-Meteo provides stitched current forecasts, not independently attested initialization timestamps. [Source documentation](https://open-meteo.com/en/docs).
2. **Physical distribution:** `RemainingMaximum` fits a coherent maximum-temperature CDF using interval-censored winning brackets. Training uses all eligible training prices to avoid selecting physical outcomes by price. Validation selects a market blend, including zero weather weight; Brier cannot worsen. A fixed 4F weather baseline is reported. This remains a research challenger, not an automatically promoted live model.
3. **Evaluation:** settlement collection joins outcomes as targets only. The dataset builder verifies journal chains, arrival times, market rules, book freshness, forecast coverage and age; it keeps one observation per contract/hour. The new trainer requires 90 settled dates and applies whole-date 48-hour settlement purging. The primary scoring cohort is 12–24 remaining hours and 5–95-cent market midpoints; late hours and extreme prices are separate diagnostics. Frozen features, protocol, selection and model artifacts are saved. Recording prospectively does not itself guarantee an untouched test: the report remains research-only.
4. **Execution:** live model-generated intents require a matching-model check on the execution book. The original side stays fixed, but the probability is recomputed before fee/depth/risk checks. Missing or unsupported refreshes prevent fills. Imported external predictions retain their original delayed-signal behavior. Book records carry the execution check so replay does not need to call a model later. Skips distinguish missing model support, lost edge, usable depth and risk budgets.
5. **Strategy selection:** eight predefined edge/sizing combinations are evaluated on validation only, with an adjusted uncertainty bound and at least 30 traded city/day groups. Selection is saved before test evaluation. No eligible candidate means no trade. The price-model scenario recomputes price/time features at execution, incorporates daily/session loss stops and stresses slippage after selection. Holding to settlement is fixed; early-exit optimization is deliberately not implemented without an intratrade book history. Event and portfolio caps remain in force.

## Run results

- 156 Python tests passed, including refreshed execution probabilities, blocking missing/future execution checks, physical-model fitting, probability conservation and insufficient-data handling. Frontend checks and JavaScript syntax checks passed.
- Live NWS/GFS/ECMWF source check returned data without warnings, including humidity, cloud, wind and disagreement features.
- A bounded experimental paper smoke session (`26d63cb5eadd`) recorded three books, two feature rows and two model checks without errors. It made zero fills; this does not test profitability or establish fill realism.
- Settlement collection checked that contract successfully; it had not settled. Both feature rows were excluded from training as unsettled.
- The remaining-day trainer reported insufficient data. It did not fit a real-data model or fabricate labels. Its fit and conservation tests use synthetic fixtures solely to verify implementation.
- Strategy search against `train-103cd0ff5f8a` rejected all eight validation candidates. It did not choose a test winner. Results are in `reports/strategy-v2/`.

## Using the features

Restart the app to load the Python changes, then refresh the page. Practice with **Save evidence for model evaluation** enabled. On the Research page, **Improve my weather strategy** provides:

- **Collect settled results** — obtains target outcomes after markets settle.
- **Train weather challenger** — builds the prospective dataset and trains only when requirements are met.
- **Evaluate betting strategy** — evaluates the selected epoch-trained price model using validation-only selection.

Experiments do not automatically replace the selected prediction model. Their last status is shown in the panel, and full results are saved under the selected data directory's `experiments/` folder. Finish/stop practice before freezing evidence for these jobs.

Equivalent CLI commands from the project directory:

```powershell
python -m helper --data ../../work/browser-stage-data collect-weather-outcomes
python -m helper --data ../../work/browser-stage-data train-remaining-weather --output reports/my-weather-run
python -m helper --data ../../work/browser-stage-data validate-strategy --training-folder ../../work/browser-stage-data/training/train-103cd0ff5f8a --output reports/my-strategy-run
```

Output folders must be new, preserving earlier experiments.

## Evidence still needed

Fresh settled weather data, repeated seasonal coverage and a pre-registered untouched final period require elapsed collection time. Current observations are hourly sampled maxima, not complete five-minute settlement observations. The physical model assumes a fixed 1F observation error and does not yet fit a correlated hourly temperature path or ensemble members. Historical execution scenarios still lack verified queue/depth data; their uncertainty intervals are exploratory. A profitable live strategy, early-exit policy or real-money readiness has not been established.

NWS field/source reference: [NWS API documentation](https://www.weather.gov/documentation/services-web-api).
