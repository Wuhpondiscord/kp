# Independent station-weather pilot — October 2, 2026

The first dataset and baseline independent of Kalshi contracts are implemented and exercised. No market prices, contract labels, old trading holdouts or Polymarket values are used to fit or evaluate this pilot.

## What was built

- `helper/weather_station_pilot.py`: public-source download/cache, hash verification, station identity and units checks, full standard-time daily windows, chronological fitting and forecast-distribution evaluation.
- Targets: daily high temperatures parsed by IEM from NWS CLI products, preserving the exact product ID and issue timestamp. These are weather labels, not a claim of universal equivalence to Kalshi settlement values.
- Features: GFS global archived fixed-lead `temperature_2m_previous_day2`, reduced to the maximum of 24 hourly forecasts in the station's fixed local-standard-time day. Grid coordinates and Fahrenheit/UTC units are checked.
- Two fixed baselines: raw GFS maximum with train-estimated station error width; station-bias-corrected GFS with train-estimated residual width. Both are Gaussian distributional baselines with a declared 0.5°F minimum sigma. No neural network, tuning sweep or pretrained model is involved.
- Immutable protocol, raw/source hashes, train/validation JSONL, station parameters, scores by station and quarter, PIT bins and paired CRPS interval.

## Temporal scope and limitations

2023 is training; 2024 is validation. Training ends December 28 and training labels must have been issued before the first validation prediction cutoff. Training contains 1,446 station-days over 362 dates, versus 81 events over 22 dates in the most recent contract-model comparison. Validation contains 1,383 station-days over 346 dates. The four cities are NYC, Chicago Midway, Miami and Denver. City observations are clustered together in the paired date-block uncertainty calculation.

This archive is a fixed-lead product assembled from different model runs. It is **not an explicit single-run forecast** and is not suitable to claim exact historical tradability. Availability is conservatively inferred as the latest valid hour minus 48 hours plus eight hours, with the decision 12 hours before the station day begins. That allowance is not a verified receipt timestamp. GFS model changes, local downscaling, hourly sampling versus subhourly maxima, CLI revisions and source-specific settlement definitions remain limitations.

The source returned missing forecast values on some dates. The pilot excluded 92 station-days for incomplete forecast windows across the two source years (including dates beyond the declared training cutoff), and two label exclusions were recorded. It did not impute or replace them with newer forecasts or observed weather. CLI response completeness is separately recorded in `reports/weather-independent-verification.json`. A separate short request around late December 2023 also returned missing GFS values; the annual-request boundary was not simply assumed to explain those gaps. Metrics are conditional on available cases.

## Results on 2024 validation

| Metric | Raw GFS baseline | Station correction |
|---|---:|---:|
| Mean absolute error | 2.730°F | 2.591°F |
| RMSE | 3.622°F | 3.504°F |
| Mean forecast-minus-target bias | −0.753°F | +0.045°F |
| CRPS, lower is better | 1.953°F | 1.880°F |
| Gaussian negative log likelihood | 2.653 | 2.631 |
| Coverage of nominal 90% intervals | 92.48% | 91.68% |

The average CRPS improvement is approximately 3.7%. The synchronized seven-date-block interval for corrected-minus-raw CRPS is [−0.116, −0.031]°F, conditional on this fixed comparison, the available dates and the retrieved archive vintage. This is evidence of improvement for this weather-only pilot, not proof of multi-year generalization or a market edge. Gaussian NLL treats integer-reported highs as continuous measurements; CRPS is the declared primary metric.

Station CRPS changes: Chicago improves from 2.174 to 2.092°F and Denver from 2.298 to 2.045°F. Miami worsens from 1.192 to 1.235°F; NYC is essentially unchanged at 2.147°F. We report these differences instead of selecting station-specific winners after seeing validation.

## Reproduce

From `outputs/kalshi-helper`, with the existing Python environment:

```powershell
python -m helper.weather_station_pilot --cache ../../work/weather-independent-cache --output reports/my-weather-pilot --offline
```

Use a new output directory. Omit `--offline` to download missing responses. Cached responses are verified against SHA-256 metadata. The model parameters are portable JSON rather than executable pickle. The complete offline rerun reproduced every output file byte-for-byte.

## What this changes for the project

We now have a tested route to train weather calibration without being limited to market-listed event-days or narrow contract brackets. The first simple correction is helpful overall and exposes station differences that the small joint model could not establish reliably.

This is a day-ahead weather-data pilot; it does not replace the intraday live model. The next gate is verifying exact initialized forecast runs and settlement-label equivalence on a small station/date sample, then defining a separate, fixed rolling-year evaluation. Extending the archive should precede a more complex network. No production model was replaced and no profitability test or real-money order was run.

Sources: [IEM CLI API](https://mesonet.agron.iastate.edu/json/cli.py?help), [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api). Respect source access/usage terms; the free Open-Meteo API is intended for noncommercial use.
