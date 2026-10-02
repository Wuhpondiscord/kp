# Training improvements — September 23, 2026

Correction from the subsequent station audit: this earlier weather comparison used O’Hare coordinates for Chicago, while the contracts specify Midway. Those Chicago weather figures should not be used as current performance evidence. The mapping is corrected in source; see MARKET-EDGE-EXPERIMENT.md for the later station-informed evaluation.

This pass improves the training pipeline and the independent weather experiment. It does **not** establish a stronger deployable betting model. Models were fitted and evaluated on the real archive in `work/browser-stage-data`; no synthetic data contributed to the reported evaluation. The original application data directory was not replaced.

## What changed

**Automatic model selection.** Recommended training now fits the market-offset neural network and compares its validation checkpoint with six smaller, regularized calibrators: beta, context and spline bases at two fixed regularization strengths. Context models learn how calibration varies with city, spread and horizon. Every candidate starts from market log odds. A fixed set of blend weights shrinks toward market prices. Selection minimizes validation daily log loss and rejects candidates whose validation Brier score is worse than the baseline. The market baseline remains an explicit fallback. Scaling and coefficients fit training data only; final-test labels cannot choose a candidate. Legacy and neural-only methods remain selectable.

**Richer real weather data.** Added ECMWF IFS 0.25° alongside GFS Seamless through the public Open-Meteo archive. Each supplies forecasts at 24- and 48-hour lead offsets. The loader retains raw responses, caches requests by URL, validates Fahrenheit units and timestamps, and requires a complete 24-hour settlement-day window. For each prediction, it picks the newest complete eligible lead. Eligibility includes a six-hour availability allowance. This is still inferred availability rather than an original ingestion log.

**Conditional weather uncertainty.** A new interval-likelihood model learns location/season bias and a log-scale that varies with city and forecast lead. The combined-source candidate can also learn from GFS/ECMWF disagreement. This disagreement is not described as a true ensemble spread. Winning settlement brackets are interval labels; repeated quotes and lead snapshots share one event's total fitting weight. No exact observed temperature is invented.

**App integration.** Recommended training runs automatic selection and saves a reloadable artifact used by the existing model selector and prediction service. The weather button runs the expanded experiment and exposes source-comparison results under its method details. Weather forecasts remain research-only: they do not enter live recommendations unless a separate deployment path and validation justify that change.

## Results

The complete automatic training comparison selected the existing neural checkpoint, at 75% neural weight and 25% market weight. The smaller calibrators did not improve validation enough to replace it. Neural fitting stopped after 56 epochs, keeping epoch 36. The dataset contains 51,747 quotes from 8,806 contracts across 367 dates; 30,806 training, 9,671 validation and 10,438 test samples remain after boundary purging.

| Main model, identical final-test rows | Log loss ↓ | Brier ↓ |
|---|---:|---:|
| Automatic selection | 0.208091 | 0.067651 |
| Market baseline | 0.208974 | 0.067400 |

These are unchanged from the prior neural model. The seven-day block bootstrap interval for loss difference includes zero. Test dates have been inspected before; automatic promotion remains blocked. Better validation results alone are not evidence of executable profit.

The weather experiment joined 26,151 real quote samples with both providers, with zero excluded samples in this run. 17,410 rows qualified for the fresher day-one forecast and 8,741 required day two. Train/validation/test contained 15,550 / 4,902 / 5,274 rows, grouped by date with a 48-hour settlement embargo. Weather and main-model sample counts differ because the weather experiment retains the original 24/12/6-hour quote sampling.

| Predeclared weather-only comparison, same test rows | Log loss ↓ | Brier ↓ |
|---|---:|---:|
| Older GFS forecasts, conditional calibration | 0.468004 | 0.145460 |
| Freshest eligible GFS | 0.472595 | 0.145667 |
| Freshest eligible ECMWF | 0.450864 | 0.138708 |
| Combined GFS + ECMWF | 0.442813 | 0.135047 |

Combining providers reduced weather-only log loss by **5.4%** and Brier score by **7.2%** relative to the older-GFS comparison. Simply using newer GFS did not help on these rows. These are descriptive test comparisons, not a basis for selecting a source after looking at test results. Validation chose **zero weather weight** against market odds. Consequently, the improved weather-only model did not improve the betting forecast. Forecast gridpoints, hourly maxima and settlement labels remain imperfect substitutes for station observations, especially late in the day when the market has observed much of the outcome.

## Verification

119 Python tests and the frontend regression checks passed. New tests exercise training-only scaling, outcome-independent features, learned bias, serialization, exact market fallback, cancellation, complete-day forecast availability, fallback when a newer forecast is incomplete, and event-weight invariance to duplicate quotes. The trained adaptive artifact was reactivated and successfully used by the prediction service. Browser checks cover recommended training and the new default method. The final browser result is recorded separately in `reports/training-browser-check.json`.

Evidence: `reports/adaptive-training-comparison.json`, `reports/weather-source-comparison.json`. Weather download requests, lead counts, exclusions, splits and validation candidate results are retained in the weather report. Tests use generated fixtures to verify software behavior; those fixtures are not the performance dataset.

## Research and next model work

- [Rasp and Lerch: neural weather post-processing](https://arxiv.org/abs/1805.09091) supports learning distribution corrections from numerical forecasts and local predictors. It motivates conditional distribution modeling, not an assumption that a larger network wins on this archive.
- [Open-Meteo previous-runs documentation](https://open-meteo.com/en/docs/previous-runs-api) defines forecast lead offsets and available models. [Single-run archives](https://open-meteo.com/en/docs/single-runs-api) are the next step for reconstructing coherent forecasts available at a particular decision time. Fixed-lead retrospective data remains exploratory here.
- [Chronos-2](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) supports multivariate and covariate-informed forecasting. No Chronos, vision backbone, or other pretrained weights were downloaded or fine-tuned in this pass. A fair experiment first needs sequential station observations, known-at-time forecast inputs and a checkpoint/data-contamination review. Kalshi price corrections alone are a weak reason to add a large forecasting backbone.

The next useful experiment is exact-station observations and same-day running maxima, joined to forecast issuance times and contract settlement rules. Reserve genuinely new dates before comparing additional architectures. For mentions and count markets, build separate timestamped event datasets; weather training does not provide transferable category coverage automatically.
