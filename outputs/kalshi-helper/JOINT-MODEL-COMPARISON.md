# Weather and market model comparison

The best validation result was a three-seed ensemble of small jointly trained weather/market networks. Its improvement over market prices is uncertain. This experiment does not establish a trading edge or change the active app model.

## Results

Lower log loss and Brier scores are better. Each neural result averages all three predetermined seeds, not the best seed.

| Model | Log loss | Brier |
|---|---:|---:|
| Combined weather + market MLP | **0.491363** | **0.162410** |
| Market-only fitted correction | 0.494996 | 0.164586 |
| Market prices | 0.495074 | 0.164566 |
| Combined weather + market temporal CNN | 0.495870 | 0.164884 |
| Physical model with observation innovation | 0.658286 | 0.221179 |
| Weather-only MLP | 0.967320 | 0.232620 |
| Weather-only temporal CNN | 0.969016 | 0.239904 |

The combined MLP reduces log loss by about 0.75% and Brier by 1.31% relative to market prices. Its log-loss difference is -0.003711, with a descriptive multiple-comparison-adjusted block-bootstrap interval of **[-0.021112, +0.012380]**. Zero is inside the interval. Two of its three individual seeds perform worse than the market; averaging matters. This is not a stable individual-network win.

The temporal encoder did not improve results. Weather-only neural models substantially underperformed the simpler physical model. More complexity did not solve the small and seasonally narrow training sample. The market-only control is a penalized logistic offset, not a capacity-matched neural ablation, so this comparison cannot isolate how much of the ensemble's gain comes specifically from weather information.

## Experiment design

- Training: 514 bracket rows, 87 independent city-day events, 22 dates. One stale-observation event (six rows) was excluded from every arm before fitting.
- Validation: 730 bracket rows, 257 events, 67 dates; market probability restricted to 5–95%. No validation rows were excluded for missing observation innovation.
- Original chronological split membership was preserved. Reserved test data were not loaded. Validation dates have been inspected previously; all results are exploratory.
- Same cached, explicitly issued GFS/ECMWF runs and externally estimated observation discrepancy calibration across relevant arms. Forecast availability is checked using the existing eight-hour publication-delay assumption; this is not independently verified historical publication timing.
- Fixed 200 epochs, seeds 1729/1730/1731, AdamW, no validation checkpoint selection or hyperparameter search. Architecture and settings were recorded before training.
- Losses balance dates and events. Uncertainty uses 3,000 seven-date moving-block bootstrap replicates and Bonferroni percentile tails across six comparisons. This adjustment does not undo previous reuse of validation dates.

## What changed

`helper/joint_research.py` adds an optional PyTorch research workflow. Weather networks use a small eight-unit encoder to predict a bounded remaining-temperature mean correction and conditional spread. The observed-maximum distribution retains external station-specific bias/scale calibration. Weather-only features cannot read prices or settlement labels.

The temporal variant adds a four-channel convolution over 16 points from each already-issued forecast path. It does not consume future observations. The combined variant trains a weather interval likelihood together with contract probability loss and predicts a bounded correction to market log odds. The combined MLP has 179 parameters per seed; the temporal combined model has 271. These are probability models, not learned execution policies, and their contract outputs are not constrained to sum to one across an event.

All layers are registered before training. Numerical checks cover infinite bracket bounds, finite gradients, seed reproducibility, price isolation for weather inference, and forecast availability. The full Python suite passes **205 tests**. Synthetic fixtures test mechanics; reported comparison scores use historical data.

## Artifacts and reproduction

See `reports/joint-model-comparison-v2/` for the pre-training protocol, full report, per-seed training traces and scores, saved model artifacts, validation predictions, and verification metadata. The earlier `reports/joint-model-comparison/` contains only a protocol from preprocessing that stopped on the stale observation; it produced no trained-model results.

From the workspace root, using the existing Python environment with PyTorch installed:

```powershell
python work/compare_joint_models.py --output outputs/kalshi-helper/reports/joint-model-comparison-reproduction
```

Use a new output directory. The runner uses the local historical replay files and verified forecast cache under `work/browser-stage-data`; these are required for offline reproduction. PyTorch is optional for this research module and is not a new dependency of the normal app. Only load trusted model pickle files.

## Interpretation

Retain the combined MLP ensemble as a research candidate. Keep market probabilities as the benchmark. Do not promote this result as profitable: no executable-price, fee, slippage, or bankroll experiment was run here, and the interval does not establish superior forecasting. A frozen candidate evaluated on genuinely untouched, seasonally broader paired forecast/market data is needed before any stronger claim.
