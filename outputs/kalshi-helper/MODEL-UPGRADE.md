# Model and practice audit — September 21, 2026

The app trains real models, but there is still no proven trading advantage. This update improves model selection, adds an actual weather-forecast experiment, and fixes misleading training and practice feedback. All orders remain simulated.

## What was wrong and what changed

- The earlier neural run completed at epoch 65, but a 65/150 progress bar made it look unfinished. The bar now completes, explains early stopping, and reports preparation, fitting, evaluation and failure states. Startup recovery also recognizes interrupted preparation/evaluation. Completion replaces the old “training started” notice.
- Neural forecasts were always blended 50/50 with market odds. The final blend now uses validation-only selection, including a zero-weight fallback. Best-checkpoint files save that selected weight. Test dates never choose it.
- Automatic practice previously favored the earliest closing markets, including one-sided books near resolution. It now prioritizes usable quotes within the model's tested 5–25-hour window. The account view distinguishes actual ML estimates from market-price fallbacks and preserves expanded decision logs during refreshes.
- Added beta calibration to the six-model comparison. Contract explanations identify its actual input: market probability, not weather or city.
- Added a separate forecast-based weather training button. It is research-only and cannot place paper orders or replace the active trading model.

## Measured results on real data

Archive: 8,758 settled contracts and 26,007 hourly price samples across four daily high-temperature series. No synthetic examples enter this evaluation. The final comparison uses 5,200 samples, 1,752 contracts and 73 calendar days. Samples from the same date remain grouped, with a 48-hour settlement embargo. Auto-tune uses expanding validation folds; epoch training and weather fitting use chronological 60/20/20 date partitions with boundary purging.

| Forecast | Final-test log loss ↓ | Brier score ↓ |
|---|---:|---:|
| Market midpoint baseline | 0.20018944 | 0.06459957 |
| Beta-calibrated market blend | 0.19919105 | 0.06465324 |
| Independent GFS weather challenger | 0.45951606 | 0.14293363 |
| Neural model after validation-selected 0% neural weight | 0.20018944 | 0.06459957 |

Beta calibration improved log loss by about 0.50%, while Brier score was slightly worse. Its paired seven-day-block 95% interval for log-loss difference was [-0.00161105, -0.00041095]. These dates have already been inspected in earlier experiments; this is exploratory evidence, not a fresh confirmatory test. Automatic promotion remains blocked. Historical dollar scenarios cannot establish executable profit because historical depth and queue position are unavailable.

The browser-triggered neural run `train-f8fc1b43ff93` completed in 13.58 seconds, trained 65 epochs and retained epoch 61. Validation selected 0% neural contribution. The network itself did not beat the market; returning market odds is the intended fallback, not an improvement attributed to ML. Epoch curves use a fixed half-neural blend; the final report uses the selected blend. Auto-tune subsequently restored beta calibration as the experimental active model.

## Weather experiment and research

The original notes correctly prioritize calibrated distributions, actual forecast inputs, station rules and chronological validation before larger neural networks. [Gneiting et al.'s EMOS work](https://researchportalplus.anu.edu.au/en/publications/calibrated-probabilistic-forecasting-using-ensemble-model-output-/) motivates correcting forecast bias and uncertainty. [Rasp and Lerch](https://arxiv.org/abs/1805.09091) study neural ensemble postprocessing with meteorological inputs; a network on prices alone is a different task.

The new implementation fits a Gaussian temperature distribution with forecast, station and seasonal bias terms. It integrates this distribution over each contract's integer-temperature bounds. Training uses one winning interval per event (863 training events), not multiple copies of the same outcome or invented exact station temperatures. This is a deterministic-forecast MOS-style challenger, not full ensemble EMOS: there is no ensemble spread input yet.

Inputs are archived GFS fixed-lead hourly forecasts from the [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api). Raw responses and feature rows are saved. The model uses previous-day-2 forecasts, a conservative six-hour publication allowance, and rejects forecasts whose inferred availability is after the prediction time. These are inferred availability times, not original ingestion receipts. Hourly gridpoint maxima and fixed local-standard-day assumptions still need station/provider-specific prospective verification. The [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) offers a better basis for exact initialization alignment in a later iteration.

Validation selected **0% weather weight**. Its independent test loss was much worse than market prices. We therefore keep the challenger out of live recommendations. Next useful weather improvements are exact station observations, run-time alignment, multi-model ensemble spread and recent residual calibration, followed by untouched future evaluation—not a larger network on the same weak features.

[Kull et al.'s beta calibration](https://proceedings.mlr.press/v54/kull17a.html) motivated the added probability calibrator. It learns from log(p) and -log(1-p) and is compared against the unchanged market baseline.

For mentions, the original notes' count-model approach is more appropriate than pretending temperature training generalizes to speech. [Jansche's linguistic count-data research](https://aclanthology.org/P03-1037/) supports testing negative-binomial and zero-inflated alternatives. A valid implementation needs timestamped transcripts, event duration/type, speaker identity, exact word rules and the words actually listed before each event. Those aligned datasets are not present, so no trained mentions model is claimed. Post counts, seasonal weather and county-election models also remain future category-specific work.

## Software and live-practice verification

- 99 Python tests passed, including probability conservation across weather brackets, forecast availability and missing-hour rejection, one-event-one-training-label behavior, failed-training persistence, validation fallback, practice-market ranking and truthful calibrator explanations. Both JavaScript files passed syntax checks.
- Used the actual browser controls to train the neural network, auto-tune, fit the weather challenger and start/stop experimental practice.
- Live run `e3cab2666913`: $1,000 simulated bankroll; 24 markets checked; 11 eligible ML estimates per polling cycle; 33 prediction records, 33 pending signals, 22 skips after fresh books and 11 cancellations at shutdown. No source errors, fills or realized profit. Skips cited insufficient net edge, usable depth or risk budget. No thresholds were weakened to manufacture trades.
- The approximately three-minute session stopped successfully. This verifies data-to-prediction-to-decision plumbing, not a full-day soak, settlement lifecycle or profitability. Those remain separate validation requirements.

Reproducible evidence: [model-upgrade-check.json](reports/model-upgrade-check.json). Weather artifacts are saved under `data/weather-model/`; neural checkpoints, split manifests and epoch logs are under `data/training/`. Run `python -m unittest discover -s tests -v` from this project to rerun the software suite.
