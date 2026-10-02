# Current professor review — September 24, 2026

This document supersedes older narrative summaries in the package. Those documents remain as experiment history. The project is a local Kalshi paper-trading app, implemented in NumPy, SciPy and scikit-learn. It has no real-money execution adapter and no deep-learning framework. The new Polymarket integration is read-only.

## Assessment

There is still **no demonstrated profitable edge**. The clearest recent improvement is in usable cross-market information, not measured prediction accuracy. A corrected Polymarket parser recovered a constrained regional temperature distribution from incomplete books that the previous implementation discarded. Accuracy and P/L improvement from that information cannot yet be estimated: there are no adequately sized, settled, paired prospective examples.

The package now includes the complete frozen 51,747-row price-model feature dataset, the exact training configuration and a standalone offline reproduction script. No credentials, account data or pickle models are needed. This is materially more reproducible than the earlier sample-only package. Raw historical Kalshi responses and the complete weather-history training corpus are not included.

## What changed in this pass

The former Polymarket integration required two-sided books for every bracket and collapsed the result to a normalized median. That systematically discarded events with empty tails. The new implementation preserves every parseable bracket whose book request succeeds, represents missing bids/asks as uninformative interval ends, and **does not invent a midpoint** when either side is missing.

For each bracket, the observed bid/ask gives a quote interval. Subject to unit total probability, the implementation computes lower and upper feasible CDF bounds. Quantile intervals follow from those bounds. These are constraints on a distribution consistent with the quotes, **not confidence intervals for true probabilities**. Inconsistent unit-mass constraints are rejected. A finite median interval no wider than a predeclared 10F is required for the initial feature. Unknown tail probabilities can coexist with a tightly identified median.

Using exactly the same archived NYC API responses, the previous implementation retained 6 of 11 brackets and produced no feature. The corrected version retains all 11 definitions and identifies a median interval of 65.5–67.5F. Fresh source checks produced usable median intervals in all four supported city/date events. That is a four-event source-coverage check, not an accuracy sample or a coverage-rate estimate over time.

A subsequent bounded live Kalshi paper session recorded two usable paired weather/Polymarket feature rows with no session errors and no fills. The real ablation was rerun afterward and still correctly reported insufficient settled paired data. This validates integration and target exclusion, not profitability.

The augmented weather model now receives median displacement from the NWS forecast, median uncertainty width, quote spread, two-sided-book fraction, and a quartile-range feature with an explicit missingness flag. These supplement NWS/GFS/ECMWF forecasts and station observations. Polymarket station/source rules remain distinct from Kalshi's: the features are regional context, never replacement labels or probabilities for Kalshi contracts.

## Models and measured evidence

### Frozen price model

Run `train-103cd0ff5f8a` reproduced `train-a3765d7dccfa` exactly before this pass. Its market-offset network has 32/16 hidden units, seed 1729, 56 completed epochs, best checkpoint at epoch 36, and validation-selected blend 0.75. The adaptive selector also considered calibration/tree alternatives. Splits are 30,806 train, 9,671 validation, 10,438 test and 832 purged rows. The test comprises 74 dates and 296 city/day events.

| Same 74-day test | Model | Market |
|---|---:|---:|
| Log loss | 0.208090656 | 0.208974284 |
| Brier | 0.067651398 | 0.067400294 |

The log-loss difference interval crosses zero: approximately [-0.0024143, +0.0005903]. Test dates were reused. Brier is worse. No promotion is justified.

### Focused challenger and hierarchy

These use the narrower 12–24-hour, 5–95-cent cohort, so their absolute losses must not be compared directly with the table above. Both are research on previously inspected validation dates.

| Later strategy-validation cohort | Log loss | Brier |
|---|---:|---:|
| Existing model | 0.513877912 | 0.170572428 |
| Focused context challenger | 0.513422503 | 0.170463372 |
| Shared calibration | 0.512770705 | 0.170349992 |
| Hierarchical city deviations | 0.512781006 | 0.170322927 |
| Market | 0.512304623 | 0.170244165 |

The focused challenger was selected using a separate inner validation split inside the original training period. It slightly improved the later forecast metrics and changed the fixed 2% edge, 1%-bankroll scenario from -$23.96 to +$5.09. However, only 10 city/day groups traded versus 31 previously, and uncertainty/minimum-event requirements still fail. The hierarchy adds stronger-regularized city intercept/slope deviations to shared calibration. Its mixed metrics do not establish a benefit. Neither candidate replaced the live model.

### Remaining-day weather and Polymarket ablation

The physical challenger models final maximum as the maximum of a noisy observed maximum and a Gaussian remaining-day maximum. Winning brackets are interval-censored targets. It is **not** a correlated hourly path model; the observation-error scale is fixed at 1F. NWS and two deterministic forecasts do not constitute a full forecast ensemble.

The paired ablation uses the same eligible Kalshi rows and chronological partitions with and without Polymarket features. Validation selects blends, including zero, subject to a non-worsening Brier gate. The model fitter uses training prices without the scoring-cohort price filter, avoiding selection of physical training outcomes by market price. Late-hour and extreme-price scoring remain separate in the weather trainer.

The real-data ablation remains blocked by insufficient settled paired observations. A synthetic end-to-end fixture verifies that both arms fit, save selection/artifacts and score identical held-out rows; those synthetic scores are not performance evidence. There is no claim that the new fusion model beats the market or its baseline.

## Execution and statistical safeguards

- Live model-generated intents require a new, matching-model prediction on the execution book. Side stays fixed; probability, cost, depth and risk are reassessed. The refreshed probability is preserved for scoring/replay. External prediction-file signals retain delayed stale-signal behavior.
- Current historical strategy scenarios recompute the price model at later hourly quotes, impose costs, slippage, shared event/portfolio limits and realized-loss entry stops. Historical queue position and full depth remain unverified. Holding to settlement is fixed; early exits are not optimized.
- Eight predeclared edge/sizing combinations are selected on validation only. Eligibility requires at least 30 traded city/day groups and a positive adjusted net-profit lower bound. All currently tested strategies failed. A passing numerical gate would still not establish untouched-test performance.
- Journal hashes detect local alteration against a recorded head, but are not third-party timestamp attestation. Sequential cross-market book requests are not atomic. Receipt-time checks prevent using data before local arrival.
- One event appearing on two exchanges is not a second independent outcome. Grouped dates remain together across the ablation arms. The 90-date minimum is an engineering requirement, not a power calculation or sufficient seasonal coverage.

## Verification and reproduction

The current source suite contains 171 passing Python tests. New CDF-bound formulas were checked against 150 independent linear-program optimizations, in addition to missing-book, inconsistent-mass and ambiguous-median tests. The full synthetic ablation test checks training mechanics. An independently extracted package passed all 171 tests and frontend checks, and its full offline training reproduced the saved reference scores. See `reports/professor-package-verification.json` and `verification-logs/` for the actual outputs.

From the extracted zip, after installing dependencies:

```text
python -m unittest discover -s tests -q
node tests/frontend_checks.cjs
python reproduce_frozen_training.py
```

The reproduction command trains from `training-run/features.jsonl` without network access, validates its dataset hash, and compares the resulting test loss/Brier against saved reference metrics. Other scripts under `research-scripts/` preserve original workspace paths and require datasets not included in this package. They are source references, not standalone reproduction promises.

`MANIFEST.json` lists packaged file hashes. `environment.json` reports the actual runtime. The existing tzdata mismatch remains: runtime 2025.1 versus requirements >=2025.2. Fixed-standard-time weather boundaries do not use that package, but the mismatch still needs cleanup for full environment parity.

## Files worth reading first

1. `helper/polymarket.py`, `tests/test_poly_intervals.py`: quote-interval reconstruction and its assumptions.
2. `helper/poly_ablation.py`, `helper/weather_research.py`, `helper/remaining_weather.py`: cross-source features, interval model, data eligibility and evaluation.
3. `helper/training.py`, `helper/residual.py`, `helper/adaptive.py`: reproducible price-model learning and selection.
4. `helper/hierarchy.py`, `reports/hierarchy-v1/`: partial pooling experiment.
5. `helper/core.py`, `helper/live.py`, `helper/sizing.py`, `helper/strategy_research.py`: execution and profitability assumptions.
6. `reports/poly-interval-replay.json`, `reports/poly-v2-*.json`: actual source responses and feature extraction, not settled prediction results.

## Questions for the professor

1. Is quote-constrained distribution recovery a useful feature representation when fees, asynchronous books and venue-specific risk premia mean bid/ask intervals need not contain physical probabilities?
2. Would modeling a station-to-station forecast residual be preferable to directly adding the regional implied median and spread?
3. Should the interval likelihood be replaced with a latent hourly path model, and how should sparse observed maxima and observation error be handled?
4. What prospective sample size and pre-registration design would adequately distinguish a small net-cost edge from noise and strategy-search effects?
5. Is stronger partial pooling across stations/seasons justified, or should limited capacity go first toward richer physical forecast uncertainty?
6. Which execution assumptions most urgently require measured book histories before any real-money pilot?

Current recommendation: collect the improved paired features and settled outcomes, keep the market baseline and no-trade option, and avoid increasing model complexity or exposure based on the coverage improvement alone.
