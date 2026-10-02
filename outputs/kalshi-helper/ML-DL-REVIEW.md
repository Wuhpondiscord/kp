# BetCheck: ML/DL technical review for academic feedback

Code and saved-results audit: September 23, 2026. This describes the implementation, distinguishes experiments from deployed behavior, and proposes research priorities. It is not a claim that the system beats executable market prices. No new architecture was trained for this documentation audit.

**External-review follow-up:** read [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md) for later diagnostics, packaging fixes, structured signal control and the discovered dataset-reconstruction mismatch. Original results below describe their saved runs; reconstructed-data results are reported separately.

## 1. Executive assessment

BetCheck currently has a functioning supervised probabilistic forecasting pipeline for **daily high-temperature contracts in New York, Chicago, Miami and Denver**. It does not yet have a general model for every category shown in the market browser.

The live predictor is primarily a **market-price correction model**: it learns whether historical outcomes justify adjusting the market's probability, conditioned on price, spread, time remaining and city. It cannot independently understand an approaching weather system because weather observations and forecast fields are not inputs to that live model.

Separate experiments learn from archived GFS/ECMWF forecasts and station observations. The newest station experiment improves historical log loss and Brier score, but its tested trading scenarios lose money. Its incremental improvement over a comparable price-only tree model is statistically uncertain. It is deliberately not installed into live picks.

My assessment is that the largest opportunities are **better timestamped domain data, coherent event-level probability distributions, and stronger evaluation**, followed by architecture changes. The 51,747 training/evaluation rows do not represent 51,747 independent weather outcomes. There are only 1,468 city-day events across 367 dates. A substantially larger network would have little independent information to learn from at present.

## 2. What exists versus what was proposed

| Capability | Current implementation |
|---|---|
| Supervised market calibration | Implemented: logistic/beta baselines, market-offset calibrators, residual neural network and boosted trees |
| Train/validation/test workflow | Implemented with chronological dates, checkpoint selection, manifests and saved weights |
| Weather forecast post-processing | Experimental Gaussian distribution models using forecast values and resolved winning brackets |
| Station observations | Experimental market-offset trees; not connected to live inference |
| Pretrained neural backbone or weights | None. All project neural weights are trained from scratch |
| Physics forecasts | Existing GFS/ECMWF outputs are used in experiments; this is not fine-tuning those weather models |
| Transformers, RNNs, CNNs, LLM embeddings | Not implemented |
| Unsupervised learning, regime detection, hierarchical Bayes | Not implemented |
| Reinforcement learning | Not implemented; practice trading uses fixed rules |
| Mentions, post counts, elections, seasonal totals, entertainment, economics | Browsing/planning is broader than trained-model coverage; no trained domain models for these categories |
| Weekend rain | NWS context, not a trained contract-level rain probability model |
| Replay | Evaluates supplied timestamped predictions; does not itself train a model |

The original notes correctly emphasized underlying distributions, partial pooling, nested temporal evaluation and pretrained features. Several remain proposals, not completed capabilities. In particular, the current live model's independent binary outputs do not implement the notes' shared event-distribution design.

## 3. Data sources, labels and joins

### Kalshi market data

`helper/kalshi.py`, `history.py`, `research_store.py` and `storage.py` fetch and store market metadata, settled outcomes and hourly YES bid/ask candles. Live practice separately collects current order books. Raw responses are journaled; research records retain download times.

The label is `y=1` for a settled YES contract and `0` for NO. History extraction requires finalized/settled binary dollar contracts with recognized outcomes and settlement timestamps. It checks payout consistency when that metadata is present. Canceled/nonbinary/unsupported outcomes are excluded.

The main training path requests observations at 26, 24, 18, 12, 6 and 3 hours before market close. It uses the latest completed valid candle at or before each cutoff, rejecting candles over an hour behind the cutoff. The actual time remaining is retained. Consequently the saved model's supported window is about 3.98–26.98 hours, not exactly 3–26 hours.

Each row contains market/event identifiers, series, UTC close-date group, decision and settlement times, bid, ask, midpoint, spread, nominal horizon, actual hours remaining, previous price movement, candle volume, label and a later quote for execution scenarios. The later quote is not a model input.

Important limitations:

- Midpoint is a useful forecast benchmark, but is not generally a price at which a bet can be executed.
- Candle end time is treated as feature availability; original ingestion time is unknown.
- Final metadata is obtained retrospectively. Historical rule revisions and close-time changes are not independently reconstructed.
- Discovery has a three-page cap per endpoint/series and can stop after an older page. Its manifest records caveats; this is not a demonstrated census of all eligible markets.
- Stored market history is append-once per ticker. Corrected upstream data are not automatically reconciled/versioned.
- The downloader's requested window does not trim older records already stored. Training uses the accumulated archive, which explains why a nominal 365-day request can yield 367 dates.
- Missing candles, missing source data and rule filters can produce selection bias. Report exclusion rates by city/date/horizon, not just total row count.

### Archived numerical weather forecasts

`weather_model.py` accesses Open-Meteo's Previous Runs API for Fahrenheit two-meter temperatures. `weather_experiment.py` uses `gfs_seamless` and `ecmwf_ifs025`, selecting day-one or day-two fixed-lead data according to an assumed availability rule. It takes the maximum of 24 hourly temperatures over the station's local-standard-time settlement day.

These are **two deterministic forecast products**, not individual members of a calibrated probabilistic ensemble. Their disagreement is a feature, not a validated uncertainty distribution. Previous Runs provides fixed-lead values; a complete 24-hour vector need not represent one original model initialization. Original run structure should be checked using an explicit initialization-time archive. [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api), [Historical Forecast documentation](https://open-meteo.com/en/docs/historical-forecast-api).

Current code adds a six-hour publication allowance when inferring availability. This is a conservative assumption, not historical proof. It requires complete forecast windows and both sources; even the fresher-source experiments require the older GFS comparison input, which can unnecessarily reduce their sample universe.

Chicago originally used O'Hare coordinates despite contracts specifying Midway. This was corrected, and station-name checks were added. Older weather reports using the previous mapping should not be relied on without rerunning. Other important remaining checks are gridpoint versus station differences, rounding rules, report-provider changes and whether the official daily maximum differs from the maximum of hourly reports.

### IEM ASOS station observations

`station_observations.py` downloads NYC, MDW, MIA and DEN observations, including temperature and METAR text. It rejects invalid values, wrong stations, known corrected reports and conflicting timestamps. It extracts six-hour maxima where available. Features only use observations before the decision minus a 30-minute delay; a 90-minute delay is also evaluated. Observations older than three hours are treated as unavailable.

The running maximum excludes the prior settlement day. A six-hour maximum is only used when its entire measurement window belongs to the current settlement day. Raw hashes and retrieval metadata are saved.

However, these are retrospective archives. Excluding explicit corrections cannot detect every backfill or establish first publication time. Hourly and six-hour reports are imperfect proxies for final settlement maxima. See [IEM archive documentation](https://mesonet.agron.iastate.edu/request/download.phtml) and the [NWS ASOS manual](https://www.weather.gov/media/asos/aum-toc.pdf).

### NWS rain context

`weather.py` retrieves station/point forecasts and displays precipitation periods. It returns no trade-eligible contract probability. A maximum/sum bound over period probabilities is not a learned weekend-rain forecast, and rain-period dependence and settlement thresholds remain unresolved.

## 4. Features actually used

The main model receives **nine numerical inputs** from `evaluation.features()`:

1. Logit of midpoint, clipped for numerical stability.
2. Midpoint probability.
3. `p(1-p)` (a deterministic transform, not measured market uncertainty).
4. Bid/ask spread.
5. Hours remaining divided by 24.
6. Four city/series indicators.

`prior_move` and `quote_volume` are collected but not used by this feature function. The main model also lacks temperature thresholds, current station temperature, forecast changes, seasonality, weather regimes, liquidity depth, order-flow information and cross-bracket relationships.

The station tree adds 14 inputs: observation presence; lower/upper thresholds relative to the observed maximum; unbounded-tail indicators; maximum minus current temperature; two-hour temperature trend; observation age; local day hour; GFS and ECMWF minus observed maximum; forecast disagreement; and thresholds relative to the mean forecast. Some distances are clipped. The observation count is recorded but not used by the tree.

Station trees operate on individual contract rows. There is no explicit physical support constraint preventing a forecast from assigning mass below an already observed daily maximum, nor any event-level normalization. These are high-value review topics.

## 5. Model architectures and objectives

### Residual neural network: the current live winner

`residual.py` implements a small NumPy network with train-only standardization, two tanh hidden layers and one output. Default dimensions are **9 → 32 → 16 → 1**, totaling **865 trainable parameters**. Supported alternatives are 16/8 and 64/32 hidden units.

Its basic forecast is:

`q = sigmoid(logit(market_probability) + network(features))`

The output layer starts at zero, so initial predictions match market probabilities up to endpoint clipping. It learns a correction to log odds, not a regression target equal to `y-p`. Training uses date-weighted binary cross-entropy, weight L2 regularization and a manually implemented Adam optimizer. Every date has equal total training weight despite different quote counts. The effective regularization coefficient is configured `alpha / number_of_training_rows`.

Defaults: learning rate 0.001, batch size 128, alpha 1, seed 1729, maximum 150 epochs and patience 20. Validation daily log loss selects the saved checkpoint. A tiny improvement threshold controls patience; the absolute best checkpoint is still saved. The recent run stopped after 56 epochs and selected epoch 36.

There is no pretrained backbone, GPU training, dropout, attention, sequence encoder, learning-rate schedule, multi-seed ensemble or learned uncertainty over parameters. Numerical-gradient tests check the custom backprop implementation. CPU-sized training is appropriate to current scale, but custom optimizer code increases maintenance responsibility.

### Adaptive selection and calibration

`adaptive.py` compares the neural checkpoint against six regularized offset calibrators (beta/context/spline bases, each with two penalties) and two market-only tree variants when enough rows exist. It evaluates probability-space blend weights 0.25, 0.5, 0.75 and 1, with the unchanged market as a candidate.

Candidates must have validation Brier score no worse than the market, then win on validation daily log loss. The recent winner is the neural model at weight **0.75**. A better log-loss candidate can be rejected by the Brier constraint; this is an intentional multi-metric selection rule worth discussing.

The calibrators use L-BFGS-B, date weighting, L2 penalties and market logits as offsets. Context features add horizon, spread and city interactions; spline features add fixed logit hinges. This is a bounded candidate search, not unrestricted architecture search or an autonomous learning system. Validation selects epochs, candidate family and blend, so repeated tuning can still overfit validation.

### Other comparison models

`evaluation.Predictor` retains market-only, logistic calibration, beta calibration, small histogram-gradient-boosting classifiers and a 16/8 ReLU MLP. Its comparison path uses three expanding chronological validation folds, then refits on development data. Most of these older estimators fit unweighted rows despite being scored by daily means, unlike the newer residual models.

`training.py` also retains a legacy sklearn MLP classifier trained directly on outcomes, with Adam and train-only scaling. These older paths are useful controls but have different selection procedures. Reports must identify the exact path rather than treating all Auto/training results as interchangeable.

### Weather distribution model

`WeatherDistribution` fits a Gaussian location and scale. Location depends on forecast temperature, seasonal sine/cosine terms, city and forecast lead. Log scale depends on city, lead and, for the multi-source version, model disagreement. Standard deviation is bounded to 0.5–20°F.

Training maximizes the probability assigned to the **winning contract interval**, rather than pretending its midpoint is an exact measured temperature. Repeated event/lead combinations are deduplicated and weighted so each event contributes equally. L-BFGS-B optimizes regularized interval likelihood; at least 80 winning events are required.

The output is `P(lower ≤ temperature < upper)` from Gaussian CDF differences. Validation chooses GFS-old/GFS/ECMWF/combined sources and a log-odds blend with market probabilities. This blend differs from the probability-space blend used for the neural models. The older implementation in `weather_model.py` uses constant variance. `models.py` also contains a simple Gaussian probability converter for user-supplied forecasts; that converter does not learn weights.

A Gaussian distribution is a strong small-data baseline, but may miss asymmetric or multimodal forecast errors. The weather-only distribution can give coherent bracket probabilities when intervals form a valid partition; blending each bracket independently with market odds can break that property. It is not yet a full ensemble EMOS implementation.

### Station-informed Newton boosting

`StationOffset` adds trees to market log odds. It fits weighted logistic Newton updates using sklearn decision trees, then explicitly recalculates regularized leaf values. Candidates have four or seven leaves, minimum leaf size 100, learning rate 0.05, leaf penalty 10 and up to 120 trees. Validation selects checkpoints every five iterations and chooses a blend.

Only rows with usable station observations contribute to fitting. Missing observations produce exactly the market probability at prediction time. The selected historical candidate has four leaves, 120 trees and weight 1. This sparse model is ML, not DL. `MarketOffset` reuses the architecture with price features only and is eligible for the main live pipeline.

## 6. Splits, evaluation and what results mean

Main epoch training splits by UTC close-date groups approximately 60/20/20. It purges complete dates if settlement timestamps violate a 48-hour embargo before the next partition's earliest prediction. Scaling uses training data only; validation labels influence checkpoint/blend selection and are included in the saved trained-through timestamp. Test labels are not arguments to fitting functions.

Latest main run `train-d556814b896c`:

| Partition | Dates | Quote rows | Range |
|---|---:|---:|---|
| Train | 217 | 30,806 | 2025-09-20 to 2026-04-24 |
| Validation | 70 | 9,671 | 2026-04-28 to 2026-07-06 |
| Test | 74 | 10,438 | 2026-07-10 to 2026-09-21 |

832 rows were purged. The test contains 296 events and 1,776 contracts. Multiple brackets and horizons are correlated; grouping by date helps, but this is only about a year of seasons and four stations. The sequence also creates seasonal differences between partitions.

Scores are daily means of log loss and Brier loss, then averaged across dates. Log loss measures probability quality with a strong penalty for confident errors; Brier score is squared probability error. Calibration tables use ten probability bins but row-weighted counts, which do not match the daily weighting of headline metrics. Calibration curves alone do not establish tradable advantage. [scikit-learn calibration documentation](https://scikit-learn.org/1.8/modules/calibration.html).

Uncertainty uses 2,000 circular moving-block bootstrap draws of paired daily loss differences with seven-day blocks. This retains same-day grouping and some temporal dependence. It does not correct for all research attempts, guarantee that seven days is appropriate, or treat gaps as explicit missing calendar days.

### Saved results: compare like with like

| Main 10,438-row test | Log loss | Brier |
|---|---:|---:|
| Market | 0.208974 | 0.067400 |
| Selected neural blend | 0.208091 | 0.067651 |

The neural model improves log loss slightly but worsens Brier score. Its paired log-loss interval crosses zero: [-0.002414, 0.000590]. It has not demonstrated reliable superiority.

| Station experiment: same 5,274 test rows | Log loss | Brier |
|---|---:|---:|
| Market | 0.199210 | 0.064322 |
| Existing neural model | 0.198429 | 0.064612 |
| Price-only tree diagnostic | 0.197622 | 0.064277 |
| Station-informed tree | 0.197390 | 0.064206 |

The station test also spans 74 dates and 1,776 contracts, but uses fewer horizons and requires forecast availability. Do not compare its absolute loss with the main test. Its market-relative log-loss interval is [-0.002553, -0.000985]. However, these historical dates have been reused; this is exploratory evidence. The station-versus-price-tree interval includes zero, and that control was fitted after observing the primary result. It suggests architecture explains much of the improvement, not proof that station data is useless.

### Forecast accuracy versus trading performance

`quote_scenario()` uses later hourly quotes, one-contract entries, assumed fees/slippage, a 4-point minimum net edge, event caps and a shared cash budget. It has no historical depth or queue position. Station scenarios returned -$13.40, -$18.55 and -$17.71 from $1,000 under the three saved cost assumptions. They are hypothetical fills, not executable return estimates.

The live paper engine is a separate test: it collects books, pins the model selected at session start and applies execution/risk rules to simulated money. It neither retrains online nor learns an execution policy. Neither a short profitable paper run nor a successful software test establishes a statistical edge.

## 7. Inference, guardrails and reproducibility

`ModelService`, `coverage.py` and `model_library.py` load selected local artifacts, check hashes and training-time/coverage restrictions, and provide market fallback when prediction is unsupported. The default careful strategy requires the validation gate; experimental practice permits unproven models. Coverage is derived from categories and horizon ranges represented across partitions, not learned out-of-distribution uncertainty. It does not establish support for every combination of price, weather regime and horizon within those ranges.

The station model is not a normal live-library selection because it requires forecast/observation features absent from the live adapter. Simply removing coverage gates or loading its pickle into the price-only adapter would not make it valid.

Explanations in `explain.py` report price adjustments, costs, validation results and input perturbations. They are sensitivity diagnostics, not causal explanations or SHAP attributions. No language model generates the predictions or explanations.

Research artifacts include data hashes, source hashes, model checksums, settings, split manifests, epoch histories, reports and prediction records. These are useful foundations. Remaining reproducibility gaps include incomplete dependency/environment capture, hashes that cover only selected source files, raw data not embedded in every experiment, mutable shared state, old reports using earlier geography, and the absence of an immutable global registry of every hypothesis tested. Pickles need matching code/dependencies and should not be treated as a portable model format.

`requirements.txt` pins NumPy 2.1.1, SciPy 1.15.2 and scikit-learn 1.6.1; tzdata has a lower bound. The environment captured for this review uses Python 3.13.0 and matches those three numerical-library pins, but has **tzdata 2025.1, below the declared requirement of 2025.2**. This is a reproducibility discrepancy to reconcile; it does not by itself establish a prediction error. The review package records actual versions in `environment.json`.

## 8. Testing: what is established and what is missing

The last implementation pass passed 127 Python tests plus frontend checks. Tests cover timestamp/quote validation, data extraction, chronological boundaries, train-only scaling, numerical gradients, learning a known synthetic bias, serialization, date weighting, station matching, observation cutoffs, missing-data fallback, model selection, scope and paper accounting. Synthetic examples are valid software tests, but are not evidence of real-market performance.

Still needed for stronger scientific claims: multi-season out-of-time tests; nested temporal selection for the default architecture path; multiple random seeds; leave-station-out checks when enough stations exist; systematic exclusions/missingness analysis; event-level coherence tests; a model-independent raw-source audit; frozen prospective predictions; and execution measurements with contemporaneous fee schedules and depth. Passing the current suite does not validate all source accuracy or every live integration.

## 9. Prioritized improvements to discuss

| Priority | Proposal | Why it may matter | How to test |
|---|---|---|---|
| 1 | Record initialization, release, ingestion and decision timestamps; preserve official settlement reports and rule versions | Reduces hidden timing/target errors | Manually audit sampled event timelines and prohibit retrospective replacements |
| 2 | Build multi-year station forecast–observation pairs, including days without Kalshi listings | More independent weather outcomes | Train weather post-processing on older seasons; evaluate only eligible future listed contracts |
| 3 | Model one temperature distribution per city-day/horizon | Shares evidence across brackets; enforces probability coherence | Compare against price-only controls on identical contracts, with event-level scores and calibration |
| 4 | Add recent station forecast error, forecast revisions, cloud/wind/dewpoint and genuine ensemble summaries | Potential independent information beyond price | Predeclared feature-group ablations on temporal validation |
| 5 | Nested walk-forward selection and a genuinely fresh final period | Limits researcher/validation overfitting | Freeze a small candidate set; keep an experiment registry |
| 6 | Measure net execution value and uncertainty on the bets actually selected | Small average accuracy gains may not clear spread/fees | Record books and delayed execution, report skipped/unfilled bets and event-clustered uncertainty |
| 7 | Challenge the Gaussian model with distributional boosting or a small shared-station neural head | More flexible bias/scale without an enormous model | Match dataset, information set and tuning budget to the simple baseline |

For weather, a pretrained model is more plausibly an additional forecast source than a replacement for the complete pipeline. First test whether a source adds information beyond GFS/ECMWF and the market. Published station-level neural forecast post-processing supports using small distributional networks and shared station information, but does not establish profit on Kalshi. [Rasp and Lerch](https://arxiv.org/abs/1805.09091). Permutation-invariant models become relevant if actual ensemble members are collected. [Ensemble post-processing research](https://arxiv.org/abs/2309.04452).

Pretrained time-series models are possible future controls for hourly post counts or streams, once timestamped sequences exist. They are not a drop-in backbone for the current nine-feature contract classifier. Assess pretraining-date contamination and calibration before interpreting results. [TimesFM research](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/). No such model is currently installed.

For mentions, first build speaker/event/word datasets with settlement-aware transcript labels; compare rate or hierarchical models before adding frozen text embeddings. For post counts, start with an overdispersed count model conditioned on counts so far. For cumulative weather, simulate the remaining period conditional on accumulated observations. These are distinct tasks, not extensions obtained by switching the current model's category flag.

## 10. What to share with your professor

Start with this document, `MARKET-EDGE-EXPERIMENT.md`, the three results JSON files and the original ML design note. Ask for review of the scientific design before asking which large architecture to use.

| Files | What your professor can assess |
|---|---|
| `helper/history.py`, `kalshi.py`, `research_store.py`, `storage.py` | Data universe, labels, candle timing, persistence and revisions |
| `helper/evaluation.py`, `training.py` | Features, losses, splitting, candidate selection, bootstrap and trading scenarios |
| `helper/residual.py`, `adaptive.py` | Neural math, weighting, regularization, calibration and blends |
| `helper/weather_model.py`, `weather_experiment.py` | Forecast availability, station/rule matching, censored targets and distribution fit |
| `helper/station_observations.py`, `station_model.py` | Observation leakage controls, new features, boosting and fallback |
| `helper/coverage.py`, `model_library.py`, `live.py`, `core.py`, `explain.py` | Whether the tested model is used consistently in practice |
| `tests/test_residual.py`, `test_training.py`, `test_research.py`, `test_feature_integrity.py`, `test_station_model.py`, `test_model_upgrade.py` | Important numerical and methodological regression checks |
| `reports/adaptive-training-comparison.json`, `station-model.json`, `station-ablation.json`, `station-prospective-protocol.json` | Current results, competing models, caveats and future protocol |
| `requirements.txt`, run config, split manifest, epoch CSV, actual environment and representative real rows | Reproducibility and training behavior |

A local **professor-review.zip** accompanies this writeup. It contains the source modules and tests, focused reports, original notes, the main run's config/split manifest/epoch results, actual environment versions and a small real-data example. It intentionally does not include the full SQLite store, all forecast caches or model pickles. It is a review package, not a complete offline reproduction dataset. For a full rerun, provide a separately exported research-data snapshot with raw forecast/observation payloads and initialization/availability provenance; a tiny example is insufficient.

Suggested questions:

1. Is binary market-offset learning the right formulation, or should we prioritize a shared event-level temperature distribution and joint market calibration?
2. Are our timestamp assumptions sufficient for any historical claims? Which archive/settlement sources would you require?
3. What is the effective sample size given dates, cities, brackets and horizons, and how should that constrain model complexity?
4. Is interval-censored likelihood acceptable until exact settlement measurements are available? How should we model rounding and observation noise?
5. How would you structure nested temporal validation, seed comparisons and multiple-testing control with this amount of data?
6. Is the validation Brier constraint useful, and should a separate calibration window be reserved?
7. What experiment best separates added weather information from architecture-only calibration gains?
8. What evidence would justify advancing from improved probability scores to a credible net-of-cost paper-trading claim?

Suggested opening message: “I’m building a probabilistic prediction-market research tool using real Kalshi data and simulated money. The live model corrects market log odds; a separate weather/station experiment has small historical scoring gains but negative trading scenarios. Could you review the data timing, target formulation, temporal evaluation and model complexity, and recommend the highest-value next experiment? I’ve included the code, results and limitations rather than assuming a larger neural model is the answer.”
