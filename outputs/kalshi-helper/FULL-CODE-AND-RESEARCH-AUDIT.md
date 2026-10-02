# Full pipeline code evaluation and research diagnosis

## Main conclusion

There is no evidence of one hidden neural-network bug that explains the disappointing results. There are confirmed correctness gaps, but the larger problem is the experiment: a very small, seasonally narrow, market-selected weather dataset is being used to learn both a physical forecast and corrections to a strong contemporaneous price baseline. Execution assumptions then change which contracts are traded. Repeatedly inspecting those same dates cannot resolve whether any apparent advantage generalizes.

My highest-priority recommendation is to decouple **weather training data** from **market trading data**. Train and validate a station-specific weather distribution using a much larger weather history, then test whether its genuinely out-of-sample forecasts add value beyond the Kalshi price at the time a trade could actually execute. This can still produce one app and a shared model interface; it does not require a complex trading neural network.

## Scope and evidence

This is an end-to-end audit of ingestion, timestamps, settlement mapping, features, fitting, prediction, evaluation, selection, paper execution and application wiring. All Python source/test files were parsed and hashed; the detailed inventory is in `reports/full-code-audit/evidence.json`. Targeted manual review concentrated on the model/data/execution modules listed below. It is not a claim that every line of the UI received a formal security or mathematical proof.

Checks completed:

- 231 Python tests passed after one simulator validation regression test was added.
- Adversarial input probes established and then verified fixes for silent simulator truncation, out-of-range probabilities and execution before prediction.
- All 15 saved event-fusion scenario ledgers, not merely their P/L totals, reproduced exactly after the validation fix.
- Training-only gradient diagnostics used existing epoch-200 seed-1729 checkpoints; no model was fitted or selected in this audit.
- No reserved-test forecast scores were computed, no further architecture search was run and no live model was replaced.

## Ranked findings

### 1. Training data, not architecture size, is the dominant constraint

The latest matched comparison contains **486 bracket rows but only 81 city/day events and 22 training dates**. Six brackets are not six independent temperature realizations. Weather across nearby dates and across cities may share additional dependence. Validation has 256 events across 67 dates, repeatedly inspected by prior experiments. The earlier named-model training cohort was 87 events, which does not change this diagnosis.

`archive_replay.py` restricts the weather study to explicit-run coverage from April 2, the nominal 12-hour price horizon, and the original price-model partition. That leaves a narrow April training block. This was a defensible way to preserve historical split membership, but it is a poor basis for training a reusable weather model. Fitting weather solely on dates where suitable market quotes and complete contract ladders survived also creates avoidable selection and missingness constraints.

**What to change:** build a separate station-weather dataset across multiple seasons, independent of whether Kalshi listed a tradable contract. Pair archived *issued forecasts* with exact, documented station targets. Train weather there; retain a separate market dataset for fusion and execution evaluation. NOAA GHCN-Daily includes daily temperature maxima, but a GHCN target is not automatically the contract's settlement target: station identity, units, observation window, revisions and the official settlement source still need reconciliation. [NOAA dataset documentation](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily).

### 2. Two deterministic forecasts are being asked to stand in for a probability distribution

`forecast_archive.py` uses the average of the GFS and ECMWF remaining-hour maxima. Its disagreement variable is the absolute difference between two deterministic products. This is not an ensemble spread sampled from many plausible weather trajectories. Missing meteorological context includes cloud/radiation, dew point, wind/advection and recent forecast revisions in the named baseline feature set. Forecast locations may be grid points displaced from the precise station; the current acceptance check permits differences of up to 0.6 degrees in each coordinate.

These choices are not implementation errors, but they leave the network to learn local bias, uncertainty and weather regime dependence from 81 events. Adding hidden units cannot supply missing independent weather cases or missing physical information.

**Relevant research:** EMOS fits a predictive distribution while correcting forecast bias and dispersion, typically relating uncertainty to ensemble information and fitting with a proper distributional score. It is a stronger starting point than assuming that two model maxima define a calibrated distribution. [Gneiting et al., 2005](https://www.wavestoweather.de/publications/previous_publications/gneiting_2005a.pdf). Operational ensemble systems explicitly generate perturbed realizations to represent uncertainty; that differs from our two-product disagreement feature. [ECMWF ensemble explanation](https://www.ecmwf.int/en/research/modelling-and-prediction/quantifying-forecast-uncertainty).

**What to change:** establish a station-calibrated distributional regression/MOS baseline, then evaluate actual ensemble information or short-range forecast updates. Keep model/version/run/valid-time metadata. Do not assume that adding another named data source provides independent information.

### 3. The physical maximum model contains an untested dependence assumption

In `event_weather.py` and `SmallJoint.forward`, the forecast CDF multiplies a remaining-day CDF by an observation-side CDF. For a maximum, `P(max(A,B) <= t)` equals `P(A <= t, B <= t)`; replacing that joint probability with a product assumes conditional independence of the modeled terms. Shared station conditions and the use of observation innovation can leave dependence. The implementation does not estimate or validate that dependence.

The observation calibration is an empirical completed-station-day reporting discrepancy, not pure sensor noise. Transferring its Gaussian bias/scale to a running observed maximum is explicitly an assumption. Near-close sampling does not logically guarantee zero remaining warming. The code's prior calibration-scope checks are valuable, but they cannot prove the transfer model is physically correct.

**What to change:** validate physical forecasts against exact station outcomes before involving prices. Compare forecast bias, quantile coverage, PIT/rank diagnostics and a proper distributional score by station, season and lead. For a trajectory ensemble, take a maximum within each plausible trajectory over the actual settlement interval, then calibrate the resulting maxima; do not independently sample hourly marginal temperatures and pretend their maxima have correct dependence. Any proposed observation/remaining-day joint model needs its own validation. This is a modeling hypothesis to test, not a proven cause of the current losses.

### 4. We changed more than coherence in the softmax experiment

The new event model fixes probability mass. However, it also changes binary log-odds offsets to normalized categorical log-probability offsets, and per-contract BCE to categorical NLL. Keeping the auxiliary weather coefficient at 1 does **not** keep the relative training pressure constant.

A training-only check of the retained experimental seed-1729 checkpoints shows:

| Quantity at epoch 200 | Binary matched | Event softmax |
|---|---:|---:|
| Weather interval NLL | 0.54978 | 0.54607 |
| Fusion training loss | 0.23173 | 0.68703 |
| Weather-loss encoder gradient norm | 0.02888 | 0.05944 |
| Fusion-loss encoder gradient norm | 0.01975 | 0.06172 |

The fusion losses have different definitions and are not accuracy comparisons. These diagnostics establish that the objectives exert different pressures; they do not establish gradient domination throughout training or identify an optimal coefficient. It would be wrong to conclude “softmax does not work” from this one implementation package. It would also be wrong to keep changing coefficients until the same validation window improves.

`EventNeuralPredictor` rejects incomplete ladders and inconsistent event weather inputs. The saved app families still use binary inference; the complete-event training architecture is not wired into the live service. The earlier projection interface and the new categorical objective are distinct mechanisms. A deployment change must have its own model identity and full-event input contract.

### 5. The strategy needs a larger conditional edge than headline scores suggest

`qualify_signal()` compares probability against an executable side plus a conservative one-contract fee, then imposes a 4-cent edge threshold. It does not trade against midpoint. For example, under the simulator's assumed coefficient, a 49¢ bid / 51¢ ask market has a 50% midpoint; the one-contract YES debit rounds to 53¢, so the model needs at least 57% to pass the 4-cent threshold. With a 2-cent execution-price stress, the corresponding hurdle is about 59%. Aggregate improvements of a few thousandths in log loss need not produce many opportunities that clear that gap.

This example follows the current code's assumptions. Exchange fee schedules can contain product-specific rates and changes, so it is not a universal claim about current or historical Kalshi costs. The official published schedule and series metadata should govern any executable-cost claim. [Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf).

The original binary head limits logit correction to ±0.5. That restriction intentionally limits departure from market odds, particularly at extreme probabilities. It is regularization, not an unexplained inability to learn. Increasing its bound or lowering the entry threshold would create more trades without demonstrating better information.

**What to change:** report incremental information relative to the same-time market and decision-time net edge, then separately report fill-conditioned edge and realized payoff. Decompose the number of usable events, signals, rejected signals, fills, quantity and fees. Keep trade eligibility downstream of full-event prediction.

### 6. Historical scenario and live engine answer different execution questions

`sizing.simulate()` freezes a decision probability and evaluates it against the next hourly quote. It has no historical depth, fill probability, queue position or partial-fill observations. The live engine now requests an execution-time model check before filling pending built-in-model signals. Thus it is no longer accurate to say the live path is simply the same stale-signal experiment with a shorter delay. The historical execution adapter is explicitly price-only and does not rebuild weather inputs.

The simulator also processes simultaneous contracts in ticker order, so a shared capital cap can make lexical ordering affect allocation. This is deterministic but not an economically specified portfolio policy. Cost-valued historical drawdown is not liquidation-value drawdown. The shared city/day cap is useful, but it is not a full payoff-scenario optimizer across correlated outcomes or cities.

**What to change:** maintain two clearly named estimands: frozen-signal latency sensitivity and refreshed-signal execution replay. Build the latter from timestamped books, known-at-time forecast/observation snapshots, arrivals, latency and costs. Queue/fill modeling is needed for passive strategies; a quoted resting order cannot be treated as filled merely because price touched it. PredictionMarketBench is a recent research example of deterministic order-book/trade replay and explicit execution modeling, not proof that its agents would work for our weather contracts. [Paper](https://arxiv.org/abs/2602.00133).

### 7. Confirmed simulator input defects — fixed in this audit

Before the fix, `simulate()` used `zip(rows, probabilities)` without checking lengths. A short prediction array silently dropped rows; an extra prediction could be ignored. It also accepted a probability of 1.2 and an execution timestamp earlier than the prediction. All were reproduced by small diagnostic cases.

The simulator now checks shape, finite [0,1] probabilities, chronological decision/settlement relationships and quote validity. Same-time execution remains allowed only as the existing optimistic same-quote scenario. Before/after probe results are saved in `reports/full-code-audit/`. All 15 prior event-fusion ledgers reproduce exactly, so these defects are **not demonstrated explanations of the already reported model losses**.

### 8. Live freshness has two layers and is weaker than the UI may imply

The named service refreshes its weather cache after 15 minutes, but `fetch_observations()` can reuse a current/future station response for up to one hour. Refreshing the outer cache therefore does not necessarily retrieve newer readings. A further 30-minute availability delay and three-hour observation-age ceiling are applied. A once-per-minute paper poll does not mean once-per-minute new weather data.

The explicit 06Z forecast is intentionally fixed, not “latest available model run.” Historical availability uses an eight-hour assumption rather than verified historical receipt. These choices reduce some leakage risks but also reduce responsiveness; they are consequential if the hoped-for edge is faster incorporation of observations or forecast updates. Code inspection confirms the policies; this audit did not measure an actual missed live price reaction.

**What to change:** make source age, receipt time, expected refresh, run cycle and feature age explicit in the prediction record/UI. Define one cache freshness policy per source and log why a refresh did or did not occur. Measure market reactions from actual arrival timestamps before claiming a speed advantage.

### 9. Validation reuse and incomplete uncertainty accounting remain decisive

The chronological split correctly purges whole dates using the slowest settlement and a 48-hour embargo. That is valuable, but it does not undo repeated inspection of later scores. The new archive mapping preserves whole-date membership; I did not find evidence that it reintroduces purged rows through date mapping.

The seven-date bootstrap uses observed date groups, which can have calendar gaps. It resamples realized trade aggregates; it does not rerun a path-dependent bankroll policy on newly sampled market paths, nor refit the model. With 2,000 bootstrap draws and a 15-comparison adjustment, extreme interval endpoints depend on very few tail draws. Those intervals are useful descriptive warnings but should not be sold as precise deployment confidence bounds. They also do not cover all project-wide selection.

Repeated backtest choice is itself a statistical problem even when each run has a chronological split. [Bailey et al., probability of backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). The project's old dates should remain development evidence. A prospective evaluation must lock configuration and record every attempt, including UI checkpoint training and automatic tuning.

### 10. Broad engineering checks that passed or remain limited

- The previously unexplained station-feature hash mismatch is resolved: secondary settlement rules accounted for the exact 18-row difference. It is not an open nondeterminism finding.
- Train-only scaling, observation availability checks, forecast run checks, frozen hashes and the explicit signal-eligibility flag are substantive protections.
- Gradient checks, event partition tests, fee/risk tests and replay-accounting tests pass. This supports implementation correctness, not profitability.
- `LogisticMarketOffset` and `TreeMarketOffset` now have distinct names with historical aliases. NumPy Adam+L2 and PyTorch AdamW remain mathematically different regularizers.
- The local server binds to loopback and checks Host/Origin on mutation routes. This is a local prototype, not a reviewed internet-facing service. Pickle artifacts are trusted-code inputs; hashes detect alteration but do not make an untrusted pickle safe.
- Historical outputs, training families and live services have separate capabilities. The generic price service does not use weather; the experimental named service does. Polymarket is not a trained input to the named baselines and still lacks sufficient paired prospective evidence.

## What the research suggests building

| Candidate | Why it fits | What must precede a claim |
|---|---|---|
| Station MOS/EMOS-style distributional regression | Small, interpretable bias/dispersion baseline | Multi-season issued forecasts plus aligned station targets; true ensemble features if calling it ensemble EMOS |
| Small neural distributional postprocessor | Nonlinear weather effects and partial pooling across stations | Weather-only benchmarks and materially more independent station-days |
| Short-range guidance/ensemble trajectories | Better representation of intraday maxima and uncertainty | Timestamped archive coverage, model-version accounting, settlement-window maximum mapping |
| Simple fee-aware trading policy | Transparent separation of forecasting and execution | Executable-book replay and positive prospective conditional edge |
| Learned execution/selection model | Could estimate fill probability, adverse movement or whether to abstain | Real order-book/arrival/fill or suitable replay labels; out-of-fold upstream forecasts |

Rasp and Lerch's neural postprocessing work uses auxiliary predictors and station information; the authors' repository documents a decade of forecasts. Its relevance is richer weather supervision and pooling, not the claim that a larger network trained on our 22 dates should win. [Paper](https://arxiv.org/abs/1805.09091), [author implementation](https://github.com/slerch/ppnn).

NOAA HRRR is a high-resolution, hourly updated source, and its official page points to archives extending back to 2014. This makes it worth an explicit coverage/size/target-alignment pilot, not an immediate assertion that a compatible multi-year dataset has been downloaded. [NOAA HRRR](https://rapidrefresh.noaa.gov/hrrr/). NWS also publishes probabilistic temperature guidance derived from the National Blend of Models, a useful independent benchmark where the correct vintage is available. [NWS probabilistic guidance](https://www.weather.gov/mlb/probabilistic).

Pretrained global weather models are another source of trajectories, not drop-in weights for the small contract network. GenCast forecasts global probabilistic weather and publishes code/weights; its target and spatial/temporal scale differ from exact settlement-station daily maxima. Downscaling, issue-time archives and local calibration would still be needed. I would not make it the next implementation step. [GenCast paper](https://doi.org/10.1038/s41586-024-08252-9).

For trading, prediction-market learning papers distinguish the probabilistic prediction, agent preferences/risk and the market mechanism. That supports a modular design, not a requirement for a second deep network. [Hu and Storkey, 2014](https://proceedings.mlr.press/v32/hu14.html). Kelly sizing is sensitive to probability estimation errors; it is a capital-allocation rule, not a source of predictive edge. A prediction-market-specific analysis discusses that sensitivity. [Kelly application preprint](https://arxiv.org/abs/2412.14144). These papers are conceptual support, not empirical validation of a Kalshi weather strategy.

## Recommended order of work

1. **Freeze this model-search branch.** Preserve the current cohorts, checkpoints and complete search ledger. Keep profitable historical point estimates descriptive.
2. **Specify the weather target and build its data independently.** Exact station/day/rounding/source; multi-season issued forecasts and station labels; audited corrections and publication/arrival times.
3. **Benchmark weather without market prices.** Persistence/current maximum, raw guidance, MOS/EMOS-style distribution and one small pooled postprocessor. Use rolling seasonal blocks, event-level scores, CRPS where exact targets support it, interval coverage and station/horizon diagnostics. Choose and document this protocol before scoring fresh dates.
4. **Construct one complete-event prediction contract.** Supply all brackets together, keep immutable source provenance and normalize market information explicitly. Test weather's incremental value against same-time market probabilities. If training stages are separate, use time-respecting out-of-fold weather predictions for fitting downstream fusion.
5. **Align historical and live execution.** Collect timestamped depth and source arrivals; evaluate refreshed weather at execution; specify portfolio allocation ordering, actual fees and conservative fills. Keep stale-signal scenarios as separate diagnostics.
6. **Only then evaluate trading and sizing.** Require a positive, uncertainly quantified net edge on fresh data. Consider a learned selection/execution model only after enough such data exists. No added leverage, threshold relaxation or bankroll increase solves missing edge.

The practical failure is not “weather prediction is impossible.” It is that station weather prediction, market incremental-information prediction and executable trade profitability are three different questions, and the current dataset and evaluation cannot yet support all three. The corrected code is useful infrastructure; the next meaningful improvement should be in target-aligned independent weather data and execution evidence.
