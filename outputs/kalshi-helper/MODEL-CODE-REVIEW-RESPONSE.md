# Response to the implementation-level model review

## Result of this pass

The review correctly identifies independent binary fusion as the source of event incoherence. A new, explicitly versioned `EventNeuralPredictor` now trains with event-level categorical probabilities and categorical loss. It is implemented in `helper/joint_research.py`, with synthetic correctness tests in `tests/test_event_joint.py`. No new comparison on the repeatedly inspected historical validation window was run. Existing saved models, predictions and historical reports have not been replaced.

The earlier `predict_event()` interface was a post-processing projection of binary predictions, not categorical training. This implementation addresses that distinction. It is available for a future complete-event training protocol; it is not yet a trained or deployed replacement for WeatherSignal, MarketGuard or ConsensusBlend.

## Exact categorical architecture

For every complete event at a decision timestamp, the existing weather encoder/head still produces a common maximum-temperature distribution. Its interval probabilities and latent features feed the existing bounded market-correction head. Rather than independently applying a sigmoid to binary log odds, the event architecture computes:

`q_i = softmax_i(log(max(market_p_i, 1e-6)) + correction_i)`

The softmax denominator spans all brackets of that event only. It uses **log market probabilities**, not log odds: softmax of binary log odds would distort even a zero-correction baseline. At initialization, corrections are zero and the distribution equals normalized, floored market mass. Every event sums to one. A bracket's probability depends on the other brackets; different events do not share the softmax denominator.

The fusion training objective is categorical negative log likelihood of the single winning bracket, balanced by event/date, plus the existing weighted auxiliary weather interval loss. Optional shrinkage penalizes squared log-probability ratios relative to the normalized market reference. This penalty has a different definition from the legacy binary-logit correction penalty; equal coefficients do not establish comparable regularization strength. The bounded correction also remains a deliberate structural restriction, not a newly optimized hyperparameter.

Inputs must contain both tails and a gap-free, nonoverlapping partition. Duplicate tickers, inconsistent timestamps/stations/events, missing brackets, conflicting weather features, differing forecast paths, invalid market values and zero market mass are rejected. Training requires exactly one winning bracket per event/timestamp. Inference does not read labels. Unlike the historical 5–95 cent comparison cohort, complete ladders must be assembled **before** any price-based trade eligibility filter.

`EventNeuralPredictor` does not accept warm starts in version 1. Reusing old binary-head weights under a different likelihood without explicit versioning would conceal a change in meaning. `SmallJoint.forward()` retains its legacy behavior for old artifacts without `event_coherent=True`.

## What remains outside this implementation

- No categorical artifact has been fitted on the historical development data or promoted.
- The app still serves the saved binary-fusion models in its experimental named-model path. Wiring complete, timestamped live event snapshots and a separately identified event model into that path remains unfinished.
- Existing blend benchmark numbers describe the binary models only. They cannot be relabeled as categorical model performance.
- Coherence guarantees valid probability mass, not calibration, accuracy or positive expected trading returns. A joint temperature model with calibrated dependence remains a separate modeling question.
- A fresh prospective evaluation should predeclare the event dataset, weighting, primary metric, trading rule, fee assumptions and stopping rule. This pass does not reopen the old validation search.

## Two naming and regularization clarifications

The logistic control is now explicitly named `LogisticMarketOffset`; the tree control is `TreeMarketOffset`. Active training call sites use those names. Both modules preserve `MarketOffset` as a compatibility alias so historical pickle class references still load. Old reports are not rewritten to pretend these were different historical models.

The NumPy residual MLP uses Adam with L2 added to the gradient. The PyTorch models use AdamW's decoupled weight decay. These operations differ, and their coefficients are not directly comparable. Both are small feed-forward neural models; the 179-parameter joint network uses a deep-learning framework but is not a large or pretrained deep architecture. The name of the framework does not supply statistical power absent from the event dataset.

## Where the data actually enters the current application

| Path | Kalshi | GFS/ECMWF + IEM | Polymarket |
|---|---|---|---|
| Legacy price-model service | Market features | Not used by its price-only predictor | Not used |
| Experimental named-model service, `LiveNamedModel.predict_signal` | Market probability, spread, contract bounds | Calls `fetch_run`, `archived_feature`, `fetch_observations`, `ObservationIndex`, `observation_innovation`; supplied to the loaded joint model | Not used |
| New event-softmax implementation | Complete-event probabilities and spreads | Shared weather features and forecasts | Not used |
| Prospective Polymarket research | Pairing/settlement reference | Depends on experiment | Data sufficiency remains an independent blocker |

Thus, “weather sources never touch the live model” is accurate for the legacy price-only service but not for the current experimental named-model service. This is a statement about implemented paths, not proof that any particular running session had sufficient inputs or generated profitable signals. The named-model path is gated by supported cities, available station data, forecast availability and an experimental 12–14 hour decision window.

The forecast central value is indeed the equal-weight mean of two deterministic remaining-day maxima. Disagreement is an input feature, not a calibrated ensemble of many weather trajectories. This pass changes neither the data sources nor those weights.

The older tail-price concentration finding should also not be overstated: it identifies where score gains accumulated. It does not by itself prove station observations contained no information. The current named-model benchmark excludes below-5-cent rows, so the older below-2% explanation cannot directly account for that entire comparison.

## Verification scope

Synthetic tests exercise categorical training with finite loss and decreasing training loss, per-event unit mass after optimization, permutation invariance, pickle round trips, label-independent inference, cross-bracket coupling, between-event isolation, incomplete-event rejection and numerical gradient agreement. These are correctness checks, not evidence of market edge. Saved legacy-model predictions are separately checked for compatibility.

Completed checks: all 230 Python tests passed; frontend checks passed; all three saved named models reproduced their 730 validation predictions (maximum absolute difference 2.22e-16). No historical candidate was retrained or rescored for selection.
