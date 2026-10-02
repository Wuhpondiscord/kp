# Why the latest physical model underperforms

This audit identifies and tests a material missing input. It does not establish
that every cause has been isolated, or that the corrected model has a trading edge.
No reserved-test scoring, real-money action, or active-model promotion occurred.

## Measured findings

1. **The forecast training set became very small.** The archive starts in April,
   near the end of the original training partition. Preserving that partition
   leaves 88 city/day events across 22 April dates, with only 22 examples per city.
   Validation spans 67 dates into July. This is not the same training population
   as the earlier price-only models. Their reported losses are not directly
   comparable to this different 12-hour/5–95c cohort.
2. **A station-specific bias changes between periods.** For exact settlement
   targets, NYC's mean settlement-minus-model-center error moves from +0.33 F in
   training to -2.91 F in validation. NYC center RMSE moves from 2.16 to 3.94 F.
   These center errors are diagnostics, not the exact expected value or spread
   of the final maximum distribution. The fixed remaining spread is 2.22 F, while
   the probability reliability/ECE diagnostics show the resulting poor calibration.
3. **Current forecast errors were ignored.** Latest station temperature was in
   the dataset but absent from the mean model. The observed maximum only entered
   the maximum-distribution floor. The model did not ask whether its forecast was
   already too warm or too cold at the observation timestamp. Across exact targets,
   the correlation between this known current error and raw final-temperature
   error is 0.399 in training and 0.556 in validation. Correlation on validation is
   descriptive; the correction coefficient was fitted on training only.
4. **The generic feature matrix is needlessly redundant for this archive.**
   It has 19 columns but rank six. Missing NWS variables generate repeated constant
   indicators, and paired model deviations are linearly dependent. This is not
   evidence of 19 independent learned signals. Duplicate constants also alter the
   effective ridge penalty. A compact six-column control removes that ambiguity.
5. **The information supplied to the weather model is limited.** Its runs are
   initialized 10–12 hours before the decisions. It lacks archived NWS forecast
   revisions, humidity, wind and clouds in this replay. The market baseline is
   contemporaneous. Whether newer forecasts or local grid/station differences
   explain the remaining gap requires another controlled data comparison; the
   current results do not prove which of those is dominant.

The historical price-derived model has a different problem: earlier experiments
found little incremental prediction value beyond market prices and no demonstrated
net-of-cost edge. Better weather prediction and profitable execution are separate
requirements; one unsuccessful physical-model experiment does not show either is
impossible, and neither is guaranteed by adding ML complexity.

## Checks that did not reveal the cause

On the saved training/validation rows: zero conflicts between usable numeric
expiration values and winning brackets; zero future observation-availability
violations; no exact settlement more than 0.5 F below the sampled running maximum.
Complete bracket ladders conserve probability to floating-point precision
(maximum absolute error 5.6e-16). There are 87 usable exact targets in training and
267 in validation; these exact targets are used for diagnostics, not silently
substituted for all censored training labels. This is a scoped audit, not proof
that every source or historical receipt timestamp is correct.

## Fixed experiment and result

The new feature is **observed temperature minus average forecast temperature at
that same observation time**. Already-issued hourly forecast values are linearly
interpolated to the observation time; no later actual observation is used. Both
observation availability and forecast publication allowance must precede the
decision. Missing innovation remains missing rather than becoming zero.

A compact archive-only model adds one mean coefficient to six explicit features:
intercept, three station indicators, model disagreement and the signed model
difference. Observation calibration stays fixed and is identical across arms.
Both new fits use the same 87 usable training events; one of the original 88
events has stale observations and is excluded from both new fits. The saved
existing model had used all 88 events. Evaluation uses the same 730
validation rows / 257 events. There is no spread/threshold/strategy search.

| Validation arm | Log loss | Brier | Event/date-weighted ECE |
|---|---:|---:|---:|
| Existing physical model | 0.731025 | 0.236000 | 0.127774 |
| Compact control | 0.723471 | 0.235028 | 0.127211 |
| Compact plus current-error correction | 0.658286 | 0.221179 | 0.099311 |
| Market midpoint | 0.495074 | 0.164566 | 0.024517 |

The training-fitted correction is approximately +0.327 F in the future forecast
center for every +1 F of observed-minus-forecast current temperature error.
The improved log loss is about 10% below the previous physical model. Against the
compact control, the daily log-loss difference is -0.0652, with a seven-day block
interval [-0.0812, -0.0512]. This supports the usefulness of the missing connection
on these dates. It is still an exploratory comparison on inspected validation
data, not a new confirmatory finding or evidence of profitable trading.

The corrected physical model remains substantially worse than market probabilities.
The change should remain a research candidate. Do not increase exposure, activate
real orders, or widen validation gates on the strength of this result.

## Two-model architecture

The project already separates prediction from a deterministic paper-trading engine.
It does **not** yet have a validated second ML model for execution/trade selection.
The useful separation is:

1. **Weather model:** estimate a station's settlement-temperature distribution,
   uncertainty and coverage using weather information. Score it independently.
2. **Trading/execution model:** combine that distribution with executable prices,
   fees, spread, depth, quote age and forecast age to estimate fill likelihood and
   net expected value at execution. It may abstain. Its job is not to change
   weather labels until historical profit looks attractive.
3. **Deterministic risk layer:** enforce bankroll, shared city/day exposure and
   loss caps independently of either learned model.

A learned trading model needs point-in-time book/intent/fill data. Hourly candle
prices cannot label actual fill likelihood or queue position. Its training inputs
must use chronologically out-of-sample weather predictions, not fitted predictions
from the same weather training labels. Compare it against the existing rule-based
engine on identical opportunities and costs. A profitable unconstrained historical
policy is not a substitute for that test.

## Reproduction and evidence

From the workspace root:

```powershell
python work/diagnose_physical_model.py
```

This uses cached archives only and reads training/validation files; the reserved
test file is not loaded. Outputs are in `reports/physical-root-cause`: protocol,
metrics, model artifacts and exact-temperature error cases. The current-error
feature is implemented in `helper/forecast_archive.py`; the compact research model
is `helper/archive_model.py`. Neither is automatically selected by the live app.

The Python suite passes **199 tests**, including observation-time interpolation,
future-source rejection, missing-data handling, station/unit checks, label-invariant
features, correction direction, serialization and probability mass. The real-data
fit above separately verifies the research path end to end.

Next data priority: extend verified pre-April forecast training coverage across
seasons, then assess the fixed correction on fresh dates. Next trading priority:
collect the execution evidence needed for a second learned model. These address
measured deficiencies rather than asking a more complex network to invent missing
information.
