# Bias, station uncertainty, and disagreement-dependent spread

2026-09-24. Research implementation and development-data audit. No demonstrated
real-weather calibration improvement and no change to the no-proven-edge finding.

## Bias correction verified

The observation component uses
`Phi((bound - (observed_max + bias_f)) / sigma_f)`.
For settlement minus observation residuals with positive mean, the corrected
center moves upward. Subtracting bias from the observed temperature itself would
give the wrong sign. The existing implementation already had the correct sign;
it now has a direct numerical regression test, including station-specific means.

## Uncertainty and station structure

The 575 eligible development pairs span 31 station-month clusters and eight
calendar months. We resample entire months within each station (2,000 replicates,
seed 1729), retaining all eligible days within each sampled cluster. This preserves
within-month persistence rather than pretending that all days are independent.
The pooled point SD remains 0.698 F; its percentile 95% interval is **0.277–1.102 F**,
with bootstrap standard error **0.252 F**. The corresponding mean discrepancy
interval is +0.040 to +0.193 F.

A separate synchronized-calendar-month bootstrap resamples the same months across
all stations to retain cross-station dependence. Its pooled SD interval is
0.273–1.050 F. Neither design resolves persistence across month boundaries,
incomplete seasons, nonrandom missing exact targets, or the small number of months.

| Station | Days | Bias F | SD F | Station-month bootstrap 95% SD interval F |
|---|---:|---:|---:|---:|
| NYC | 148 | +0.103 | 0.450 | 0.222–0.695 |
| Chicago Midway | 151 | +0.092 | 0.423 | 0.232–0.633 |
| Miami | 124 | +0.042 | 0.311 | 0.126–0.460 |
| Denver | 152 | +0.165 | 1.180 | 0.087–1.933 |

These are clearly different point estimates, but their intervals overlap and
Denver's estimate is sensitive to a few large discrepancies, including the
previously flagged 13 F residual. This does not establish a stable geographic
variance effect. Estimating four scales rather than one still adds estimation
uncertainty, even though they are estimated outside the physical-model optimizer.

`reports/observation-calibration/calibration-station.json` now enables separate
bias and SD by station. Each needs at least 80 eligible pairs; otherwise the
declared fallback is pooled calibration. `calibration.json` retains the pooled
option. Both record block uncertainty, source hashes and eligible events.
The model checks training scope and copies the artifact into its protocol.
Neither option is silently activated in the live system. The full-day residual
still measures reporting/sampling discrepancy; transferring it to an early-day
running maximum is not established.

## One spread challenger implemented

The fixed-spread model remains available as a control. The challenger uses:

```
d = min(model_disagreement_f / 5, 4)
sigma_remaining = exp(log_base_sigma + beta * d)
```

The existing base-scale bounds remain 0.5–20 F. One new coefficient, beta, has
predeclared bounds [0, 1], so disagreement can widen the distribution. The feature
cap at 20 F prevents unlimited exponential extrapolation. These bounds are design
assumptions, not tuned validation results. The mean model and its regularization
are identical between arms, including the existing disagreement mean feature.
Observation biases/scales remain fixed externally. Old serialized constant-spread
models still have their original parameter interpretation.

Missing or invalid disagreement is not treated as zero. The paired comparison
excludes those rows from both arms after forming the original chronological
partitions. It fits both arms on training data, reports validation results, and
never scores the reserved test. It reports log loss, Brier, fixed-bin reliability,
event/date-weighted calibration error, and strata below 2 F, 2–5 F and above 5 F
disagreement. ECE is descriptive and bin-sensitive; Brier is not a pure calibration
measure. No automatic winner or live promotion is produced.

The variance-link motivation is consistent with
[Gneiting et al.'s EMOS work](https://stat.uw.edu/research/tech-reports/calibrated-probabilistic-forecasting-using-ensemble-model-output-statistics-and-minimum-crps).
Our two deterministic model maxima are a spread proxy, not a full forecast
ensemble; their disagreement's predictive value must be measured locally. This
log-linear link is a research alternative, not an implementation of that paper's
exact variance model or CRPS fitting procedure. A motivated hypothesis narrows
the search but does not erase prior data inspection or multiple-comparison risk.

## What was actually tested

The real-data comparison was invoked with station-specific calibration and saved
to `reports/weather-spread-comparison`. It correctly returned **insufficient_data**:
zero settled eligible rows/dates and six unsettled candidate records. The existing
minimum of 90 settled date groups remains in force. Therefore **no real-data refit
or validation improvement can be reported**. An archived daily maximum was not
substituted for a missing prospective remaining-day forecast.

The 187-test Python suite passes. The new tests verify bias direction, station
dispatch, scale monotonicity, missing-data refusal, zero-slope equivalence, old
model compatibility, probability mass, block-bootstrap reproducibility and refusal
to fit with insufficient data. A synthetic experiment fits both models to 400
generated events and evaluates 200 separate generated events with a known spread
relationship. It recovers a positive spread coefficient and improves Brier/log
loss there. This is a mechanics test only. The full comparison/reporting path is
also tested with an unusable reserved-test sentinel to catch accidental test use.

## Reproduce

From the workspace root:

```powershell
python work/calibrate_observation_discrepancy.py
```

From `outputs/kalshi-helper`, choose a new output directory:

```powershell
python -m helper --data ../../work/browser-stage-data compare-weather-spread --output reports/new-spread-comparison --observation-calibration reports/observation-calibration/calibration-station.json
python -m unittest discover -s tests -q
```

The current contribution is explicit assumptions, stratified estimates, quantified
uncertainty and a testable variance-link model. Improved real-world calibration,
tradability and positive net-of-cost performance remain separate unproven claims.
