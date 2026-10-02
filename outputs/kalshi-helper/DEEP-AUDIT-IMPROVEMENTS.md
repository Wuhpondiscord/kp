# Follow-up implementation audit

This pass improves consistency and reproducibility. It does not retrain or promote a model, and introduces no new claim of market outperformance.

## Changes

**Zero-quantity quotes:** book validation, forecast baselines and live model prices now ignore levels with zero quantity. Previously a price with no available contracts could affect the midpoint even though the simulator's execution code ignored it. Zero-size levels also no longer falsely make a book appear crossed. Tests cover both conditions.

**Inference validation:** ModelService rejects nonfinite, out-of-range or crossed dollar quotes before using them. It also rejects invalid model outputs before flagging them as signals. These checks cover the model boundary independently of upstream parsing.

**Scoring the signals actually used:** the simulator retains its first-forecast metric for compatibility, but now separately reports the actual filled-order forecast against (a) its decision-time market midpoint and (b) its execution-time midpoint. Fill records include probability, signal time, both midpoints and signal age. This distinguishes initial forecast quality, trade selection and stale-signal performance without rewriting old metrics. The practice page exposes the comparisons under an optional accuracy panel. Unsettled or unpaired predictions do not receive scores. These remain descriptive market averages; event dependence and trading selection still matter.

**Stable Newton leaf corrections:** newly fitted boosted trees use sklearn's public `apply()` routing plus an explicit dictionary of regularized leaf corrections. They no longer overwrite the internal `tree_.value` buffer. Existing plain sklearn trees still work through their `predict()` interface. Tests check that the underlying estimator is unchanged and the wrapper survives serialization. This is an implementation improvement, not a new architecture claim.

**Frozen main training data:** new main-model runs save `features.jsonl` alongside the split manifest, plus Python and numerical-library versions in `environment.json`. Station runs already gained feature snapshots in the previous pass. Old artifacts are not retrospectively presented as fully reproducible; the original station reconstruction discrepancy is still unresolved.

## Validation

133 Python tests pass. Frontend checks also pass, including old-report compatibility, waiting-for-settlement handling and resolved score rendering. Existing model-learning and serialization tests continue to pass. This pass did not perform a new live-market soak, full model retraining or browser interaction test.

## What remains the better next research step

The next model experiment should use a coherent distribution of the remaining-day maximum conditioned on the observed maximum, with a separately recorded station-to-official-settlement measurement model. Do not impose a hard support constraint from a possibly revised or differently rounded observation without checking that measurement chain. This is a stronger formulation than blindly clipping independent bracket probabilities.

Before judging that model, preserve exact source vintages and decision timestamps, predeclare complete-event evaluation and price/horizon slices, and use a fresh evaluation period. More candles alone do not add independent weather outcomes. Continuous live horizons, changing market rules, source revisions, first-release availability and cross-event dependence remain open issues. These changes deliberately do not loosen the promotion gate to manufacture trades.
