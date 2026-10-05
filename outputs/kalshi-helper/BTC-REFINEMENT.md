# BTC bottleneck follow-up: calibration, timing and market information

The useful improvement is to anchor predictions to market odds and demand evidence
before taking the external BTC signal seriously. Standalone sigmoid calibration
does not generalize across the inspected weeks. We retained a market-anchored
research implementation and strengthened execution-input validation; none of the
new candidates is promoted to live trading or substituted into the hosted lab.

## Controlled comparison

Base architectures and training remain fixed: the logistic signal and shallow trees
train on January–April 2025 Binance data. No September labels fit those base models.
Small correction models fit earlier settled September Kalshi outcomes, then predict
later dates with a two-hour embargo. Three expanding development folds cover
September 7–18; the later validation window covers September 19–24 (559 contracts).
All these dates were already inspected in earlier work. This run does not open the
September 25–30 test file; that does not make the reused validation dates untouched.

The protocol was saved before fitting. There is no parameter grid, threshold search,
cost reduction or increase in risk. All scenarios start with $1,000, keep the 4-cent
edge requirement, 2-cent assumed slippage, 0.07 assumed fee coefficient, 1% per-bet
budget and existing shared-day/portfolio caps. All reported forecast scores use
execution-time inputs and the matching market price. Historical depth is unavailable.

| Later validation, same 559 contracts | Log loss ↓ | Brier ↓ | Net simulated P&L | Entries |
|---|---:|---:|---:|---:|
| Market odds / no-trade control | 0.374852 | 0.119655 | $0.00 | 0 |
| Original volatility | 0.386163 | 0.124433 | +$3.72 | 53 |
| Original external logistic signal | 0.386510 | 0.124290 | −$133.47 | 53 |
| Original external regime trees | 0.391974 | 0.125322 | −$98.27 | 53 |
| Execution-aligned signal calibration | 0.390662 | 0.125258 | −$147.35 | 56 |
| Market-only calibration control | **0.374453** | **0.119566** | $0.00 | 0 |
| Market-anchored BTC signal | 0.374657 | 0.119688 | $0.00 | 0 |
| Signal calibration fitted on older decision inputs | 0.390521 | 0.125237 | −$147.35 | 56 |

These six-day values must not be compared directly with the previously published
30-day dollar totals. The existing $1,000 full-month lab still reproduces −$122.52,
−$319.83 and −$222.99; its predictors and results are unchanged.

## What the experiments isolated

**1. Standalone recalibration is unstable over time.** On the earlier pooled forward
folds, calibration improves logistic log loss from 0.426245 to 0.422550 and simulated
P&L from +$8.72 to +$67.72. It then worsens both metrics on the later validation
window. The calibration's fitted positive log-odds intercept is not a dependable
correction for the later period. A positive aggregate development P&L would have
been an unreliable selection rule. The fold-level report preserves every result.

**2. Matching clocks is necessary for valid comparisons, but not the main cure.**
The older-input calibration ablation has nearly the same later losses as the
execution-aligned version. Fixing freshness alone does not explain away the poor
results. The remaining five-second candle-publication allowance is still assumed,
not a measured order-book receipt or fill latency.

**3. Most of the apparent forecasting improvement comes from using the market.**
The anchored model lowers logistic log loss by about 3.1% and Brier by about 3.7%,
but the market-only control is slightly better. The anchored coefficient on clipped
external-minus-market log odds is 0.0377 (this is not a 3.77% probability blend).
Its paired Brier difference versus market is +0.0000327, with exploratory day-block
95% interval [−0.0000457, +0.0001036]. The market-only control's difference is
−0.0000892, interval [−0.0001883, +0.0000131]. Both intervals cross zero.

**4. Avoiding bad bets is a useful trading behavior, not a discovered profit.**
Both market-based candidates make zero entries under the unchanged policy across
all three forward folds and validation. They preserve the bankroll instead of
manufacturing four-cent edges from miscalibrated external predictions. Return on
invested funds is undefined when nothing is invested; the report keeps it null.
The raw market/no-trade control does equally well in dollars without a fitted model.

**5. Input validation was weaker at execution than at the original decision.**
The legacy execution adapter accepted nested refreshed features without checking
their availability or freshness. The new shared validator checks decision < close,
execution before close/settlement, feature availability no later than execution,
maximum 65-second feature age, ten finite features, valid diffusion probability,
and unique contracts. Missing later quotes remain excluded rather than invented.
No evidence shows future inputs in the current verified archive; this is a defensive
correctness fix, not a claim that past P&L was caused by leakage.

## What changed in code

- `helper/btc_refinement.py`: bounded, regularized corrections, chronological fitting,
  explicit market-only ablation, matched forecast/trading evaluation, paired forecast
  intervals, trading uncertainty and a frozen protocol with hash-linked outputs.
- `helper/btc_inputs.py`: reusable validation of execution-time inputs.
- `helper/btc_lab.py`: the paper lab now uses that validator locally. No model
  selection or forecast weights changed; this update has not been deployed.
- Regression tests cover identity fallback, confidence reduction on deliberately
  overconfident predictions, probability bounds, split embargo/settlement cutoff,
  malformed labels, future/stale features, duplicate contracts and invalid chronology.

The experiment stores correction coefficients in `reports/btc-refinement-v1/model.json`.
These belong to the corresponding frozen external predictor and protocol; they are
not standalone probabilities or a general-purpose live checkpoint.

## What to pursue next

The evidence favors testing genuinely additional, time-stamped information against
market odds, rather than making standalone price classifiers more confident.
The most relevant unresolved data issues are the exact settlement-index target,
availability of its constituent/aggregate prices, executable quotes, and bid/ask
depth. More proxy-only candles cannot resolve those mismatches. A new source needs
untouched paired historical evaluation before another promotion decision.

Until then, retain the market benchmark and abstention option. Do not tune a lower
entry threshold to make the new anchored model trade, or deploy the calibrated
signal just because its earlier rolling profit was positive.

## Reproduce

```powershell
python -m helper.btc_refinement
python -m unittest discover -s tests
```

All **298 tests passed**. The experiment completed on real bundled data. Report:
[reports/btc-refinement-v1/report.json](reports/btc-refinement-v1/report.json).
Its intervals use only six validation days and do not account for prior experiment
selection or cross-day dependence; no proven forecast or trading edge is claimed.

Method references: scikit-learn's [calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)
requires separating classifier fitting from calibration data; its
[time-series cross-validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
describes chronological training windows and gaps. Here, explicit settlement-time
cutoffs and a two-hour embargo implement those separation principles.
