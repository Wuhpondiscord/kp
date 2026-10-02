# Event-fusion update: measured benefits and limits

A single fixed comparison was run at the user’s request. The new categorical architecture fixes probability mass, but it did not improve the forecast or trading performance of the matched binary control. Normalization alone improved two proper scores while worsening simulated trading returns. No model was promoted or installed.

## Fair comparison and sample

Both architectures trained from scratch on the same 486 bracket rows: 81 complete city/day events across 22 dates. Each used seeds 1729/1730/1731, 200 epochs, weather auxiliary coefficient 1, residual penalty 0, AdamW learning rate 0.01 and weight decay 0.01. Every seed was averaged; there was no checkpoint or hyperparameter selection. The binary control corresponds to the WeatherSignal recipe, not MarketGuard’s separately regularized recipe.

Validation contains 1,536 bracket rows from 256 complete events across 67 already-inspected dates. Forecasts were computed on full ladders first, then the unchanged 5–95 cent price filter selected 713 rows for contract metrics and trading. Seven training events were excluded (six incomplete ladders and one stale observation event); twelve validation events had incomplete ladders. Exclusion details and exact cohort files/hashes are saved. No missing quotes, weather observations or brackets were fabricated.

This changes the cohort from the previous 514-row/87-event training and 730-row validation comparison. In particular, the positive binary-control P/L below cannot be interpreted as an improvement over the previous retained WeatherSignal P/L: training membership and evaluation membership changed. Complete-case selection can itself affect results.

## Contract probability scores on the 713-row price subset

Lower is better. ECE uses fixed bins, is noisy, and is not interchangeable with a proper scoring rule.

| Model | Log loss | Brier | ECE | Mean complete-event mass error |
|---|---:|---:|---:|---:|
| Market price | 0.494088 | 0.164040 | 0.023606 | 3.381 percentage points |
| Normalized market | 0.494274 | 0.164011 | 0.021672 | 0.000 percentage points |
| Binary fusion, matched training | 0.492303 | 0.162569 | 0.029597 | 7.149 percentage points |
| Binary fusion + event normalization | 0.488496 | 0.160684 | 0.041175 | 0.000 percentage points |
| New categorical softmax | 0.496347 | 0.163279 | 0.021853 | 0.000 percentage points |

The binary control’s mean absolute event-mass error is 7.15 percentage points, with a maximum of 25.17 points. Softmax and normalization remove that error to floating-point precision. The softmax log-loss difference versus binary is +0.00404 (worse), with a descriptive adjusted seven-date-block interval [−0.00946, +0.01799]. Normalization’s difference is −0.00381, interval [−0.00673, −0.00079]. That interval excludes zero within this comparison but does not account for all prior project-wide selection on these dates. Normalization’s ECE is worse despite better log loss and Brier; it is not an across-the-board calibration improvement.

## Complete-event categorical scores

Raw independent binary outputs are not categorical distributions, so they are not assigned categorical NLL here. Both scores below are averaged equally by date and then by event; categorical Brier sums across brackets.

| Coherent distribution | Categorical NLL | Categorical Brier |
|---|---:|---:|
| Normalized market | 0.858046 | 0.482478 |
| Binary fusion + event normalization | 0.860228 | 0.474720 |
| New categorical softmax | 0.869647 | 0.480783 |

## Paper trading: $1,000 starting bankroll

Unchanged 4-cent minimum net edge, 1%/$10 per-bet cap, 3%/$30 shared city/day cap, 20%/$200 portfolio cap and 100-contract limit. Standard assumed fees are included. Exposure caps apply to concurrent capital; cumulative deployed capital may exceed the initial bankroll as capital is recycled.

| Model | Same quote (optimistic) | Next hourly quote | Next hourly quote +2¢ |
|---|---:|---:|---:|
| Market price | $+0.00 | $+0.00 | $+0.00 |
| Normalized market | $+0.00 | $+0.00 | $+0.00 |
| Binary fusion, matched training | $+185.08 | $+276.43 | $+94.69 |
| Binary fusion + event normalization | $+107.72 | $-19.47 | $-141.01 |
| New categorical softmax | $-67.99 | $+25.26 | $-104.50 |

Delayed scenarios retain stale decision-time probabilities and recheck edge against the later quote. They are execution sensitivity tests, not live fills or execution-time reforecasts. Historical depth and series-specific historical fee overrides remain unverified.

### Delayed +2-cent scenario detail

| Model | Entries / city-days | Net per contract | Return on deployed capital | Adjusted net/contract interval |
|---|---:|---:|---:|---|
| Binary fusion, matched training | 125 / 106 | +2.89¢ | +8.18% | [-12.75¢, +18.94¢] |
| Binary fusion + event normalization | 83 / 66 | -5.65¢ | -19.24% | [-15.21¢, +2.40¢] |
| New categorical softmax | 172 / 104 | -2.25¢ | -7.03% | [-12.63¢, +7.92¢] |

All trading intervals include losses and gains. The binary control’s +$94.69 point estimate is not demonstrated edge. These intervals are exploratory seven-date cluster bootstraps with adjustment across the 15 reported model/scenario combinations, not a correction for all historical project searches. Contract quantities do not create new independent weather outcomes.

## Interpretation and decision

Coherence is a real correctness benefit. It does not imply more accurate probabilities. The categorical model changes the output reference from binary log odds to normalized market log probabilities, changes the loss from per-contract BCE to categorical NLL, and changes the relative scale of the auxiliary weather objective even though its numerical coefficient remains 1. This experiment tests that fixed implementation package; it cannot isolate every mechanism or establish that all categorical architectures are inferior.

Normalization can improve aggregate proper scores while changing which contracts cross a hard trading threshold, their chosen sides and their sizes. The observed score/trading divergence persists in the 5–95 cent subset and therefore cannot be explained solely by the older below-2-cent tail finding.

Retain the categorical implementation and tests as a research capability, but do not replace the saved/live families on this evidence. The saved baseline weights were not modified. The next defensible performance claim requires fresh data, broader seasonal coverage and a declared complete-event evaluation, rather than selecting another historical weight or threshold.

## Artifacts and verification

- `reports/event-fusion-benefits/protocol.json`: fixed design and input/source hashes, written before fitting.
- `report.json`: all metrics, seeds/training traces, cohort exclusions, ledgers, risk/accounting checks and uncertainty.
- `train-cohort.jsonl`, `validation-cohort.jsonl`, `predictions.jsonl`: frozen rows and predictions.
- Six versioned experimental checkpoint files: three per architecture; kept separate from named-model artifacts.
- `work/test_event_fusion_benefits.py`: offline reproduction script (requires a fresh output folder).

All 230 Python tests passed. All six experimental checkpoints reproduce their predictions after serialization. Scenario settlement P/L matches the ledger totals; all configured risk limits were checked. No reserved test rows were read. The latest user explicitly requested this additional historical comparison; it adds another exploratory look and is not a prospective validation.
