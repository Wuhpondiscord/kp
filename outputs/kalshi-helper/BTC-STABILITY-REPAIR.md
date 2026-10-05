# Why the ensemble regressed, and a controlled repair

## Outcome

Two ensemble design weaknesses were identified and corrected in a separate research implementation. Together they improve its aggregate Brier score from .121010 to .120879 and log loss from .378049 to .377737. The single fixed-anchor model remains better (.120761 / .377420). No corrected ensemble earns promotion or demonstrates profitability. Keep the simpler candidate unchanged; retain this diagnosis and the corrected implementation for reference.

## What the audit found

**Unequal day inclusion.** The original nonwrapping two-day sampler draws interior days through two possible block starts, but the first and last training day through only one. For the even-length windows here, endpoints have half the expected inclusion count of interior days. Across the 32 saved resamples, the first/last day appeared 21/18 times versus 36.17 on average for interior days in the first fold; in the last fold, 22/17 versus 33.04. This is a property of that sampling design, not evidence of a future-data leak. It changes training emphasis and systematically underrepresents the most recent day as well as the earliest day.

**Different neutral feature states.** Each member previously centered features using its own resample. Although it had no explicit intercept, each used a different feature origin. At the full training fold's mean feature vector, members deviated from the market by an average absolute 0.16–0.24 percentage points across folds. That is small, but it means the ensemble did not share one fixed definition of neutral external information. It also makes the effective regularization in raw feature units differ through member-specific scales.

**Little independent information per member.** Median unique training days were 5/8, 9/14, 13/20 and 17/26 across the four windows. Repeated days preserve nominal sample count, not independent information. Averaging these fits cannot create a missing predictive signal. This is an inherent limitation of the experiment, not something the two repairs eliminate.

## Changes and controls

Reuse the exact original saved resamples, all 32 members per fold, rather than drawing new favorable samples. Keep features, optimizer, L2 .1, coefficient bounds ±.25, chronological splits, market offset, costs and risk unchanged. Compare:

1. Original ensemble, loaded from saved artifacts.
2. Shared scaler: use the full training fold's mean/std for every member. Evaluation data never enter the scaler.
3. Boundary weights: weight each sampled row by the inverse expected inclusion count of its day under the original sampler; normalize the member's weighted likelihood. This compensates for unequal inclusion in the loss. Finite resampling and per-member normalization mean it does not guarantee an exactly unbiased fitted estimator. Member-specific scaling is intentionally left unchanged in this arm to isolate the loss-weight change.
4. Both repairs together.

These choices and input hashes were frozen before scoring the variants. This is a small mechanism ablation on previously inspected development dates, not an independent test. No cost scenario, parameter setting or model is automatically selected.

## Matched forecast results

527 contracts across 23 evaluation days; lower scores are better.

| Predictor | Brier | Log loss |
|---|---:|---:|
| Raw market | .121078 | .377750 |
| Single fixed anchor | **.120761** | **.377420** |
| Original ensemble | .121010 | .378049 |
| Shared scaler only | .120935 | .377876 |
| Boundary-weight correction only | .120950 | .377903 |
| Both repairs | .120879 | .377737 |

Both changes help the aggregate forecast scores in this comparison. Together they recover approximately 52% of the Brier gap between the original ensemble and the single model. Shared-scaler variants return the market at the common neutral feature vector to floating-point precision. Neither finding proves a trading edge.

Both-repair Brier minus market is −.000199, with a day-cluster bootstrap 95% interval [−.001574, +.001170]. Versus the single model it is +.000118, interval [−.000183, +.000436]. Both intervals cross zero. Repeated experimentation and remaining time dependence limit these descriptive intervals.

## Trading remains weak

$1,000 initial bankroll, unchanged 1% sizing/shared caps, 4-cent entry rule, fees and hypothetical quotes. Entries are in parentheses. Costs are scenarios, not settings chosen for their outcomes.

| Slippage | Original ensemble | Shared scaler | Boundary weights | Both repairs |
|---|---:|---:|---:|---:|
| 0¢ | −$36.33 (13) | −$21.81 (10) | −$12.17 (9) | −$12.17 (9) |
| 0.5¢ | −$31.25 (9) | −$39.08 (8) | −$30.45 (7) | −$38.53 (6) |
| 1¢ | −$12.82 (7) | −$21.09 (6) | −$21.50 (6) | −$29.77 (5) |
| 2¢ primary | −$0.78 (2) | −$10.64 (3) | $0 (0) | $0 (0) |
| 5¢ | $0 (0) | $0 (0) | $0 (0) | $0 (0) |

The repaired model's disagreement guard also abstains in every scenario. Zero trades are not profitability. Better aggregate scores still fail to produce reliable ranking among the few eligible bets; sparse outcomes cannot identify the exact size or cause of that failure. Improved weighting and centering fix part of the regression, not the underlying evidence shortage.

## Retained changes and limits

Retain explicit common training scaling, inclusion-aware weighted loss, diagnostics and regression tests in the experimental repair module. Preserve the previous artifacts and the single model. Do not make the ensemble the app default or loosen entry conditions to manufacture activity.

The results support a narrower explanation than “ensembles do not work”: this small block ensemble introduced uneven training emphasis and a moving feature origin, while each member had fewer unique training days. Correcting the first two improved forecasts, but the available external signal remained too fragile to establish net returns. Future work needs independently useful information or new evaluation evidence; further tuning of these same dates would not establish an edge.

The same historical publication-time, depth, fee and reused-date limitations remain. No holdout file was evaluated and no code was deployed.

## Review and reproduction

From `outputs/kalshi-helper`:

```powershell
python -m helper.btc_stability_repair
python -m unittest discover -s tests
```

- `helper/btc_stability_repair.py`: expected inclusion counts, weighted fitting, shared scaling, matched resample ablation and audit report.
- `tests/test_btc_stability_repair.py`: even/odd sampler inclusion accounting, uniform-weight equivalence, weighted-loss equivalence to duplicated rows with fixed scaling, common neutral origin, and rejection of out-of-training resamples.
- `reports/btc-stability-repair-v1/protocol.json`: fixed comparison and source hashes.
- `reports/btc-stability-repair-v1/models-*.json`: all repaired member coefficients and scalers per fold.
- `reports/btc-stability-repair-v1/report.json`: day inclusion counts, unique-day coverage, neutral-input shifts, forecast scores, paired uncertainty, ledgers and cost scenarios.

The runner checks that the original saved ensemble reproduces its prior headline scores before saving the comparison. Reruns are marked explicitly.
