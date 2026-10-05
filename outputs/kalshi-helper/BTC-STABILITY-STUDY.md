# Training-day stability study

## Decision

Do not replace the single fixed-anchor candidate with the ensemble. Averaging across resampled training days did not improve aggregate forecast scores or trading results. Retain the new sensitivity measurement and disagreement guard as research diagnostics. Neither the candidate nor its guard earned deployment, and the July 21–31 holdout remains sealed.

## Implemented experiment

The previous execution study showed that small cost changes selected very different, poorly performing trade subsets. This study tests whether those apparent edges are sensitive to which days train the model.

For each of the same four chronological splits, fit 32 fixed-anchor models to moving blocks of two consecutive training days, sampled with replacement. Keep the number of sampled days equal to the original training window; truncate the final sampled block if necessary. Each member fits its own training-only scaler. Use seed 1729, unchanged four features, L2 penalty .1 and coefficient bounds ±.25. Average member probabilities; do not tune ensemble weights, block length or member count against outcomes.

The new disagreement guard uses the fifth percentile of member YES probabilities for a YES signal and the 95th percentile for a NO signal, clamped toward the market so it cannot reverse the ensemble's side. This conservative input is used only for trade qualification; it is not reported as the model's forecast. Across-fit dispersion is a sensitivity measure, not a calibrated confidence interval or a bound on true probabilities. All members can share the same bias or missing information.

Models are saved before scoring each evaluation window. The study uses the same 527 previously inspected development contracts, embargoes, outcomes, market quotes, fees, sizing, 4-cent entry threshold and primary 2-cent slippage. The 60-second input-age scenario is retained. No network collection, hyperparameter search, holdout evaluation or production change occurs.

## Forecast results

| Model | Brier | Log loss |
|---|---:|---:|
| Raw market | .121078 | .377750 |
| Single fixed anchor | **.120761** | **.377420** |
| Training-day ensemble | .121010 | .378049 |
| Ensemble with 60-second delayed inputs | .122389 | .381405 |

The ensemble is worse than the single model on both aggregate metrics. Compared with the raw market, it has a tiny Brier improvement but worse log loss. Its Brier difference versus market is −.000068 with a day-cluster bootstrap 95% interval [−.001592, +.001458]. Versus the single fixed anchor, the difference is +.000249 with interval [−.000133, +.000711]. Both intervals cross zero and are descriptive, not adjusted for repeated model experimentation.

| Evaluation starts | Single-anchor Brier | Ensemble Brier | Median fifth-to-95th percentile member span |
|---|---:|---:|---:|
| June 28 | .126594 | .127723 | 3.81 percentage points |
| July 4 | .138557 | .138643 | 3.77 percentage points |
| July 10 | .108658 | .108648 | 4.13 percentage points |
| July 16 | .107540 | .107254 | 2.75 percentage points |

The recent-period improvement again fails to generalize to the earliest period. Wider training-sample sensitivity is relevant to whether a small price difference is robust, but the spans do not by themselves quantify uncertainty about realized returns.

## Trading results

$1,000 initial bankroll, existing 1% risk/shared caps, unchanged fees and entry threshold. Parentheses show entries. Each row is a different assumed execution scenario, not a deployable setting selected for profitability.

| Additional slippage | Single fixed anchor | Ensemble | Ensemble + disagreement guard |
|---|---:|---:|---:|
| 0¢ | +$3.14 (7) | −$36.33 (13) | $0 (0) |
| 0.5¢ | −$29.20 (3) | −$31.25 (9) | $0 (0) |
| 1¢ | −$9.82 (1) | −$12.82 (7) | $0 (0) |
| 2¢ primary | $0 (0) | −$0.78 (2) | $0 (0) |
| 5¢ | $0 (0) | $0 (0) | $0 (0) |

The guard rejects every candidate trade. It prevents the ensemble's losses in this replay by abstaining completely; it does not demonstrate better profitable selection. Sparse-trade confidence intervals remain suppressed. Zero-slippage scenarios still include bid/ask spread and fees and do not establish achievable historical fills.

## What this changes

Keep the single fixed-anchor model as the better of these research candidates on current aggregate scores. Do not deploy the ensemble merely because it is more complex or because it improves the last evaluation period. The new training-day sensitivity diagnostic gives a concrete reason to distrust the small estimated edges: their apparent value changes substantially across plausible training resamples.

The remaining problem is finding independently informative, timely inputs with evidence that survives costs and new evaluation dates. Repeatedly modifying these models on the same development outcomes cannot provide that evidence. This study adds a reproducible rejection and an optional diagnostic guard; it does not alter the earlier no-proven-edge conclusion.

## Files and reproduction

From `outputs/kalshi-helper`:

```powershell
python -m helper.btc_stability_model
python -m unittest discover -s tests
```

- `helper/btc_stability_model.py`: contiguous training-day resampling, deterministic ensemble fitting, saved-model loading, guarded decisions and evaluation.
- `tests/test_btc_stability_model.py`: reproducible block selection and calendar adjacency, label-independent sampling, guard direction/strength constraints, artifact round trip and inference immutability.
- `reports/btc-stability-v1/protocol.json`: fixed choices and hashes of development data and prior model/age artifacts.
- `reports/btc-stability-v1/models-*.json`: 32 member artifacts per fold and their sampled training days.
- `reports/btc-stability-v1/report.json`: scores, per-period comparisons, member spans for each prediction, cost scenarios, ledgers, uncertainty status and code hashes.

The prior single model's headline scores are checked automatically before the report is saved. Reruns are marked. The original model artifacts remain unchanged. Historical arrival-time, depth, fee and reused-date limitations from the preceding studies still apply.
