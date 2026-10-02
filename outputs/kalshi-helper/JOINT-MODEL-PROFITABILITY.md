# Profitability check: original and revised combined ensembles

Neither model demonstrates robust profitability. The revised model's optimistic profit comes from only four city-days; delayed execution removes most opportunities. The older ensemble has more signals but loses under the delayed quote plus slippage scenario.

## Fixed setup

We evaluated the saved predictions for 730 contracts across 257 city-day events and 67 dates, with decisions from April 27 to July 5, 2026. These are previously inspected validation dates, not untouched test data. No model was retrained and no threshold was optimized for these returns.

Starting cash is $1,000. The fixed decision-time minimum net edge is 4 cents, with the side frozen at decision and edge rechecked at execution. Contracts are held to settlement. Sizing is capped at 1% per bet ($10 maximum), 3% shared across brackets for the same city/day ($30), and 20% open portfolio cost ($200), with at most 100 contracts per order. The bankroll sizing policy also reduces the bet budget when capital falls. A separate one-contract policy measures results without scaling quantities. There are no loss stops in these scenarios.

Fees assume the standard 0.07 coefficient, using the existing conservative cent-rounded debit calculation. This follows the standard structure in the [Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf); historical series-specific overrides were not verified. Hourly quote closes provide no evidence that the assumed quantities were available. Results are hypothetical fill scenarios, not verified executable returns.

## $1,000 bankroll results

| Execution assumption | Original net P/L | Revised net P/L | Original / revised entries |
|---|---:|---:|---:|
| Same quote, no added slippage | +$147.11 | +$57.54 | 234 / 6 |
| Same quote, 2-cent slippage | +$179.93 | +$20.25 | 161 / 1 |
| Next hourly quote, no added slippage | +$6.32 | -$19.79 | 139 / 2 |
| Next hourly quote, 2-cent slippage | -$68.70 | -$10.00 | 119 / 1 |
| Next hourly quote, 5-cent slippage | -$170.82 | -$9.75 | 98 / 1 |

Same-quote execution is an optimistic counterfactual. Next-hour execution is a stale-signal stress test: the probability is fixed at decision time and weather inputs are not refreshed. It does not estimate the performance of a properly refreshed live weather/market model. The two scenarios cannot establish actual live returns.

Slippage can increase reported profit here because the minimum-edge recheck changes which bets qualify; the rows are not identical portfolios with different costs. The +$179.93 scenario must not be interpreted as evidence that worse execution helps.

## Edge, deployment, and opportunity counts

| Scenario and model | Contracts | City-days traded | Total deployed | Net per contract | Return on deployed |
|---|---:|---:|---:|---:|---:|
| Same quote, original | 4,896 | 157 | $2,193.89 | +3.00 cents | +6.71% |
| Same quote, revised | 185 | 4 | $58.46 | +31.10 cents | +98.43% |
| Delayed +2 cents, original | 3,214 | 101 | $1,029.70 | -2.14 cents | -6.67% |
| Delayed +2 cents, revised | 34 | 1 | $10.00 | -29.41 cents | -100% |

Deployment includes reused cash across the evaluation period, so it can exceed starting bankroll. Quantities do not create independent outcomes. The revised model's striking same-quote return reflects only six entries across four city-days. Its delayed +2-cent result is a single losing entry, not 34 independent losses.

With one contract per entry, original/revised same-quote profits are +$7.97 / +$2.04. Delayed +2-cent profits are -$3.15 / -$0.30. Scaling increases dollar swings; it does not establish an edge.

The revision has 145 positive estimated net-edge rows, but only six exceed the unchanged 4-cent threshold. The original has 530 positive rows, of which 234 qualify. The market-price control makes no bets because spread and fees consume its apparent edge. Regularizing toward market prices improved forecast stability while greatly reducing trading opportunities.

## Uncertainty and checks

Exploratory seven-date block-bootstrap intervals for same-quote net profit per contract are approximately [-11.68, +15.33] cents for the original and [-28.42, +69.03] cents for the revision. Both include losses. These use 2,000 replicates with a conservative adjustment across 30 reported scenarios; extreme-tail endpoints are Monte Carlo unstable and are descriptive only. They resample the realized ledger, not the entire path-dependent sizing strategy.

For the revised delayed scenario, all nonempty resamples repeat one losing city-day. Its degenerate interval is not meaningful evidence of certainty; 656 of 2,000 resamples contain no trades. The sample is too sparse to establish profitability.

All scenarios reconcile final cash against settled ledger P/L, leave zero unsettled positions, and meet bet/event/portfolio caps. The three existing sizing tests pass. Predictions are joined to their exact contract/time records; input hashes and the fixed protocol are saved. Reserved test data and active app models remain unchanged.

## Artifacts and interpretation

`reports/joint-profitability/protocol.json` records the assumptions before evaluation. `report.json` includes all 30 scenarios, complete ledgers, opportunity counts, fees, exposure, drawdown, and uncertainty. `work/test_joint_profitability.py` reproduces the offline analysis from the saved predictions and validation quotes; its output directory must not already exist.

The revised model is not established as the better trading model. Keep it and the original as forecast research candidates. The next useful profitability evidence requires contemporaneous book depth and a fully refreshed prediction at execution, evaluated prospectively under a frozen policy. Lowering thresholds or scaling risk based on these reused dates would not resolve the evidence gap.
