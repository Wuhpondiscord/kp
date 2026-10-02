# Third model: fixed blend of the two combined ensembles

The third model has the best validation probability scores, but is not the best trading model under delayed execution. None of the three demonstrates robust profitability.

## Construction

Model 3 averages Model 1 (original) and Model 2 (regularized revision) with fixed 50/50 weights. Each parent averages three training seeds, so the new artifact contains six equally weighted pretrained networks. This is a new ensemble predictor, not a newly trained network. No additional labels, validation-fitted mixing weights, city routing, or profitable-trade selection are used. The design was saved before evaluation.

This deliberately places corrections between the more aggressive original and the more cautious revision. Averaging can improve probability scores when the parents make different errors; it cannot guarantee profitable threshold-selected bets or profitable execution.

## Forecast comparison

Lower scores are better. All models use exactly the same 730 validation rows, 257 city-days, and 67 dates. The parent training set contains 87 events across 22 dates.

| Model | Log loss | Brier |
|---|---:|---:|
| Market reference | 0.495074 | 0.164566 |
| 1: Original | 0.491363 | 0.162410 |
| 2: Revised | 0.491068 | 0.162524 |
| 3: Fixed blend | **0.489681** | **0.161883** |

Model 3 improves observed log loss by about 1.09% and Brier by 1.63% relative to market prices. Its descriptive adjusted block-bootstrap interval for log-loss difference versus market is [-0.01644, +0.00512], including zero. These repeatedly inspected validation dates cannot establish independent generalization.

## Profitability comparison

Start with $1,000. Hold to settlement; require 4 cents estimated net edge before entry; freeze the side and recheck edge at execution. Limit each bet to 1% of available cost-valued capital and at most $10, each city/day to $30, and total open cost to $200. Maximum order size is 100 contracts. Same fee assumption as the previous report: coefficient 0.07 and conservative cent-rounded debit, with historical overrides unverified.

| Execution assumption | Model 1 P/L | Model 2 P/L | Model 3 P/L |
|---|---:|---:|---:|
| Same quote, no slippage | +$147.11 | +$57.54 | +$98.23 |
| Next hourly quote, no slippage | +$6.32 | -$19.79 | -$64.38 |
| Next hourly quote + 2 cents | -$68.70 | -$10.00 | -$91.49 |

These are illustrative historical scenarios, not actual fills. Same-quote fills are optimistic; quotes have no historical depth. The delayed scenarios retain the decision-time probability, so they stress stale signals rather than simulate fully refreshed weather inference. Slippage changes the qualifying portfolio because edge is rechecked. Repeated validation use applies to profits as well as forecasting scores.

For Model 3, immediate execution produces 95 entries across 76 city-days, totaling 2,130 contracts, $880.77 deployed, and $34.10 fees. Net profit per contract is +4.61 cents; return on total deployed is +11.15%. Delayed execution plus 2 cents produces 42 entries across 36 city-days, 1,166 contracts, $365.49 deployed, and $16.06 fees. Net per contract is -7.85 cents; return on deployed is -25.03%.

Model 3's immediate net-per-contract interval is approximately [-20.12, +27.10] cents; the delayed +2-cent interval is [-25.96, +27.02] cents. These conservative descriptive intervals resample date blocks from the ledger, not the entire path-dependent strategy. The adjustment across 48 scenarios uses extreme quantiles of 2,000 replicates, so endpoints are Monte Carlo unstable. Neither interval establishes a positive edge. Model 2's smaller dollar losses reflect far fewer bets, not proof of a superior trading policy.

## Verification and artifacts

Both parent prediction vectors reproduce their saved values to 1e-12. The third artifact survives serialization and reproduces their arithmetic mixture to 1e-12. Parent model hashes are verified. All scenario cash/ledger reconciliations, settlement completion, and shared risk-cap checks pass.

`helper/fusion_blend.py` provides the reusable predictor. `reports/three-fusion-comparison/` contains the fixed protocol, `blend.pkl`, artifact/member hashes, full forecast metrics and city/month/price breakdowns, all 48 trading scenarios with ledgers and uncertainty, and per-row predictions. Only load trusted pickle artifacts.

From the workspace root, with the existing PyTorch environment and offline forecast cache:

```powershell
python work/compare_three_fusion.py --output outputs/kalshi-helper/reports/three-fusion-reproduction
```

Use a fresh output directory. No reserved test was loaded and no active app model was replaced. Keep Model 3 as the leading forecast candidate on reused validation, while retaining the parents for comparison. Further profit claims require contemporaneous depth and refreshed inference on new dates; this experiment does not justify higher exposure.
