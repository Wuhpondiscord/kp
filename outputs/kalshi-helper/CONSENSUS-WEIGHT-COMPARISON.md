# ConsensusBlend weight sensitivity

## Requested 60/40 follow-up

After the original grid was inspected, the user requested **60% WeatherSignal / 40% MarketGuard**. This additional exploratory evaluation gives log loss **0.489759** and Brier **0.161895**. Both are slightly worse than 50/50 and better than 70/30. The adjusted descriptive log-loss difference interval versus 50/50 is [-0.001026, +0.001244], including zero.

With the unchanged $1,000 bankroll policy, 60/40 makes **+$157.99** at the immediate quote (132 entries), **+$54.28** at the next hourly quote (68 entries), and **-$21.14** at the next quote plus 2-cent slippage (62 entries). It improves the delayed +2-cent result versus 50/50 (-$91.49), but underperforms 70/30 (-$1.87). It does not establish profitability.

Artifacts are in `reports/consensus-sixty-forty/`. The original seven weights were rerun as controls; their scores and scenario P/L reproduce. All scenario accounting and risk-cap checks passed. Uncertainty adjustments cover eight weights / 48 trading scenarios for this follow-up; this does not remove the risk from requesting another weight after seeing earlier results. The retained blend stays 50/50, with no new model artifacts or reserved-test access.

Reproduce from the workspace root with a fresh output directory:

```powershell
python work/compare_blend_weights.py --include-sixty --output outputs/kalshi-helper/reports/consensus-sixty-forty-reproduction
```

## Original seven-weight experiment

Seven predetermined WeatherSignal/MarketGuard weights were evaluated on the same 730 previously inspected validation rows (257 events, 67 dates). No networks were retrained. Both parents are the retained named-family versions. The reserved test was not read, and no model artifact or active app selection was changed.

The user's example, 70% WeatherSignal plus 30% current ConsensusBlend, equals **85% WeatherSignal plus 15% MarketGuard**, because current ConsensusBlend is 50/50. Both that interpretation and direct 70/30 parent weighting are included.

## All results

Lower forecast scores are better. P/L starts with $1,000 and uses unchanged standard fee assumptions, 4-cent minimum estimated net edge, 1%/$10 maximum per bet, $30 city/day cap, $200 open-cost cap, and 100-contract order limit. Hold to settlement; side is frozen at decision and edge is rechecked at execution.

| WeatherSignal / MarketGuard | Log loss | Brier | Immediate P/L | Next-hour P/L | Next-hour +2c P/L | +2c entries |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 100 | 0.491068 | 0.162524 | +$57.54 | -$19.79 | -$10.00 | 1 |
| 15 / 85 | 0.490351 | 0.162209 | +$87.51 | +$21.40 | +$19.53 | 4 |
| 30 / 70 | 0.489889 | 0.161999 | +$18.70 | -$77.14 | -$80.50 | 19 |
| **50 / 50** | **0.489681** | **0.161883** | +$98.23 | -$64.38 | -$91.49 | 42 |
| 70 / 30 | 0.489961 | 0.161953 | +$203.26 | +$104.22 | -$1.87 | 75 |
| 85 / 15 | 0.490509 | 0.162129 | +$196.31 | +$27.55 | -$18.38 | 99 |
| 100 / 0 | 0.491363 | 0.162410 | +$147.11 | +$6.32 | -$68.70 | 119 |

The 50/50 blend remains best on both forecast metrics among these tested weights. Direct 70/30 parent weighting improves the observed trading scenarios versus 50/50 but is approximately break-even after the delayed +2-cent assumption. The requested nested 70/30 blend (85/15 parents) also improves delayed scenario P/L versus 50/50, but remains negative with slippage. No finer grid was searched after these results.

## Opportunity and uncertainty checks

The 15/85 blend's positive delayed +2-cent P/L comes from just four entries across four city-days: 142 contracts, $39.47 deployed, +13.75 cents per contract, and +49.48% return on deployed cash. Its descriptive net-per-contract interval is roughly [-29.41, +69.53] cents. This sparse sample cannot establish an edge; 101 bootstrap samples contain no trades.

The 70/30 blend has 75 entries across 65 city-days: 2,007 contracts, $658.87 deployed, -0.093 cents net per contract, and -0.284% return on deployed cash. Its interval is roughly [-20.46, +26.83] cents per contract. The 85/15 blend has 99 entries across 85 city-days, -0.666 cents per contract, and -2.11% return on deployed cash.

These seven-date ledger bootstrap intervals are descriptive, adjusted across 42 reported sizing/execution scenarios, and use 2,000 replicates. Their extreme-tail endpoints are Monte Carlo unstable. They do not replay path-dependent sizing on resampled data and do not correct for all earlier research on these dates. Forecast-loss difference intervals versus 50/50 also all include zero.

## Why the profit ranking is irregular

Blend probabilities change smoothly with weight, but the 4-cent entry rule changes the portfolio discretely. Different weights select different contracts, quantities, and execution survivors. Therefore P/L is not a smooth interpolation of parent profits. The previously recorded selection/cost diagnosis remains applicable. Observing a profitable weight on this reused sample is not evidence that its profit will persist.

Immediate quotes are optimistic hypothetical fills without depth. Next-hour scenarios retain stale decision-time predictions rather than refreshing all weather inputs. Historical per-series fee overrides are not verified. Slippage changes qualifying trades as well as costs. These scenarios are useful sensitivity checks, not verified live returns.

## Saved behavior and reproduction

`FusionBlend` now accepts an explicit `weather_weight` in [0,1], while existing artifacts without that attribute retain their 50/50 behavior. All three seeds within each parent keep equal weights. Tests cover endpoints, invalid weights, the nested example, and legacy artifacts.

The retained ConsensusBlend remains **50/50**. No extra model families or weight-specific pickle files were created. `reports/consensus-weight-comparison/` contains the fixed protocol, input hashes, all metrics/scenarios/ledgers and uncertainty, and per-row predictions. Parent model hashes and reproduction of retained 50/50 predictions are checked. Accounting, settlement completion, and risk caps are checked for every scenario.

From the workspace root:

```powershell
python work/compare_blend_weights.py --output outputs/kalshi-helper/reports/consensus-weight-reproduction
```

Use a fresh output directory and the existing environment and saved reports. A prospective comparison of fixed 50/50 and 70/30 policies with refreshed inference would be more informative than selecting a new default from this repeatedly inspected grid.
