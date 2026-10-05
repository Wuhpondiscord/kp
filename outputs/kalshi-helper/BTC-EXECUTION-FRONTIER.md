# BTC execution feasibility: cost sensitivity and selection instability

The next priority was to distinguish weak signals from expensive execution. This experiment loads the saved models without fitting, holds the entry rule and risk limits fixed, and evaluates explicit slippage scenarios. It also reprices a fixed cohort to separate arithmetic cost increases from changing trade selection.

**Decision:** keep both models experimental. Neither cheaper assumed fills nor the fixed-anchor accuracy improvement establishes a robust trading edge. Do not select the most profitable slippage setting, lower the edge threshold, increase risk, or open the reserved holdout.

## Matched results

527 previously inspected development contracts across 23 days; frozen four-window models, $1,000 starting bankroll, existing 1% sizing/shared caps, 4-cent net-edge requirement, .07 fee coefficient and conservative cent rounding. Spreads and fees apply even in the zero-slippage scenario. Additional slippage is charged per contract at entry. Quotes do not establish available depth or executable fills.

| Additional slippage | Fixed anchor P&L | Trades | Full flow P&L | Trades |
|---|---:|---:|---:|---:|
| 0¢ | +$3.14 | 7 | +$25.30 | 19 |
| 0.5¢ | −$29.20 | 3 | +$10.15 | 10 |
| 1¢ | −$9.82 | 1 | +$11.46 | 6 |
| 2¢ (unchanged primary) | $0.00 | 0 | −$0.46 | 2 |
| 5¢ | $0.00 | 0 | $0.00 | 0 |

These are scenario replays with changing eligibility and quantities, not a monotonic cost curve. Zero P&L with no trades means abstention. Every scenario has fewer than 30 entries; sparse-trade profit confidence intervals remain suppressed. Contract quantities are not independent outcomes.

At zero assumed slippage, fixed anchor deploys $68.86 in 183 contracts: net profit is 1.72 cents per contract, or 4.56% of deployed money, across seven trades on five days. Full flow deploys $181.70 in 256 contracts: 9.88 cents per contract, or 13.92%, across 19 trades on ten days. These optimistic scenario returns are not evidence of achievable execution or dependable returns.

## What the signals can tolerate

For each signal qualifying before additional slippage, the new diagnostic calculates the maximum extra buy price that preserves the existing conservative one-contract qualification. It uses Decimal fee calculations and a 0.0001-dollar grid; it never uses the realized outcome to set a limit.

| Model | Signals before slippage | Median tolerance | Maximum tolerance |
|---|---:|---:|---:|
| Fixed anchor | 7 | 0.39¢ | 1.30¢ |
| Full flow | 19 | 0.54¢ | 2.29¢ |

These are model-implied entry-rule tolerances, not empirically established break-even costs. They describe the current simulator's conservative one-contract gate; final quantity-level fee rounding can differ. Sub-cent tolerances make accurate arrival, bid/ask and depth observations relevant if this model is ever used live. We do not possess that historical execution evidence.

## Costs versus selection

The fixed-cohort diagnostic starts with the zero-slippage fills and holds side and quantity fixed while repricing them. This is accounting only: it intentionally does not enforce the new cost scenario's entry or sizing limits and is not a runnable strategy.

| Slippage | Fixed-anchor same-seven-trade P&L | Full-flow same-19-trade P&L |
|---|---:|---:|
| 0¢ | +$3.14 | +$25.30 |
| 0.5¢ | +$2.22 | +$24.07 |
| 1¢ | +$1.28 | +$22.82 |
| 2¢ | −$0.58 | +$20.33 |
| 5¢ | −$6.15 | +$12.89 |

For fixed anchor, adding 0.5¢ to the same fills costs $0.92 including the fee change. The actual policy result drops by $32.34 because the remaining three qualifying trades all lose; both cohort selection and quantity can change. Therefore, attributing the entire loss to transaction costs would be wrong. The model's stronger estimated edges did not produce a stronger realized subset here, but three outcomes are far too few to establish systematic inverse ranking.

Full flow has a related concentration problem: at zero slippage its last evaluation period contributes +$27.54, while the other periods contribute +$2.20, $0 and −$4.44. A positive aggregate result is not broad period consistency.

## Implications

1. Retain the fixed-anchor model as a forecast research candidate, not a promoted trading model. Its small aggregate accuracy improvement does not imply reliable ranking among the few trade candidates.
2. Retain realistic-cost gating. Picking zero slippage after seeing these results would replace an assumption with a favorable assumption, not fix the model.
3. Execution telemetry would answer whether the narrow tolerances are achievable, but execution alone cannot establish that the probabilities are calibrated on selected bets. Both remain open questions.
4. The next substantive model/data experiment should add independent information or genuinely new evaluation evidence. Another search across these same costs, thresholds or blend weights would not provide trustworthy confirmation.

The earlier input-age study also showed that a one-minute feature delay erased the fixed-anchor forecast advantage. Together, these results argue for verifiable timing/execution data before a live trading claim, rather than more model complexity or exposure.

## Implementation and reproduction

```powershell
# Run from outputs/kalshi-helper; no network or training needed.
python -m helper.btc_execution_frontier
python -m unittest discover -s tests
```

- `helper/btc_execution_frontier.py`: frozen-model replay, slippage tolerance, fixed-cohort cost analysis, and baseline reproduction checks.
- `tests/test_btc_execution_frontier.py`: exact last-passing slippage tick, invalid input handling, sensitivity direction, and fixed-cohort fee/P&L monotonicity and reconciliation.
- `reports/btc-execution-frontier-v1/protocol.json`: pre-evaluation choices and hashes of source data/reports/model artifacts. Paths record this local run; use a new output directory when reproducing in another checkout.
- `reports/btc-execution-frontier-v1/report.json`: complete per-period/scenario results, ledgers, uncertainty status, all signal tolerances, code hashes and fixed-cohort diagnostics.

The runner verifies that frozen predictions reproduce the earlier Brier/log-loss scores and exact primary trade ledgers. No retraining, network requests, holdout evaluation, production strategy changes or deployment were performed. Historical fee/depth/publication assumptions and reused-development limitations still apply.
