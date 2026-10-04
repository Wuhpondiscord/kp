# BTC diagnosis and next decision — October 4, 2026

The current priority is **reliability of claimed trading edge**, not greater model
complexity or larger stakes. A new audit quantifies source differences and compares
expected with realized per-contract returns. It uses the existing 1,704 training
and 559 validation contracts; no original test contracts are scored.

## Source differences: relevant, but not yet the main explanation

The final Coinbase minute close disagrees with the official Kalshi label in
72/1,704 training contracts (4.23%) and 23/559 validation contracts (4.11%).
However, that compares a final trade with a settlement based on a minute average.
It is **not a measurement of BRTI/Coinbase basis alone**.

Only zero training contracts and one validation contract have an official outcome
opposite to the entire Coinbase final-minute low/high range. Most close-label
differences could therefore be consistent with final-minute averaging. This does
not prove the underlying feeds agree; it limits what can be inferred from OHLC.

The opening Kalshi target lies outside Coinbase's corresponding minute range for
277 training contracts (16.26%) and 48 validation contracts (8.59%). Opening
Coinbase-close minus target has mean -0.468 and -0.092 basis points respectively.
Median absolute differences are 1.244 and 1.419 basis points; 95th-percentile
absolute differences are 4.953 and 5.342 basis points. These compare different
price statistics and potentially different boundary conventions, not synchronized
index ticks. Exact BRTI data would allow that ambiguity to be tested.

All required opening/final-minute candles were present for these development rows.
Source bytes are checked against cached hashes. Final-minute values are used only
after settlement for this diagnostic and are never added to forecasting features.

## Predicted trading edge is overoptimistic

The following uses refreshed next-minute forecasts/quotes, 2-cent assumed slippage
and the existing fee calculation. Each eligible opportunity gets one hypothetical
contract, isolating signal quality from bankroll sizing. This is not a second
portfolio backtest. All bucket boundaries were fixed in the audit code before
reading its output; the dates were nevertheless already inspected development data.

| Model | Predicted edge bucket | Opportunities | Mean predicted net | Mean realized net |
|---|---|---:|---:|---:|
| BTCSignal | 4–8¢ | 72 | +5.65¢ | -2.46¢ |
| BTCSignal | 8–15¢ | 22 | +10.07¢ | +1.68¢ |
| BTCSignal | >=15¢ | 3 | +19.43¢ | -48.00¢ |
| BTCRegime | 4–8¢ | 90 | +5.54¢ | -2.97¢ |
| BTCRegime | 8–15¢ | 27 | +9.86¢ | -12.48¢ |
| BTCRegime | >=15¢ | 1 | +16.73¢ | -65.00¢ |
| BTCMarketGuard | 4–8¢ | 2 | +5.09¢ | -18.00¢ |

The small highest-edge groups cannot support strong conclusions. Even the larger
groups show that treating model-minus-price as trustworthy expected profit is not
justified here. Increasing position sizes would magnify these estimation errors.
Do not select the lone positive BTCSignal bucket as a new strategy: that would be
another choice based on previously inspected outcomes.

This explains why the learned filter's abstention is sensible on this dataset,
without establishing that the filter is profitable or superior to always abstaining.
It also explains how respectable overall classification accuracy can coexist with
poor trading: trades concentrate on departures from market prices, which can be
the model's mistakes rather than opportunities. This is a hypothesis supported by
the bucket results, not an identified causal explanation of every loss.

## Decision

1. Preserve current models and the no-deployment status; no threshold/weight search
   on these dates. Keep unconditional abstention as an explicit strategy benchmark.
2. When credentials become available, perform the planned BRTI source comparison.
   It is optional, and it is not assumed to solve the model problem.
3. The next model comparison should test whether predicted net returns are reliable
   on new dates, with models/risk rules fixed beforehand. Report all opportunities,
   including rejected ones, fees, quantity limits and day-level uncertainty. Avoid
   announcing an improvement from one winning trade or a smaller loss due to abstention.

No new model is promoted, no trades are placed, and no background collector or
scheduled task has been started. The next-date evaluation has not yet been run.

## Reproduction

From outputs/kalshi-helper, using the existing local public-data cache:

```powershell
python -m helper.btc_diagnostics --cache ../../../../btc-public-cache
python -m unittest discover -s tests
```

Output: `reports/btc-source-edge-audit/report.json` includes per-event source
diagnostics; `manifest.json` records input/cache/report hashes. This module is
diagnostic-only and is separate from the forecasting and app code.
