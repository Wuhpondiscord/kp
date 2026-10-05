# BTC flow follow-up: broader testing does not establish an improvement

## Decision

Keep the new flow models experimental. Do not replace the deployed model, lower the trading threshold, increase exposure, or open the July 21–31 holdout. The earlier +$19.30 result involved three trades; expanded rolling evaluation does not confirm an edge.

## What changed

Added public June 20–July 7 history to the existing July development dates, with Binance BTCUSDT minute taker-buy volume, Coinbase BTCUSD minute closes, and actual Kalshi BTC15M historical quotes and settlement labels. Contracts closing at whole UTC hours were selected before downloading earlier outcomes. There are 732 sampled markets and 727 constructed development rows. The collector saves source-response hashes and verifies Binance's published archive checksums.

Five fixed learners were compared: market-only calibration; the original four-feature flow correction; volume-only (5/15-minute imbalance); basis-only (cross-exchange log-price gap and its five-minute change); and basis-change-only (removing the absolute USDT/USD gap). Every learner retains the same market-log-odds offset, L2 penalty .1, coefficient bounds ±.25, and training-only feature standardization. This is an ablation study, not a claim that a larger architecture is needed.

Four expanding training windows end June 28, July 4, July 10, and July 16. Corresponding evaluation windows end July 4, July 10, July 16, and July 21. Each evaluation starts after a two-hour embargo; training outcomes must settle before its boundary. The combined evaluation has 527 contracts across 23 UTC days. July dates were previously inspected: these are exploratory development results, not an untouched test.

## Matched-cohort results

Lower log loss and Brier are better. Trading starts with $1,000, risks at most 1% per bet with the existing shared exposure caps, requires a 4-cent estimated net edge, and applies 2-cent slippage plus the existing .07 fee coefficient and cent account rounding. No historical depth or actual fills are available.

| Predictor | Log loss | Brier | Simulated net P&L | Trades |
|---|---:|---:|---:|---:|
| Market benchmark | .377750 | .121078 | $0.00 | 0 |
| Market-only calibration | .381645 | .122014 | $0.00 | 0 |
| Full flow | .379743 | .121173 | −$0.46 | 2 |
| Volume only | .381157 | .121514 | $0.00 | 0 |
| Basis only | .380167 | .121630 | $0.00 | 0 |
| Basis change only | .380870 | .121704 | $0.00 | 0 |

Zero P&L with zero trades is abstention, not profitability. Full flow invested $19.46 across 41 contracts, paid $0.72 in fees, and lost 1.12 cents per contract (−2.36% on deployed money). Two entries on two days cannot establish profit uncertainty; the report suppresses misleading sparse-trade confidence intervals.

Full flow's model-minus-market Brier difference is +0.000095, with a day-cluster bootstrap 95% interval [−0.001861, +0.002186]. The interval crosses zero. These intervals are descriptive and do not adjust for repeated experiments or all serial dependence.

| Evaluation starts | Market Brier | Full-flow Brier | Full-flow P&L |
|---|---:|---:|---:|
| June 28 | .124759 | .128430 | $0.00 |
| July 4 | .139583 | .139310 | $0.00 |
| July 10 | .108872 | .107503 | −$9.62 |
| July 16 | .109706 | .107785 | +$9.16 |

Combined P&L is replayed as one continuous bankroll, rather than summing independently reset fold returns. Here the numbers coincide.

## What this reveals

1. **The recent gain is unstable across dates.** Full flow improves Brier in three windows, but the earliest window erases those improvements. On the same July 16–20 validation rows, expanding training from 189 to 610 rows changes the previous +$19.30 / three-trade result to +$9.16 / one trade, and Brier from .106750 to .107785. More history reduces the apparent recent advantage; it does not automatically improve the model.
2. **Neither isolated source is a reliable improvement.** Volume, basis, and basis-change ablations all score worse than the market in aggregate. Dropping the absolute basis level is reasonable as a robustness hypothesis, but did not help here.
3. **Input freshness is a bottleneck.** Delaying only the new features by 60 seconds increases full-flow Brier from .121173 to .122598 and log loss from .379743 to .383545; qualifying trades fall from two to zero. This is a timestamp stress scenario, not measured exchange latency or proof of causation. A live implementation needs actual arrival-time measurements before these features can support trading claims.
4. **Calibration itself can hurt.** The market-only correction is worse than the unmodified market. The full model beats that weak control but still fails to beat the market. A learned adjustment must earn its place against both benchmarks.
5. **Coefficient direction is not stable.** For example, the standardized five-minute imbalance coefficient changes from +.0208 in the first fold to −.0958 in the last. The basis-change coefficient also changes sign. Correlated features and changing scalers mean these are diagnostics, not causal interpretations.

## Retained improvements and next decision

Retain the larger reproducible dataset, fixed feature-ablation runner, delayed-input stress test, hash checks, and per-window reporting. Keep the previous model artifacts unchanged. No candidate earned deployment or holdout release. The defensible next investigation is timestamp/arrival quality and whether information adds value beyond an unmodified market price, rather than another weight search on these dates.

June's .07 fee coefficient is an assumption; the earlier July fee evidence does not establish historical June overrides. Other limits include USDT versus USD basis effects, assumed five-second publication allowance, finalized opening-target metadata, selected hourly contracts, and hypothetical fills without depth. None of these results demonstrates real-money profitability.

## Reproduction and files

From `outputs/kalshi-helper`:

```powershell
python -m helper.btc_flow_followup --cache <public-cache-directory>
python -m helper.btc_flow_followup
python -m unittest discover -s tests
```

The first command collects data and evaluates; add `--offline` once the raw cache is complete. The second evaluates the bundled derived development dataset without network access. Repeated evaluations are explicitly marked as reruns.

- `helper/btc_flow_followup.py`: collection, frozen protocol, masked models, embargoed rolling splits, and evaluation.
- `tests/test_btc_flow_followup.py`: original-model equivalence, excluded-feature invariance, artifact round trip, lag boundary, settlement purge/embargo, and dataset/holdout-date rejection.
- `reports/btc-flow-followup-v1/protocol.json`: fixed experiment choices.
- `reports/btc-flow-followup-v1/manifest.json`: data/source hashes and exclusions.
- `reports/btc-flow-followup-v1/development.jsonl`: derived development inputs and labels, including delayed feature vectors.
- `reports/btc-flow-followup-v1/models-*.json`: all five fitted models for each training window.
- `reports/btc-flow-followup-v1/report.json`: full calibration bins, scores, paired intervals, ledgers, delayed-input results, and dependency hashes.
- `reports/btc-flow-followup-v1/first-run-metrics.json`: first evaluation metrics; the final runner reproduced these exactly.

Validation: all 312 Python tests pass. July 21–31 model holdout was not loaded; no opened marker was created. This change is local research, not an HF deployment or a UI change.

Source definitions: [Binance public data](https://github.com/binance/binance-public-data), [Kalshi historical markets](https://docs.kalshi.com/api-reference/historical/get-historical-markets), [Kalshi historical candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks). Binance taker-buy volume is traded aggressor volume, not order-book depth.
