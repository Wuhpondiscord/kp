# Why the BTC paper strategies lost money

The strongest bottleneck is inaccurate estimated edge on the trades selected by
the models. Costs then consume the modest gross returns. Increasing exposure or
adding a larger neural network is not justified by this evidence.

This audit reconstructs the existing September replay: 2,830 Kalshi contracts
across 30 calendar days, with models trained on January–April 2025 Binance data.
Dates have already been inspected. This is diagnosis, not a new holdout, strategy
search, or proof that the simulated orders could have filled. Weather is unchanged.

## 1. The selected bets are much less reliable than the models predict

Each entry receives equal weight in this table; probabilities refer to the side
purchased, which may be YES or NO.

| Forecast | Entries | Predicted win rate | Actual win rate | Market-implied win rate |
|---|---:|---:|---:|---:|
| Volatility baseline | 399 | 47.61% | 37.09% | 36.24% |
| External-trained logistic | 409 | 58.91% | 46.21% | 46.71% |
| External-trained regime trees | 435 | 52.46% | 40.92% | 40.38% |

The market is much closer to realized win frequency on these selected cohorts.
Conditional overconfidence matters more to this strategy than overall accuracy:
the strategy buys precisely where the model disagrees with the market. A model
can forecast most outcomes reasonably well while selecting bad disagreements.

A 3,000-resample day-block bootstrap gives exploratory 95% intervals for mean
predicted-minus-realized win probability of 6.40–15.08 percentage points for
volatility, 8.30–17.65 for logistic, and 6.95–16.45 for trees. These intervals
ignore cross-day dependence and the history of inspecting/selecting experiments;
they are diagnostic, not confirmatory evidence from an untouched evaluation.

## 2. Where the dollars went

The audit holds the original selected contracts and quantities fixed. Every trade
reconciles as midpoint P&L minus spread, assumed slippage and fees equals net P&L.

| Forecast | Contracts | Gross midpoint P&L | Spread cost | Assumed slippage | Fees/rounded debit | Net P&L |
|---|---:|---:|---:|---:|---:|---:|
| Volatility | 15,277 | $386.88 | $52.63 | $305.54 | $151.23 | **−$122.52** |
| Logistic | 11,114 | $72.06 | $43.08 | $222.28 | $126.53 | **−$319.83** |
| Regime trees | 13,643 | $254.18 | $51.50 | $272.86 | $152.81 | **−$222.99** |

Midpoints are not executable purchase prices. The simulation assumes 2 cents of
slippage and a 0.07 quadratic fee coefficient; it does not measure realized fills.
Subcent quote arithmetic and rounded total debits explain small rounding differences.

Removing only assumed slippage, holding original fees, trades and quantities fixed,
changes these totals to +$183.02, −$97.55 and +$49.87. These are accounting
sensitivities, not zero-slippage strategy reruns. They justify verifying execution
costs, not lowering an assumption until the backtest wins. Archived candles do not
establish available depth or whether a five-second-later order would have filled.

The models predicted net profits of approximately $1,141, $890 and $1,079 on
these same fills. The large prediction-to-realization gap cannot be explained by
fees alone: the expected-profit calculation already subtracts the modeled costs.

## 3. A timing mismatch obscured the comparison

The previous report's “next-minute quote” means the next candle relative to the
last eligible candle, not a minute after the decision. In every audited row:

- Execution timestamp is **5 seconds after the recorded decision**.
- Decision-time spot features are **60 seconds old**.
- Refreshed execution-time spot features are **5 seconds old**.

The recent BTC replay does refresh model probabilities at execution. The issue
is that headline forecast scores and trading results use different input freshness
and clocks. It is misleading to interpret them as evaluations of identical predictions.
The archived quote availability delay is an assumption, not a measured receipt time.

Recomputing forecast metrics on the actual execution rows gives:

| Predictor | Execution-time log loss | Execution-time Brier |
|---|---:|---:|
| Market benchmark | **0.398155** | **0.127812** |
| Volatility | 0.406408 | 0.130673 |
| Logistic | 0.407702 | 0.131356 |
| Regime trees | 0.409666 | 0.131493 |

All three still trail the market when compared at the same timestamp. The timing
correction improves reporting integrity; it does not uncover a hidden advantage.

## 4. Other bottlenecks and things that are not the primary fix

**Sizing is not the sole cause.** Equal-contract contributions on the same selected
trades still sum to −$11.14, −$17.55 and −$14.29. This calculation allocates the
actual bulk fees per contract; it is not a separate one-contract backtest with
recomputed fee rounding. Larger bankroll exposure would magnify an unproven strategy.

**Losses are not confined to extreme prices.** Purchased-side asks between 30 and
70 cents account for −$159.15, −$198.57 and −$202.97 respectively. Some cheaper
buckets win historically, but selecting those after seeing the outcome would be
another post-hoc strategy search, not a defensible fix.

**More candles do not necessarily mean more relevant training data.** Binance
BTCUSDT minute-close proxy labels differ from Kalshi's settlement index and averaging
rule; evaluation inputs come from Coinbase. The earlier [source audit](BTC-DIAGNOSIS.md)
measures these disagreements. Current feature mean shifts are small (largest about
0.16 training standard deviations), but means cannot rule out tail, conditional,
exchange-basis or label shift. This audit cannot assign a causal share of losses
to each source mismatch.

**The effective sample is smaller than the contract count.** Intraday BTC trades
share market regimes. Thirty dates, repeatedly inspected, cannot support repeated
model/threshold searches or reliable claims of a small net edge.

## What to work on next, in order

1. Make decision, feature availability, quote receipt and execution timestamps
   explicit throughout evaluation. Report decision-time and execution-time scores
   separately; match training freshness/horizon to the intended inference path.
2. Add untouched historical months of paired Kalshi quotes/outcomes and underlying
   inputs where available. Keep proxy forecasting and exact-contract evaluation
   separate. Exact settlement-index data would reduce a known mismatch; it is not
   required to keep the public-data app usable.
3. On training/development data only, evaluate market-relative probability corrections
   and out-of-fold calibration of selected edge. Compare against market-only and
   no-trade baselines. Freeze the policy before evaluating new dates. A trading
   filter needs held-out forecast errors, not in-sample model confidence.
4. Verify historical fee schedules and execution assumptions. Report a fixed,
   predeclared cost-sensitivity range. Public current books can inform future depth
   measurements; they cannot reconstruct historical executable depth.
5. Revisit architecture only if these aligned evaluations identify a reproducible
   forecast weakness. Do not increase risk, optimize winning historical buckets,
   or treat lower assumed costs as a model improvement.

These are priorities, not changes silently applied to the frozen experiment. No
new strategy was tuned, no 30-day collector was started, and nothing was deployed.

## Reproduction and verification

From the app directory, run:

```powershell
python -m helper.btc_loss_audit
python -m unittest discover -s tests
```

The audit validates source report/dataset hashes, reconstructs the deterministic
external-trained models, checks every ledger cost and reconciles reported P&L.
The full suite passed **283 tests**. New tests cover YES/NO attribution, invalid
ledger payouts/costs, invalid probabilities and an empty ledger.

- Implementation: [helper/btc_loss_audit.py](helper/btc_loss_audit.py)
- Regression tests: [tests/test_btc_loss_audit.py](tests/test_btc_loss_audit.py)
- Full metrics and per-trade attribution: [report.json](reports/btc-loss-audit/report.json)
- Audit/source integrity references: [manifest.json](reports/btc-loss-audit/manifest.json)
- Original experiment: [BTC-HISTORICAL-SIMULATION.md](BTC-HISTORICAL-SIMULATION.md)

Reproduction requires the original local research artifacts; the deployment seed
alone is not a replacement for the historical datasets.
