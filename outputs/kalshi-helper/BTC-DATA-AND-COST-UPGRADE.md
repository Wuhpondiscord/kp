# BTC historical data, costs and new information experiment

Implemented and exercised locally. The hosted paper lab remains on the three
previous models. A new flow-and-basis candidate is promising on validation but
has only three primary-scenario trades; it is not promoted and the reserved
evaluation cohort remains sealed.

## 1. Fee and execution audit

The official July 7, 2026 schedule gives the general taker coefficient as 0.07
times the applicable multiplier. Current KXBTC15M metadata reports quadratic fees
and multiplier 1; its historical fee-change query returned an empty array. Those
facts support the coefficient, but do not establish that every historical event
had no override. These public responses are preserved in the hash-checked cache.

Official rounding documentation distinguishes cent precision for non-direct
members and $0.0001 for direct members. The new single-fill calculator first
rounds the model fee to six decimal dollars, then rounds the total debit to the
account grid. It does not invent rebates for partial fills we did not observe.

| Original September fixed fills | Prior net P&L | Direct-member precision sensitivity | Saving |
|---|---:|---:|---:|
| Volatility | −$122.52 | −$120.74 | $1.78 |
| Logistic | −$319.83 | −$318.01 | $1.82 |
| Regime trees | −$222.99 | −$220.96 | $2.03 |

This holds trades and quantities constant; it is not a reoptimized strategy.
Cent precision exactly reproduces the original totals. Rounding is not a major
loss driver. The primary scenario retains cent rounding and 2-cent slippage.
Zero- and 5-cent slippage remain explicit sensitivity assumptions. No source in
this experiment establishes historical depth, queue position or actual fills.

Sources: [official fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf),
[rounding mechanics](https://docs.kalshi.com/getting_started/fee_rounding),
[fee-change API](https://docs.kalshi.com/api-reference/exchange/get-series-fee-changes).
The schedule was inspected through the web reader; an attempted direct PDF cache
download returned HTTP 429, so no local PDF snapshot is claimed.

## 2. A genuinely separate historical cohort

Before collecting the cohort, `protocol.json` reserved July 21–31 for evaluation.
This uses the historical Kalshi API rather than the recent-market endpoint.
It paginates the entire requested series with duplicate/cursor guards and filters
dates locally, respecting the API's mutually exclusive filters. Sampling is fixed
to contracts that close on whole UTC hours. It is not a sample of all quarter-hours.

| Partition | Dates | Usable contracts |
|---|---|---:|
| Training | July 8–15, 2026 | 189 |
| Validation | July 16–20, 2026 | 114 |
| Reserved holdout | July 21–31, 2026 | 254 |

Of 576 nominal hourly slots, the API returned 564 contracts. One lacked an eligible
quote and six were removed by split embargoes, leaving 557. Missing observations
are not filled with invented values. The labels are official Kalshi settlements;
underlying price features remain proxies for BRTI. Split boundaries enforce
settlement timing and a two-hour embargo. Metadata and structural validation of
the holdout are permitted; its model scores and trading results have not been read.

Source: [Kalshi historical market API](https://docs.kalshi.com/api-reference/historical/get-historical-markets)
and [archived candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks).

## 3. New inputs and market-relative model

The July Binance BTCUSDT minute archive is verified against its published SHA-256
checksum. Its taker-buy base-volume columns provide information our old OHLCV
adapter discarded. The four new inputs are:

1. Five-minute signed aggressor-volume imbalance: `2 × buy_volume / total_volume − 1`.
2. The equivalent fifteen-minute imbalance.
3. Binance BTCUSDT versus Coinbase BTCUSD log-price difference in basis points.
4. The five-minute change in that cross-exchange difference.

These are trade-flow measurements, **not order-book imbalance**. The basis also
contains USDT/USD effects; it is not assumed to be a risk-free price discrepancy.
Only completed minute bars with the declared five-second publication allowance
enter a feature. Missing windows and invalid buy-volume totals fail validation.
The allowance is an assumption, not a measured historical arrival timestamp.

The six-parameter model starts at market log odds and learns a regularized correction
from a bias, market calibration and four standardized features. Scaling uses training
data only. Coefficients are bounded at ±0.25 and L2 is fixed at 0.1. The matched
market-only control uses the same regularization without the four new inputs.
No September labels or test results train this model; it is a new small model,
not continued training of the deployed BTC Signal checkpoint.

Source: [Binance's official public-data schema](https://github.com/binance/binance-public-data).
Free historical order-book depth was not obtained. We did not manufacture it from
candle volumes or substitute present-day books for historical liquidity.

## 4. Validation results

All rows, costs and bankroll constraints are matched. Lower forecast scores are better.

| Predictor | Log loss | Brier | Net P&L at 2¢ slippage | Entries |
|---|---:|---:|---:|---:|
| Market / no-trade control | 0.338273 | 0.109706 | $0.00 | 0 |
| Existing volatility method | 0.382898 | 0.125350 | −$119.51 | 37 |
| Market-only calibration | 0.337481 | 0.110423 | $0.00 | 0 |
| New flow-and-basis correction | **0.328217** | **0.106750** | **+$19.30** | **3** |

The new model improves log loss by about 3.0% and Brier by about 2.7% against
market odds on these 114 validation examples. Its Brier difference is −0.0029566,
with exploratory five-day block interval **[−0.0056187, +0.0009342]**, crossing zero.
The result therefore does not establish a forecasting edge.

| Assumed extra cost | Net P&L | Entries |
|---|---:|---:|
| 0¢ | +$13.70 | 5 |
| 2¢, primary | +$19.30 | 3 |
| 5¢ | +$6.44 | 1 |

These are full policy reruns: higher costs can exclude trades, so P&L need not
decrease monotonically. The primary result involves 48 contracts costing $28.70
across only two trading days. Its 67.2% return on deployed funds is dominated by
three winning entries and must not be extrapolated to a $1,000 bankroll or future
months. The combined-feature experiment does not isolate whether flow or basis
accounts for the improvement; no post-result ablation search was performed.

## 5. Holdout and uncertainty protections

Before evaluating validation, the model artifact records a release gate: beat both
market and price-only control on Brier, produce positive primary net P&L, and make
at least 30 entries. The candidate fails the entry requirement. The release command
was tested and refused access. There is no `holdout-opened.json` or holdout report.

A sparse-trade bootstrap initially produced narrow positive bounds by resampling
only observed winners. The reporting layer now suppresses profit intervals below
30 entries or 10 trading days. A reporting-only rerun verified identical forecast
scores and P&L and marks validation as previously evaluated. No coefficients,
model regularization, sampling or trading thresholds were changed after results.

The holdout release path checks the frozen model hash, source manifest and code;
it records opening before reading outcomes and refuses a second purported first
evaluation. Passing this release gate would permit evaluation, not deployment.

## Implementation and reproduction

- `helper/btc_new_history.py`: historical API adapter, checksum-verified new source,
  causally timestamped features, fixed partitions and dataset manifest.
- `helper/btc_flow_model.py`: market-relative model, account-precision fee calculator,
  fixed cost scenarios, sparse-evidence handling and locked holdout evaluation.
- `helper/btc_cost_audit.py`: fee sensitivity on the original fixed trades.
- `tests/test_btc_flow.py`: future-bar exclusion, volume integrity, pagination,
  training-only scaling, artifact round-trip, fee arithmetic and holdout locks.

```powershell
python -m helper.btc_new_history --cache C:\path\to\btc-july-cache
python -m helper.btc_flow_model
python -m helper.btc_cost_audit
python -m unittest discover -s tests
```

Use `--offline` on the collector to reproduce from a complete cache. The optional
`--release-holdout` model command currently fails intentionally. The dataset and
results are under `reports/btc-july-flow-v1`; the fee audit is under
`reports/btc-cost-audit-v1`. API keys are not required. No real orders or background
collector were started, and no deployment change was made.

Next: preserve this candidate and its sealed cohort. Expand development evidence
on an independently predeclared period with the same information sources, rather
than lowering the entry threshold to generate enough trades or repeatedly trying
the reserved July outcomes. The present result is encouraging but too small.
