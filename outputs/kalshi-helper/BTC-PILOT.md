# BTC 15-minute pilot — 3 October 2026

The first BTC experiment does **not** establish a trading edge. It adds a separate,
reproducible research pipeline; the weather models and live app selections are unchanged.
The BTC models are not promoted to live recommendations or automatic paper trading.

## What was tested

Kalshi KXBTC15M resolves on the **final-minute average BRTI versus the opening target**,
with equality resolving YES. It is not a contract on touching a price during the interval.
Coinbase BTC-USD candles supply explanatory features, not settlement labels. The
pipeline uses Kalshi's official resolved outcomes and checks published settlement
values against the target where those values are available.

Sources: [Kalshi contract example](https://kalshi.com/markets/kx/test/kxbtc15m-26sep090900),
[Kalshi minute candles](https://docs.kalshi.com/api-reference/market/batch-get-market-candlesticks),
[Coinbase candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles),
[CF BRTI](https://www.cfbenchmarks.com/data/indices/BRTI).

The data contain 2,848 settled markets during September 2026. There are 1,704
training examples (September 1–18), 559 validation examples (September 19–24),
and 567 test examples (September 25–30), with 18 examples removed at split
boundaries by the settlement and two-hour embargo rules. No rows were rejected
by the final data validator. There are 32 fewer markets than an uninterrupted
96-per-day calendar would imply; the API response is not proof of complete exchange
coverage. Missing markets are not fabricated.

One prediction is made five minutes before each contract closes. Features require
120 contiguous completed Coinbase minute bars. Both exchanges' candles are assumed
available five seconds after their end timestamp. At the primary decision this
usually means the latest spot observation is one minute old. It is a five-minute
remaining-horizon experiment, not a full fifteen-minute forecast.

## Models

* **BTCVolatility:** a zero-drift Gaussian return baseline using an EWMA of squared
  minute returns, distance from the target and time remaining. It approximates the
  final-minute average with a terminal-price distribution; it does not model BRTI
  averaging, basis or jumps explicitly. No learned coefficients.
* **BTCSignal:** L2-regularized logistic regression on target distance in volatility
  units, 1/5/15/60-minute returns, volatility and volume ratios, candle range, and
  UTC time of day. No Kalshi price features.
* **BTCMarketGuard:** regularized logistic correction around Kalshi's market logit,
  using the same spot features plus market logit and spread.
* **Price-only control:** the same market-offset architecture using only market
  logit and spread. This tests whether spot features add information beyond price.
* **Market:** unmodified YES bid/ask midpoint, the required forecast benchmark.

Scalers and coefficients are fitted on training only. Architectures and a single
paper strategy were specified before the first results. There is no neural network
or pretrained backbone in this pilot. Research on [DeepLOB](https://arxiv.org/abs/1808.03668)
and [digital-asset limit order books](https://arxiv.org/abs/2010.01241) motivates
investigating order-book features, but those studies do not establish profitability
for these Kalshi contracts from minute OHLCV data.

## Forecast results

Lower log loss and Brier are better. Accuracy alone ignores confidence and costs.

| Model | Validation log loss | Test log loss | Test Brier | Test accuracy |
|---|---:|---:|---:|---:|
| Market | 0.413306 | 0.409925 | 0.131144 | 81.66% |
| BTCVolatility | 0.427762 | 0.431890 | 0.137304 | 80.78% |
| BTCSignal | 0.433219 | 0.439909 | 0.142515 | 79.89% |
| BTCMarketGuard | 0.411535 | 0.414348 | 0.133642 | 80.78% |
| Price-only control | 0.411718 | 0.407783 | 0.130801 | 81.83% |

BTCMarketGuard's small validation advantage reversed on test. Adding spot features
performed worse than the price-only control. The control's Brier improvement is
only 0.000343 (about 0.26% relative skill); its six-day clustered interval for
model-minus-market Brier is [-0.000653, -0.000047]. That exploratory interval uses
only six clusters, ignores dependence across days and multiple-model selection,
and must not be read as robust evidence of a durable edge. Ten-bin calibration
counts, means and outcomes are included in report.json.

## Paper trading

All scenarios start with $1,000 and use at most 1% of current capital per bet,
capped at $10, 100 contracts, $30 concurrent exposure per BTC UTC day and $200
total concurrent exposure. Entry requires at least four cents of estimated edge
after the assumed fee. The fee is 0.07 × quantity × price × (1-price), rounded up
to the cent by the existing engine. Historical fee validity and executable depth
are not verified. The day budget is shared across BTC contracts; it is not a daily
loss stop. Position equity is carried at cost until settlement.

| Model | Same quote, 0¢ | Same quote, 1¢ | Same quote, 2¢ | Next minute, 0¢ | Next minute, 1¢ | Next minute, 2¢ |
|---|---:|---:|---:|---:|---:|---:|
| BTCVolatility | -$152.87 | -$154.25 | -$16.69 | -$238.82 | -$149.82 | -$152.93 |
| BTCSignal | -$379.43 | -$341.99 | -$246.61 | -$428.85 | -$369.30 | -$358.23 |
| BTCMarketGuard | -$27.60 | +$10.57 | +$5.90 | -$21.46 | -$16.38 | -$1.85 |
| Price-only control / Market | $0 | $0 | $0 | $0 | $0 | $0 |

Same-quote fills are optimistic. Next-minute results recompute **both** spot inputs
and market probabilities at the later quote; they are a different decision-time
scenario, not guaranteed fills for the earlier signal. These later observations
are slightly outside the single training horizon. Both scenarios still assume
that archived quotes remained executable after publication. Higher slippage can
remove losing trades by failing the edge threshold, so P&L need not decline
monotonically with slippage. Selecting the best column would be test-set tuning.

For the next-minute +2¢ scenario (a stress case, not a preselected winner):

| Model | Entries | Contracts | Capital reused/deployed | Net per contract | Return on deployed capital |
|---|---:|---:|---:|---:|---:|
| BTCVolatility | 89 | 4,118 | $783.93 | -3.71¢ | -19.51% |
| BTCSignal | 130 | 5,616 | $1,052.23 | -6.38¢ | -34.04% |
| BTCMarketGuard | 9 | 163 | $85.85 | -1.13¢ | -2.15% |

The price-only control and market generate no entries under the fixed threshold;
their zero P&L is abstention, not a profitable strategy. Trade quantities are not
independent examples. Exploratory day-bootstrap net-per-contract intervals at
+2¢ are [-12.38¢, +6.56¢], [-13.25¢, +0.47¢], and [-42.58¢, +41.04¢], respectively.
All include zero. These ratios include no-trade calendar days in sampling, exclude
undefined all-zero-trade draws, and do not rerun bankroll-dependent sizing on each
bootstrap path. Full opportunity counts, fees, ledger and ROI intervals are in the
machine-readable report.

## What this suggests, and what it does not

The evidence supports retaining the market as the benchmark. The Coinbase-only
signal is less informative than the market here; the market-offset model reduces
the damage but does not supply reliable tradable departures. Possible mechanisms
include Coinbase/BRTI basis, minute-bar latency, missing order-book information,
and changing volatility regimes. This experiment does **not** isolate which
mechanism causes the gap. Increasing model size cannot be assumed to fix it.

Useful next work is to record synchronized prospective spot, Kalshi quotes/depth,
actual receipt timestamps and the exact contract target; quantify basis and delay;
then declare a new comparison on untouched dates. A final-minute-average volatility
baseline and multivenue features are justified hypotheses for that future test.
Do not tune these six exposed test days until a profitable setting appears.

BTC and weather probabilities concern different events and should not be directly
blended. They can share calibration, temporal validation, execution accounting and
risk infrastructure. Raw accuracy, Brier, log loss and dollar P&L from the earlier
weather experiments are not a fair ranking against BTC: event difficulty, horizon,
dates, turnover, and quote assumptions differ. Compare each with its own market
benchmark, and eventually compare simultaneous prospective paper ledgers using
the same risk rules. Neither track currently establishes net profitability.

## Reproduction and audit

From outputs/kalshi-helper:

```powershell
python -m helper.btc_research --cache ../../../../btc-public-cache --output reports/btc15m-pilot-v1 --offline
python -m unittest discover -s tests
node tests/frontend_checks.cjs
```

The raw cache is local to the workspace, outside the repository. To collect afresh,
omit --offline and use a new output directory. Kalshi's moving historical cutoff
may require an archive adapter by then; this pilot deliberately refuses to treat
an inaccessible historical window as empty data. The included train/validation/test
JSONL, JSON coefficients, protocol and manifest support inspecting the fitted
experiment without network access. The manifest lists 329 source response hashes
and URLs, plus result-file hashes. It does not redistribute the complete raw cache.

The final report is marked test_previously_evaluated=true: the initial run was
replayed during correctness testing. Fixes refreshed quote provenance, included
no-trade days and ROI in profit intervals, accurately marked recomputed execution,
and checked settlement-rule variants and optional settlement values. No model
architecture, coefficients' hyperparameters, strategy threshold or test window
was selected from the results. Final fitted results reproduce the initial run.

Validation: **247 Python tests passed**, including nine new BTC tests, and all
existing frontend checks passed. BTC tests cover publication delays, future-bar
invariance, missing/invalid bars, train-only normalization, actual learning on a
known signal, coefficient serialization, label-independent inference, settlement
rules and ties, chronological embargo, cache integrity, execution provenance and
day-block accounting. This is correctness evidence, not evidence of profitability.
