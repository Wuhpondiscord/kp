# Historical BTC simulation instead of a 30-day live run

Follow-up: the [loss attribution audit](BTC-LOSS-AUDIT.md) reconciles costs and
selected-trade calibration. It also clarifies that the next candle used here is
five seconds after the recorded decision, not a full minute later, and reports
forecast scores on the execution-time inputs. Original result artifacts are retained.

Historical-first research is now implemented and executed. No forward collector
or scheduled 30-day task is running. No API key is required. Weather models and
the deployed app remain unchanged.

## External data and forecasting

Downloaded six monthly BTCUSDT one-minute archives from Binance (January–June
2025), totaling **260,640 real minute candles**. Each archive is verified against
its published SHA-256 checksum. The parser explicitly handles Binance's 2025
microsecond timestamps, validates minute boundaries, OHLC ranges and duplicates,
and does not fill missing intervals.

The price-only proxy task uses quarter-hour intervals: at five minutes before close,
predict whether the final Binance minute close will meet/exceed the minute close
immediately before the interval opened. Ten minutes of that interval have already
elapsed. Thus roughly 80% accuracy is **not** an ability to predict full 15-minute
returns with 80% accuracy. The target uses BTCUSDT closes, not BRTI minute averages;
these examples are not represented as actual Kalshi contracts.

The split is chronological: 11,511 training examples in January–April, 2,967
validation examples in May, and 2,871 test examples in June. Boundary decisions have
a two-hour embargo; initial incomplete lookbacks are excluded. Parameters were
fixed before the first run. An offline replay following stricter candle-duration
checks reproduced the results; the report marks the test as previously evaluated.

| June proxy forecast | Log loss | Brier | Accuracy |
|---|---:|---:|---:|
| Constant training YES rate | 0.69318 | 0.25002 | 49.01% |
| Volatility baseline | 0.42310 | 0.13677 | 79.69% |
| Logistic model | 0.42201 | 0.13650 | 79.48% |
| Shallow boosted trees | 0.42575 | 0.13718 | 79.73% |

These scores measure the exchange-price task only. There are no invented prediction
market odds or profits attached to Binance-only examples. The logistic model's
small advantage over volatility is not an established statistical improvement.

## Applying external-trained models to actual Kalshi data

Using the existing September 19–24 development rows with official Kalshi labels
and Coinbase inputs, external-trained logistic log loss improves from the original
BTCSignal's 0.43322 to 0.42573. The corresponding tree model improves from 0.43661
to 0.43079. Both still trail Kalshi's 0.41331 market benchmark. Differences combine
more training data, earlier dates, different exchange inputs and a proxy label;
this is not an isolated experiment proving that sample count alone caused gains.

Under the same fixed paper strategy, original-versus-external logistic P&L changes
from +$5.58 to -$133.47, while tree P&L changes from -$234.51 to -$98.27. Better
overall probability scores did **not** uniformly improve trading. These are reused
development dates and broad profit intervals cross zero.

## A complete retrospective month of Kalshi paper trading

The externally trained models were then applied without retuning to **2,830 real
Kalshi contracts across September's 30 calendar days**. The original export omits
18 split-boundary examples. This supplemental full-month report was added after
the development comparison; all dates were already inspected, so it is not a new
untouched holdout. No September labels train the externally fitted models.

Each comparison starts with $1,000; the policy requires four cents of estimated
edge and limits risk to 1% of capital, at most $10/100 contracts per bet, with the
existing concurrent day/portfolio caps. Forecast inputs are refreshed at the next
minute quote. Costs include a 2-cent slippage stress and assumed 0.07 quadratic
fees. The historical record contains quotes, not executable depth; reported fills
are hypothetical and do not reproduce the prospective runner's depth checks.

| Model | Trades | Net profit | Ending cash | Return on deployed capital |
|---|---:|---:|---:|---:|
| Volatility baseline | 399 | -$122.52 | $877.48 | -3.34% |
| Logistic, externally trained | 409 | -$319.83 | $680.17 | -8.67% |
| Trees, externally trained | 435 | -$222.99 | $777.01 | -5.62% |

Cash is reused after settlement, so cumulative deployed capital exceeds the initial
$1,000. Net profit per contract is -0.80¢, -2.88¢ and -1.63¢ respectively. Exploratory
day-bootstrap intervals are [-5.25¢,+3.46¢], [-7.19¢,+1.63¢] and [-5.83¢,+2.28¢].
They ignore dependence between days and do not rerun bankroll-dependent sizing;
they do not establish an edge. No real money was traded.

## What other sources can and cannot replace

External exchange histories can provide much larger training datasets and allow
fast chronological simulations across multiple regimes. Binance is integrated now;
Coinbase already supplies the Kalshi feature input. Official Kalshi outcomes and
contemporaneous prices remain necessary to estimate profit in **Kalshi** contracts.
Spot prices cannot reconstruct what a YES contract cost or how much was executable.

An anonymous probe also confirmed Kalshi's historical market archive is accessible
(returned KXBTC15M-26AUG041945-45), providing a route to older paired evaluations.
This update did not download a complete additional Kalshi month or verify its
minute-candle coverage. That is the next useful extension: broader paired dates,
with source coverage measured and strategy settings fixed beforehand.

## Reproduce

From outputs/kalshi-helper:

```powershell
python -m helper.btc_long_history --cache ../../../../btc-binance-cache
python -m helper.btc_long_history --cache ../../../../btc-binance-cache --offline
python -m unittest discover -s tests
```

Outputs live in `reports/btc-binance-history-v1`: protocol, complete report/ledgers,
compressed proxy features/labels and source/result hashes. Original raw archives
remain in the workspace cache, outside the repository. No model is deployed.
The full suite passed **280 tests**, including timestamp, archive-integrity and
future-label-invariance checks across the historical modules.

Sources: [Binance public archives and timestamp conventions](https://github.com/binance/binance-public-data),
[Kalshi historical markets](https://docs.kalshi.com/api-reference/historical/get-historical-markets),
[Kalshi historical candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks).
