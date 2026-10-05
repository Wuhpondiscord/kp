# Custom models with one-second historical inputs

## Decision

Retain the new validated data pipeline, but do not replace the existing fixed-anchor model. The richer short-horizon inputs improve some periods and worsen others; their aggregate forecast scores do not beat the custom baseline. The combined model loses $13.11 in the unchanged primary trading scenario. The original custom models, hosted app and sealed holdout remain unchanged.

## Data upgrade

Downloaded 32 public Binance BTCUSDT one-second daily archives, June 19–July 20, 2026, totaling 73,600,191 compressed bytes. Each is verified against its published SHA-256 checksum. Daily parsing requires exactly 86,400 consecutive microsecond-stamped one-second bars with valid OHLC, volume, taker-buy volume and close timestamps. Only windows required by the development contracts are retained in memory for features: 87,967 seconds supporting 727 constructed development rows.

The collection was interrupted and resumed from the checked cache; no partially written results were treated as completed. Missing bars cause a failure, not silent forward filling or row selection.

Binance documents public one-second klines and the volume/trade fields. These are trade aggregates, **not order-book depth**, and this is finer information from an existing exchange source, not an independent settlement-price feed. [Official source documentation](https://github.com/binance/binance-public-data).

## Timing and features

Keep the existing five-second assumed publication allowance. For a decision at `t`, use second bars ending no later than `floor(t − 5 seconds)`. A bar starting at that cutoff is excluded. Thus the experiment does not assume a faster feed than the old minute-bar baseline. At the sampled decision times, both feature sets share the same latest completed minute endpoint; the new inputs expose the path inside that minute.

Four fixed features:

1. Signed taker-buy imbalance over the last 15 seconds.
2. Log return over 15 seconds, expressed in basis points.
3. Five-second imbalance minus 60-second imbalance.
4. Log return over five seconds, expressed in basis points.

A genuinely zero-volume window has zero imbalance. Missing seconds are different and are rejected. Trade counts are retained for provenance/diagnostics, not added as an extra fitted feature. Feature endpoint and assumed availability timestamps are recorded per contract. A stress scenario delays only these new inputs by 60 seconds; existing five/fifteen-minute flow inputs and market quote remain current.

The timestamps are source event times plus an assumed allowance. They do not prove actual historical local receipt times or executable liquidity.

## Independent aggregation audit

For all 727 decision windows, reaggregate the last 60 second-bars and compare with the separately archived one-minute source:

| Field | Maximum absolute difference |
|---|---:|
| Closing price | 0 |
| BTC base volume | 2.84 × 10⁻¹⁴ |
| Taker-buy base volume | 7.11 × 10⁻¹⁵ |

All 727 windows pass. This validates source alignment and aggregation to numerical precision. It does not prove the financial features are predictive or establish real-time publication latency.

## Matched custom-model comparison

Use the existing fixed market-logit architecture, training-only standardization, L2 .1, coefficient bounds ±.25 and the same four chronological splits. Compare new inputs alone with old flow plus new inputs. No parameter search. All 527 outer evaluation contracts across 23 days remain the same; earlier training rows are separate. These are previously inspected development dates, not new independent outcomes.

| Predictor | Brier | Log loss | Primary P&L | Trades |
|---|---:|---:|---:|---:|
| Market | .121078 | .377750 | $0 | 0 |
| Existing fixed anchor | **.120761** | **.377420** | $0 | 0 |
| New second-level features only | .121173 | .378326 | $0 | 0 |
| Old flow + second-level features | .121003 | .378276 | −$13.11 | 10 |
| Combined, new inputs delayed 60s | .121639 | .380300 | +$34.59 | 12 |

Trading retains $1,000 initial bankroll, 1% risk/shared caps, fees, four-cent minimum net edge and two-cent slippage. Zero trades are abstention. The delayed scenario's positive P&L is accompanied by worse forecast scores and only 12 trades on nine days; it is not selected as an improvement. Changing feature age changes which trades qualify, so P&L need not decline monotonically with stale data.

Combined-model Brier minus fixed-anchor Brier is +.000242, with day-cluster bootstrap 95% interval [−.001164, +.001697]. The interval crosses zero and does not adjust for repeated experimentation. Sparse-trade profit intervals remain suppressed.

## Period consistency

| Evaluation starts | Fixed-anchor Brier | Combined Brier | Combined period P&L |
|---|---:|---:|---:|
| June 28 | .126594 | .124508 | $0 |
| July 4 | .138557 | .141004 | −$26.33 |
| July 10 | .108658 | .110559 | +$4.54 |
| July 16 | .107540 | .105878 | +$9.16 |

The new inputs help the first and last periods, but worsen the middle two enough to erase the gains. The aggregate bankroll replay need not equal the sum of separately reset period profits. The delayed scenario's positive return is concentrated in the July 4 period rather than repeated across all periods.

## What this tells us

The data/timestamp work succeeded: source checksums, contiguous bars, availability boundaries and minute aggregation all validate. Forecast generalization did not. A missing-bar or obvious second/minute mismatch is therefore not a supported explanation for the result.

Finer granularity alone did not add a stable, tradable correction to current market prices in this setup. That is narrower than concluding that all high-frequency inputs are useless. These four summaries still lack depth, queue position, actual receipt times and a direct exact-settlement reference. Increasing model complexity or selecting the lucky delayed scenario would not fix those information limitations.

Retain the one-second archive loader, integrity/availability checks, aggregation audit and reusable derived inputs. Preserve the previous model as the baseline. No candidate earned deployment or holdout evaluation.

## Reproduce and review

From `outputs/kalshi-helper`:

```powershell
python -m helper.btc_microdata --cache <second-archive-cache>
# Add --offline after all archives are cached.
python -m helper.btc_microdata
python -m unittest discover -s tests
```

The second command evaluates the saved derived inputs without network access. It reproduced the reported metrics and primary trading results; the final report is marked as a rerun.

- `helper/btc_microdata.py`: source loader/parser, four features, timestamp attachment, custom correction models, aggregation audit and benchmark.
- `tests/test_btc_microdata.py`: cutoff/future exclusion, delayed inputs, zero-volume versus missing data, full-day timestamp validation, checksum failure, artifact round trip and frozen inference scaler.
- `reports/btc-microdata-v1/protocol.json`: frozen choices.
- `reports/btc-microdata-v1/manifest.json`: source and derived-data hashes.
- `reports/btc-microdata-v1/features.json`: current and delayed features with availability timestamps.
- `reports/btc-microdata-v1/aggregation-audit.json`: source-alignment evidence.
- `reports/btc-microdata-v1/models-*.json` and `report.json`: fitted models, all metrics, period comparisons, uncertainty and simulated ledgers.

The raw 73.6 MB archive cache lives outside the Git checkout. The existing holdout remains sealed and no hosted changes were made.
