# Market-relative weather experiment

Evaluated September 23, 2026. The station-informed candidate improves historical probability accuracy relative to contemporaneous market prices. It does **not** demonstrate profitable trading. Historical test dates have been reused during development; all comparisons and confidence intervals are exploratory.

Follow-up: [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md) reports a different reconstructed dataset hash, price/horizon diagnostics and same-quote versus stale-signal scenarios. The results below remain the original saved experiment, not a guarantee of exact reproduction from mutable caches.

## Same-row results

74 test dates, 5,274 quotes and 1,776 contracts. Lower scores are better.

| Predictor | Log loss | Brier score |
|---|---:|---:|
| Market baseline | 0.199210 | 0.064322 |
| Existing neural model | 0.198429 | 0.064612 |
| Market-only tree control | 0.197622 | 0.064277 |
| Station-informed trees | **0.197390** | **0.064206** |

The station candidate reduces log loss about 0.91% relative to market prices. Its seven-day block bootstrap interval for the paired log-loss difference is [-0.002553, -0.000985]. This is not a confirmatory test on untouched data. Increasing assumed observation availability delay from 30 to 90 minutes yields log loss 0.197480 and Brier score 0.064258.

The post-result market-only control captures most of the gain. The station-versus-control interval includes zero, so the incremental value of station data remains uncertain. Chicago's individual result is slightly worse than its market baseline. Missing observations fall back exactly to market probability; approximately 66% of test rows have usable observations.

## Changes and data quality

- Corrected Chicago weather coordinates from O'Hare to Midway, the station named in the contracts. Station-rule matching rejects incompatible station names. Earlier weather experiments using the old mapping should be rerun.
- Added cached IEM ASOS observations for NYC, MDW, MIA and DEN, preserving raw response hashes. Parsing rejects wrong stations, corrected reports, conflicting timestamps and invalid temperatures.
- Features include available temperature maxima, current temperature, trends, observation age, and forecast differences from GFS/ECMWF. Settlement-day boundaries use local standard time; six-hour maxima crossing the boundary are excluded.
- Added shallow Newton-boosted trees that learn corrections to market log odds. The selected station candidate uses four leaves, 120 trees and validation-selected weight 1. Chronological train/validation/test splits retain date groups and a 48-hour settlement embargo. Model parameters are frozen before test scoring.

Historical observation timestamps do not establish original ingestion time. The delay sensitivity check cannot exclude retrospective backfills. IEM describes the archive and its limited quality control in its [ASOS download documentation](https://mesonet.agron.iastate.edu/request/download.phtml). See also [Midway station metadata](https://mesonet.agron.iastate.edu/sites/site.php?station=MDW&network=IL_ASOS) and the [NWS ASOS manual](https://www.weather.gov/media/asos/aum-toc.pdf) for observation conventions. Hourly observations are features, not substitutes for the contract's official settlement report.

## Trading results

| Assumed execution costs | Entries | Net P/L |
|---|---:|---:|
| Zero slippage, standard fees | 312 | -$13.40 |
| 2-cent slippage, standard fees | 241 | -$18.55 |
| 5-cent slippage, double fees | 153 | -$17.71 |

Historical depth is unavailable for these scenarios. Fills are hypothetical, and these figures are not verified executable returns. All tested scenarios lose money. Better forecast scores alone are insufficient to claim a tradable edge.

## Application behavior and validation

Model room's advanced settings expose **Train station-informed model** and its forecast and trading results. This experiment remains separate from live picks and automatic paper bets because it requires its own station/forecast inputs and prospective validation.

Main Auto training now also considers market-offset trees. A complete rerun on the broader 51,747-quote dataset still selected the existing neural architecture on validation performance. That live-model selection remains intact; its results must not be compared directly with the smaller station dataset. All 127 Python tests and the frontend checks passed, including observation timing, station matching, serialization, fallback behavior and model-selection checks.

## Reproduction and next evidence

From this project directory, with a separately supplied populated research archive (not included in the review ZIP):

```powershell
python -m helper.station_model --data PATH_TO_POPULATED_RESEARCH_DATA --report reports/new-station-model.json
python -m unittest discover -s tests -q
node tests/frontend_checks.cjs
```

Evidence: [primary results](reports/station-model.json), [post-result controls](reports/station-ablation.json), [main training comparison](reports/adaptive-training-comparison.json), and [frozen prospective protocol](reports/station-prospective-protocol.json).

The prospective protocol freezes the candidate and requests at least 30 newly resolved days, with predictions, source responses and executable quotes recorded before outcomes. It is awaiting data; no automated collector was scheduled. Do not retune against these historical test dates and treat the same dates as fresh confirmation.
