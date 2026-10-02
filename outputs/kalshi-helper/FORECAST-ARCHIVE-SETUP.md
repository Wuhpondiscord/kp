# Historical GFS/ECMWF replay: installed and exercised

2026-09-24. The project can now fetch explicit archived forecast runs, join them
to its Kalshi brackets and station observations, train both physical models, and
evaluate them on the original validation partition. No account, GPU, or new ML
dependency is needed for the selected source. This is an exploratory retrospective
workflow, separate from the still-insufficient prospective dataset.

## Hugging Face and Kaggle investigation

Public dataset cards and catalog/file-list APIs were checked. Search snapshots and
file inventories are saved under `reports/archive-source-audit`. This was a targeted
search, not a claim that every dataset on either platform has been exhausted.

| Candidate | Fit for this program |
|---|---|
| [HF Synoptic-Bench](https://huggingface.co/datasets/Aikyam-Lab/Synoptic-Bench) | Contains GFS forecasts including 2m temperature and lead times, but listed shards end in 2025 and are roughly 20–47 GB each. It is a promising longer-history source, not a small paired GFS/ECMWF station archive for our 2026 validation period. No giant shards were downloaded. |
| [HF E4DRR ECMWF references](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par) | Promising references into remote ensemble GRIB data. Requires verifying the referenced historical objects and extracting station/lead fields; its compact parquet files are references, not self-contained temperature observations. Not yet integrated or independently verified end to end. |
| [HF severe-weather GFS](https://huggingface.co/datasets/deepguess/severe-weather-env-gfs-forecast-v1) | Case-selected hazard dataset, not a continuous four-station daily-high archive. Not used as a substitute for our target population. |
| [Kaggle GFS parquet](https://www.kaggle.com/datasets/felixfdezdlm/gfs-data-parquet) | The public file list identifies Spain forecast data. Wrong geographical coverage. |
| [Kaggle GFS_FORECAST](https://www.kaggle.com/datasets/shivansh199821/gfs-forecast) | Public listing has three 12/18/24-hour NPY arrays uploaded in 2024, with no usable issue-time/location schema in that listing. Not verified compatible with our dates or stations. |
| [Kaggle weather forecasting challenge](https://www.kaggle.com/competitions/weather-forecasting-challenge-25w) | ECMWF predictors for an undisclosed station; cannot validate the required station match. |

The Kaggle ECMWF catalog also returned reanalysis/climate products. Historical
actual weather or ERA5 reanalysis must not be passed off as a forecast available
before a market decision. No Hugging Face/Kaggle download was relabeled as a
verified paired source just because its name contained GFS or ECMWF.

## Source actually integrated

[Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
accepts an explicit model initialization time. Its documentation advertises most
models from 2026-04-02; we verified actual paired responses for `gfs_global` and
`ecmwf_ifs025`. The selected ECMWF model is the 0.25-degree product, not the native
9 km historical hindcast product. Data attribution: Open-Meteo, NOAA GFS and ECMWF.
Open-Meteo's free service is for noncommercial use; review its terms before a
commercial deployment.

For this fixed experiment:

- Request four station coordinates together: NYC Central Park, Chicago Midway,
  Miami and Denver. Save raw JSON, request URL, download time and hashes.
- Use the 06 UTC initialization on the event date, Fahrenheit 2m temperature,
  three forecast days, and the original nominal 12-hour candle decisions only.
- Assume availability at initialization + eight hours and reject earlier
  decisions. This is a declared assumption, **not verified historical receipt**.
  [Open-Meteo's model timing documentation](https://open-meteo.com/en/docs/model-updates)
  distinguishes initialization, processing, and actual API availability.
- Check location, UTC, units, duplicate timestamps, nulls, array lengths, and
  complete remaining-day coverage. Hourly API values may include interpolation;
  their maximum is not the official observed settlement maximum.
- Keep the fixed standard-time station-day boundary. Join only observations
  available with the existing 30-minute delay assumption. Settlements are labels.
- The archived central forecast is the mean of the two remaining-day maxima.
  It is **not the live NWS forecast**. Humidity, wind, cloud and revision features
  remain explicitly missing. This is a historical model variant, not an exact
  replay of the live weather pipeline.
- Keep original frozen split membership before filtering for archive coverage.
  Fit and score validation only. No test scoring, live promotion or trading.

163 of 166 requested dates were retrieved. Four transient network errors recovered
on retry. Three genuine missing runs remain: ECMWF on May 12 and June 22, and GFS
on June 11, 2026. They are excluded explicitly, without substituting a newer run.
Bounded retries and cache integrity checks are implemented. Once cached, offline
reproduction needs no network.

## Actual training and validation result

| Partition | Bracket rows | City/day events | Date groups |
|---|---:|---:|---:|
| Training | 520 | 88 | 22 |
| Validation, before price filter | 1,591 | 268 | 67 |
| Reserved test, **not scored** | 1,766 | 295 | 74 |

Training date groups run April 3–24; validation April 28–July 6. These UTC groups
can be one day after the named station event. The price-filtered validation
comparison uses 730 rows across 257 events and 67 dates. Training uses all eligible
brackets, not only the price-filtered examples. External observation calibration
uses 493 exact-settlement pairs that settled before the first training decision.
It is station-specific, with the declared minimum-count fallback, and is identical
between arms.

| Validation model | Daily log loss | Daily Brier | Event/date-weighted ECE |
|---|---:|---:|---:|
| Constant physical spread | 0.731025 | 0.236000 | 0.127774 |
| Disagreement-dependent spread | 0.731025 | 0.236000 | 0.127773 |
| Kalshi midpoint baseline | 0.495074 | 0.164566 | 0.024517 |

The new spread coefficient fitted to its lower bound, **zero**. The paired
validation log-loss difference is about +0.000000057, with seven-day block interval
[-0.000002013, +0.000002220]. Small prediction differences arise from numerical
optimization, not useful learned spread variation. Both physical models perform
substantially worse than the market here. ECE uses fixed bins and is descriptive.

This is a successful real-data integration and a negative first model result.
Only 22 spring training dates remain because the archive starts near the end of
the original training split. We did not move the split to obtain better scores.
Previously inspected historical dates, assumed publication timing, limited seasonal
coverage and the different central forecast prevent a claim of untouched/live
validation. No profitability test was performed and no active model changed.

## Files and commands

Implementation: `helper/forecast_archive.py`, `helper/archive_replay.py` and
`tests/test_forecast_archive.py`. The full Python suite passes **194 tests**,
including actual cache round trips, offline refusal for missing runs, tamper
detection, bounded retries and look-ahead/boundary checks.

Downloaded responses: `../../work/browser-stage-data/single-run-archive`.
Dataset, calibration, model pickles, protocol and results:
`reports/historical-forecast-replay`.

From `outputs/kalshi-helper`:

```powershell
python -m helper --data ../../work/browser-stage-data prepare-weather-replay --features ../../work/browser-stage-data/training/train-103cd0ff5f8a/features.jsonl --output reports/new-archive-replay --observation-audit reports/observation-calibration/audit.json --offline
```

Omit `--offline` to fetch missing runs. The convenience script
`../../work/build_forecast_replay.py` runs this workspace's configured experiment.
Use a new output directory to preserve earlier reports.

The next useful data expansion is verified **earlier** explicit-run GFS/ECMWF
coverage to strengthen the training period. Merely enlarging validation with more
recent data, relabeling reanalysis, or moving inspected dates into training would
not resolve the present evidence limitations.
