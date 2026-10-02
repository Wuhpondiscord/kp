# Observation discrepancy audit and model changes

Follow-up: [station-month uncertainty and disagreement-dependent spread](STATION-SPREAD-UPDATE.md)
adds station-specific calibration, wider block intervals, and a validation-only
spread comparison. The 181-test count below is historical; the follow-up has 187.

Date: 2026-09-24. Research only; no claim of improved prediction or profit.

The hardcoded observation CDF scale is now an explicit, externally calibrated
location/scale option. It is never optimized alongside remaining-day uncertainty.
The default is still explicitly labeled an uncalibrated 1F assumption (zero
calibration events). Existing models retain their predictions. The empirical
option is available through the research CLI and Python training API, not silently
activated for live inference.

## What the data actually identify

Kalshi historical metadata contain numeric `expiration_value` for some events.
We checked agreement across contracts and with their settled YES/NO intervals.
We did not substitute contract payouts, winning-bracket midpoints, or market
probabilities for exact daily temperatures. Of 868 original training events,
268 lacked usable numeric expiration values and 25 failed station coverage checks.
The resulting 575 city/day pairs span 164 dates, with latest settlement on
2026-04-24. No validation or test outcomes were used in this calibration.

Station histories are routine/special METARs (`report_type=3,4`), not continuous
minute observations. Coverage requires at least 20 readings and no gap over two
hours, including the edges of the fixed standard-time settlement day. This is a
coverage screen, not proof that the actual daily maximum was observed. Existing
six-hour-max boundary restrictions and the 30-minute availability assumption are
preserved. A single longest stored cache per station is selected deterministically;
different archived revisions are not merged.

For **settled maximum minus completed-day sampled maximum**:

| Estimate | Result |
|---|---:|
| Mean discrepancy | +0.1034 F |
| Sample standard deviation | 0.6980 F |
| Date-block 95% interval for mean | +0.0544 to +0.1710 F |
| Date-block 95% interval for standard deviation | 0.3533 to 1.0676 F |
| Bootstrap replicates / fixed seed | 2,000 / 1729 |

These intervals describe the eligible development sample, not transport to future
seasons or the uncertainty of a pure sensor-noise parameter. Missing exact targets
may introduce selection effects. Station counts are DEN 152, MDW 151, MIA 124,
NYC 148. Station standard deviations range from 0.31 F (MIA) to 1.18 F (DEN).
A DEN event on 2026-03-15 has a 13 F residual (47 F settlement versus 34 F sampled
maximum). It is retained and flagged, not removed to obtain a preferred scale.
The residual distribution is concentrated near zero with a long positive tail;
a pooled Gaussian is only a working approximation.

One hour before market close, the residual standard deviation is 0.731 F and
2.78% of included days still have a higher subsequently observed maximum. At two
hours, these become 0.758 F and 3.13%. One hour before market close is actually
1.02 or 2.02 hours before the station-day boundary in these records. Consequently,
late residuals do not identify pure observation error merely by construction.
They remain diagnostics, not the fitted calibration target.

The empirical option describes sampling/reporting discrepancy. Applying it to an
early-day running maximum, and multiplying its CDF by the remaining-maximum CDF,
still assumes transferability and independence. Neither was established here.
It would be premature to replace the production/default constant automatically.

## Implemented safeguards and related fixes

- External calibration includes event IDs, sample counts, station summaries,
  source hashes, interpretation and latest settlement timestamp. A defensive copy
  travels with the fitted model and its protocol.
- Training rejects calibration containing later non-training events. Historical
  calibration is allowed if its latest settlement precedes all training decisions.
- Duplicate events, censored/proxy targets, incomplete days, invalid scales and
  insufficient samples are rejected. `observed=True` with a missing/nonfinite
  maximum now raises instead of silently treating it as 0 F or returning NaNs.
- RemainingMaximum, its simple-weather baseline, and both Polymarket comparison
  arms accept the same fixed calibration. It adds no optimizer parameters.
- Aware CSV timestamps are converted to UTC rather than relabeled UTC. Index
  sorting uses absolute instants rather than timestamp text. Negative/nonfinite
  observation availability delays are rejected.

## Reproduce and use

From the workspace root:

```powershell
python work/calibrate_observation_discrepancy.py
```

This reads existing data offline and rebuilds the reports. Source data are in
`work/browser-stage-data`; the frozen training split comes from
`training/train-103cd0ff5f8a/features.jsonl`. It does not fetch new observations or
rerun price-model holdout scoring.

From `outputs/kalshi-helper`, for a new research experiment with enough settled
prospective records:

```powershell
python -m helper --data ../../work/browser-stage-data train-remaining-weather --output reports/new-calibrated-weather --observation-calibration reports/observation-calibration/calibration.json
```

`compare-polymarket` accepts the same optional flag. Omitting the flag records the
legacy assumption explicitly. Insufficient prospective data still block training;
this option does not bypass that requirement or establish an edge.

Evidence: `reports/observation-calibration/audit.json` contains individual pairs,
exclusions, uncertainty, late diagnostics and flagged extremes;
`calibration.json` is the fixed model input. The full Python suite passes 181
tests, including seven new calibration/timestamp tests. The training regression
fits the physical model and checks that calibration stays fixed; its synthetic
data test mechanics, not predictive performance.

## Remaining priorities

First reconcile the large discrepancies against original official climate reports
and high-frequency station data, especially station-day boundary coverage. Then
assess the transfer of completed-day residuals to pre-close running maxima using
separate, prospectively collected records. Do not select station-specific scales,
distribution families, or availability delays by reused trading test performance.

The prior 30-minute delay, three-hour freshness rule, and independence assumption
remain assumptions. This change makes the observation scale auditable; it does
not validate those other choices.

Source references: [NWS climate observations FAQ](https://www.weather.gov/lot/weather_observations_faq)
explains standard-time days and why CLI extremes differ from METAR observations.
[IEM temperature wagering explanation](https://www.mesonet.agron.iastate.edu/onsite/news.phtml?id=1469)
describes rounding/sampling differences. The [IEM download interface](https://www.mesonet.agron.iastate.edu/request/download.phtml?network=CO_ASOS)
distinguishes routine/special reports from the five-minute feed. These explain why
the empirical residual should not be labeled isolated sensor noise.
