# Priorities implemented and tested — September 24, 2026

This pass improves the evidence needed to decide whether to risk real money. It does not establish a profitable strategy or enable real orders.

## 1. Fresh paper evidence

Practice sessions now offer evidence recording, enabled by default in the web form. The journal preserves the session protocol, frozen model reference, market records, books, model checks (including rejected signals), outcomes when available, and NWS hourly forecasts and station reports. A local hash chain detects changes; it is not independent timestamp attestation. The model artifact is copied into the session evidence folder when present.

Live smoke run `c82733aea7f8` discovered 24 real weather contracts, recorded 72 books and 66 model checks, and fetched weather snapshots for all four supported cities with no source errors. It made zero trades under the careful promotion gate. This verifies collection, not fills or profitability. A short run cannot provide settled performance evidence.

Use the normal practice form, or from the project folder:

```powershell
python -m helper --data ../../work/browser-stage-data paper --hours 24 --poll-seconds 60 --prospective
```

Keep the process running; Ctrl+C stops it. No extended trial is running from this development pass. Evidence is under the selected data directory's `prospective/<run-id>/`. `/api/readiness` verifies recorded journals and checks the cash identity. It always reports not ready under the current evidence requirements; it is not an automated model promotion rule.

## 2. Weather architecture experiment

Added and trained an experimental city/day maximum distribution. Integrating one CDF over adjacent temperature brackets conserves event probability. It combines archived forecast maxima with an uncertain observed maximum and fits interval outcomes on training data. Incomplete bracket groups are excluded.

Validation selected **zero weather weight**, so the challenger was rejected. Test event log loss was 0.756812, identical to the normalized-market benchmark. This is useful negative evidence, not an improvement claim. The experiment is separate from live inference.

The archived daily forecast maximum is only a proxy for the remaining-day maximum. Current NWS snapshots are recorded for future research; they do not yet feed this model. Five-minute station data, complete settlement-day maxima, explicit archived forecast runs, and a fitted remaining-temperature path remain unfinished. See `reports/event-weather-results.json` and the adjacent protocol and selection files.

## 3. Rolling validation

Ran three later 30-day windows, each with five neural seeds, fixed hyperparameters and chronological validation/purging. Evaluation used the 12h and 24h horizons. All dates had been inspected previously, so these are research diagnostics rather than fresh holdout evidence.

| Test dates | Market log loss | Neural log loss across seeds |
|---|---:|---:|
| June 24–July 23 | 0.293453 | 0.293318–0.293351 |
| July 24–August 22 | 0.304507 | 0.304521–0.304752 |
| August 23–September 21 | 0.271713 | 0.270005–0.270347 |

Lower is better. The middle window worsens for every seed. Improvements are not stable across periods, so there is no basis here to increase exposure. Multi-year and repeated seasonal validation still require more history. See `reports/rolling-results.json`.

## 4. Risk and accounting

Added entry stops at 3% realized loss per UTC day and 5% realized loss per session, measured against initial bankroll. Session halt persists; settlement and accounting continue. These are entry stops, not guaranteed maximum losses: open bets can settle at additional losses. Existing 1% per-bet, 3% city/day, and 20% portfolio caps remain.

Earlier sizing stress tests remain in `EDGE-AND-SIZING.md`. Larger bets increased both outcome magnitude and uncertainty; they did not prove an edge. Readiness checks reconcile initial cash plus realized P/L minus open cost against reported cash. Exchange-side order reconciliation remains unimplemented because this app places no real orders.

## Verification and next decision

All 145 Python tests and the frontend checks passed. Added tests cover journal mutation detection, probability conservation and label independence, and loss-stop accounting. The live source smoke test is recorded separately from unit tests; its 216-record journal verifies and its cash reconciles. No real-money trial should be inferred from these changes.

The next evidence requirement is a frozen prospective strategy evaluated on enough new independent city/day events, using executable prices, costs, net profit per contract, deployed-capital return and clustered uncertainty. Collecting that evidence requires elapsed time. The new weather candidate should not replace the active model, and the rolling results do not justify raising risk.

Source reference: [NWS API documentation](https://www.weather.gov/documentation/services-web-api). The collector uses hourly forecasts and recent station observations, not five-minute ASOS observations.
