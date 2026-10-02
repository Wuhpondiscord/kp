# Board and model coverage audit — September 22, 2026

The reported problem was real: discovery exposed markets beyond the selected model's scope, and refreshing prices reset the default filter to all markets. The model was also restricted to a fixed 5–25 hour window regardless of the saved training artifact.

## Changes

- Daily-weather browsing now defaults to **Ready for this model**, including after price refresh. Pasted links and other categories remain visible even when unsupported.
- A coverage banner gives the supported contract count, cities and time window. Unsupported cards distinguish **Check later**, **Window closed** and **Different model needed**.
- Neural training now samples actual archived quotes around 26, 24, 18, 12, 6 and 3 hours before close. The usable range is the intersection observed across cities and train/validation/test partitions. It is saved with the artifact and retained when switching models. Older artifacts keep their original limits.
- Automatic practice prioritizes supported contracts before liquidity. Previously, liquid unsupported contracts displaced supported contracts with thin books.
- Recommended training is disabled while a research job runs. Regression checks cover the filter reset, disjoint coverage windows and discovery ordering.

## Browser acceptance results

Tested the running app at http://127.0.0.1:8766 using the isolated copy of real data in `work/browser-stage-data`; source changes are in this project. The trained artifact and test sessions are in that copy, not the original `data` directory.

- Recommended training completed through the browser in 35.82 seconds: 56 epochs, best checkpoint 36, validation-selected neural weight 75%.
- Model `train-a52cf31ccde0`: 30,806 training samples, 9,671 validation samples, 10,438 final-test samples; 832 boundary samples excluded. Dates are grouped and settlement-embargoed. Repeated quotes are not independent outcomes.
- Saved coverage: 3.9833–26.9833 hours before close, displayed as approximately 4–27 hours; NYC, Chicago, Miami and Denver daily highs only.
- The weather board displayed 24 supported contracts from 48 discovered contracts. All-bets view explained when tomorrow's contracts enter coverage. Mentions showed **Different model needed**, with market odds explicitly distinguished from model odds.
- Practice session `191ee97f2ea0` started and stopped through the browser. It checked 24 contracts, used the saved model snapshot, and reported no source errors. Only two contracts had usable two-sided books at the observed pass; the others reported missing book sides. No simulated fills occurred and the $1,000 balance remained unchanged. This short smoke test does not establish day-long reliability or profitable execution.
- 113 Python tests and the frontend regression checks passed.

## Model performance

| Final-test metric | Trained model | Market baseline |
|---|---:|---:|
| Log loss, lower is better | 0.208091 | 0.208974 |
| Brier score, lower is better | 0.067651 | 0.067400 |

The seven-day block bootstrap interval for log-loss difference is [-0.002414, +0.000590], including zero. Test dates have previously been inspected, so automatic promotion remains blocked. Indicative one-contract quote scenarios lost $11.67, $15.72 and $25.70 under increasing costs; they lack historical book depth and are not verified trading returns. Broader sampling changes the evaluation dataset, so these numbers should not be directly compared with earlier narrow-horizon reports.

The model still learns corrections to market prices from price, spread, city and time remaining. It is not a broad weather or mentions forecasting model. The separate archived-weather challenger previously received zero weight on validation. Expanding the UI's market list does not create trained coverage for rain, mentions, post counts or other categories.

## Assessment and next work

Subjective ratings after these checks: everyday usability **7/10**, training workflow **8/10**, predictive evidence **4/10**, demonstrated profitability **1/10**. The coverage failure is substantially clearer and browsing works, but the overall project is not yet a dependable betting helper.

The next model work should build timestamp-aligned forecast/station datasets and category-specific outcomes, followed by untouched prospective evaluation. Weather needs exact settlement rules and forecasts available at prediction time; mentions needs pre-event transcripts and event-specific word labels. More layers or removing coverage limits would not address the current data and evidence gaps. Keep collecting live books for execution validation.

Machine-readable evidence: `reports/coverage-audit.json`.
