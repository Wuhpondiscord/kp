# Current audit · v0.4

The current results are in [AUDIT.md](AUDIT.md): 89 automated tests and seven real-source/training checks passed. The records below describe earlier versions and smaller datasets.

---

# Training upgrade · v0.3

September 19, 2026. **57 automated tests pass**, plus Python compilation and JavaScript syntax checks. New tests verify real weight updates, training-only scaling, exact best-epoch selection, test-label isolation, whole-date splits, configuration bounds, checkpoint reload, cancellation, conservative fees, unsupported-market abstention and stale-quote gating.

HTTP integration exercised training, live progress, cancellation, market discovery and individual contract explanations. Cancellation retained the prior active model. A fresh 150-epoch-budget run (`train-9fae5cd6ffb0`) exposed 19 distinct progress updates and selected epoch 100; the run stopped when the validation patience limit was reached. The full record is in `reports/training-upgrade-check.json`.

The actual split contained 4,002 training samples, 1,215 validation samples, and 1,432 final-test samples, with 427 purged at boundaries. A live Miami temperature contract returned a neural probability, conservative net edges, validation/test metrics and four input sensitivities. Its decision was Pass. An unsupported rain contract correctly returned no invented model probability.

The model did not pass the automatic validation gate. Prior final-test dates were reused, and no claim of profitable trading is supported. New UI paths received JavaScript/HTTP checks, not a browser visual pass. The earlier v0.2 evidence follows for continuity.

---

# Validation · v0.2

Checked on Windows / Python 3.13, September 19, 2026.

## Automated checks

48 automated tests pass with `python -m unittest discover -s tests -v`. Python compilation and `node --check helper/static/main.js` pass.

Coverage includes delayed fills, changing prices, book depth and age, shared risk caps, fee rounding, bankroll conservation, YES/NO settlements, finalized API status, conflicting outcomes, replay cutoffs, duplicate snapshots, synthetic provenance, URL/series/contract resolution, unsafe URL rejection, event fee overrides, initialization failure persistence, and auto-mode refusal to trade stale market baselines.

Research tests exercise whole-date grouping, all-label embargoes, future-information rejection, holdout-label isolation, historical decimal price units, exclusion of later candles from features, later-quote scenarios, missing quote skips, stress costs, paired bootstrap determinism, model hash verification, and exact calibration-bin boundaries. Forecast tests confirm that NWS period probabilities are not fabricated into full-event contract probabilities.

## Live API / server checks

- The user's example URL resolves to 23 real city contracts.
- NWS forecast periods were retrieved for 22 mapped settlement locations. `CLIDJT` is explicitly unsupported.
- The discovery board contains 55 real market snapshots; refresh dates are shown.
- HTML, JavaScript, CSS, state, and board routes return successfully.
- Invalid external links return HTTP 400; cross-origin mutations return HTTP 403.
- A roughly 54-second live paper session using the entire rain event reached its deadline, checked all 23 contracts, and recorded no errors. Bankroll stayed $1,000, with zero trades, correctly enforcing unsupported-model gating. Saved run: `6e0fcb9b551a`.
- Starting a paper session preserves the analyzed weather data.
- The user's prior failed series-as-contract session was stopped, and its report remains saved.

See `reports/integration-v2.json` for the captured checks. This version received HTTP/static verification, not a new browser visual pass. The earlier v0.1 browser check does not establish v0.2 visual correctness. No full-day soak test has been completed.

## Real historical evaluation

2,400 actual settled contracts, 100 calendar-day groups, 7,076 quote samples. Three expanding validation folds and a 48-hour settlement embargo. All stations on a day stay in the same partition. Final test: 20 days, 480 contracts, 1,432 samples.

Selected candidate: calibrated market odds. Final log loss 0.194851 versus 0.195267 for market prices. Paired difference interval: −0.001172 to +0.000336. The gate does not pass: uncertainty includes no improvement, and there are fewer than 30 final-test days. Brier score is slightly worse than the market baseline. No profitable-model claim is supported.

A calibration-bin boundary bug discovered during testing was repaired; the evaluation was regenerated with the same splits and fitting procedure. The new report marks the holdout as reused and cannot promote a model. The original report remains in SQLite. `reports/real-data-evaluation.json` contains the latest complete report, forecasts, scenarios, provenance hashes, warnings, and limitations.

Historical dollar scenarios assume fills at later hourly quotes; they cannot establish executable profits because depth, queue priority, and contemporaneous fee schedules are missing. This is a research prototype with targeted regression tests, not a certified institutional trading system.
