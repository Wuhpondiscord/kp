# Response to the consolidated review

The supplied review identifies itself as a simulated instructor review prepared by Claude. Its numerical re-runs are external findings unless explicitly verified below. This response does not treat them as independent proof of edge or reproduce every table.

## Confirmed fixes

- Training environment capture now tolerates missing or broken distribution metadata. Missing versions are stored as null with a diagnostic; training continues. An integration regression test actually trains while metadata lookup raises `PackageNotFoundError`.
- The historical price-strategy cohort now uses nominal horizons 12, 18 and 24, including the 24.98-hour candles previously excluded. The prospective weather cohort still refers to literal remaining hours; those are a different quantity and are not silently relabeled.
- Polymarket real spread averages now use only two-sided books. Missing-side quote-interval width is a separate feature. Older feature rows are marked with missing true spread rather than reinterpreted as measured spreads.
- Hierarchy reports now state the mean-loss penalty convention and effective degrees of freedom. Penalties are selected on rolling training folds rather than assuming 0.1 is appropriate.

## Kalshi ladder and hierarchy re-run

The fixed candidate list used penalties 1e-5, 1e-4, 1e-3, 1e-2 and 0.1, three 30-day rolling validation windows inside the original training period, and whole-date 48-hour settlement purging. Both arms use identical rows, model class and tuning budget. Original split memberships are retained before restricting to complete ladders. No original test results were used.

The ladder features use same-timestamp quotes only: total midpoint mass, bracket distance from the implied median, quartile width and missingness indicators. Future quotes cannot complete a current ladder; labels are not used. Contract bounds come from stored historical market definitions, whose original publication/version history remains unverified.

All 1,468 events were considered; 1,467 had eligible complete same-time ladders. There were 8,089 complete event/timestamp groups; 3,213 rows were excluded. Added features improved the internal rolling score but worsened the later, previously inspected validation period:

| Later validation | Without ladder | With ladder | Market |
|---|---:|---:|---:|
| Log loss | 0.510884668 | 0.511053222 | 0.509001991 |
| Brier | 0.169075448 | 0.169164197 | 0.168717977 |
| Selected penalty | 0.0001 | 0.00001 | — |
| Effective degrees of freedom | 8.35 | 13.32 | — |

The paired ladder-minus-control log-loss interval is approximately [-0.000600, +0.000959]. This result does not support adding the ladder to production predictions. It also does not establish that city effects are absent: the tuned penalties are much weaker than the original choice, but the later market benchmark still wins. Effective degrees of freedom are the local penalized-logistic Hessian trace, not a count of independent events.

Artifacts: `reports/professor-followup-v2/`; runner: `work/professor_followup.py` in the workspace. The earlier v1 computation is retained as an audit artifact; v2 explicitly preserves the original full-data split/embargo membership and produces the same results.

## Statistical design

Added `reports/prospective-design-v2.json`, a **local design proposal, not a launched or externally registered experiment**. It specifies one frozen challenger, minimum relevant effects, date-clustered analysis, a fixed confirmation stop, and no interim success declarations. Sparse trading must produce an inconclusive result rather than extending the study until significance. The numbers are planning targets, not a power guarantee; the method needs statistical review before launch.

## Findings already addressed or still open

The current station trees already use public `.apply()` routing with external Newton leaf values; they no longer overwrite `tree_.value`. The engine separately reports the first forecast and actual filled-signal cohorts. Those portions of the review refer partly to older code.

Still open: intermediate-horizon generalization; full-market JSON storage/replay memory growth; five-minute observation integration; full-season prospective evidence; measured quote persistence/fill probability; exchange-specific fee verification; and explaining the original station-feature hash mismatch. None was marked fixed by these changes.

The fixed 1F observation-noise parameter remains an assumption. Pre-close sampled maxima and winning bracket intervals cannot identify pure observation error separately from remaining-day warming and interval censoring. A defensible estimate needs complete station-day observation histories paired with settlement values or a separately identified interval-censored model. This pass did not substitute a pre-close forecast residual for that noise estimate.

## Verification

174 Python tests passed, including missing-metadata training, nominal-24h membership, ladder label independence/future-quote exclusion and true-spread versus interval-width separation. The original negative price-model evidence remains intact. The corrected-cohort strategy search also rejected all eight candidates. No model or strategy was promoted and no exposure was increased.
