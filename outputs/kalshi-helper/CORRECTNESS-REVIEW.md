# Correctness review following professor feedback

Subsequent implementation-level follow-up: `MODEL-CODE-REVIEW-RESPONSE.md` documents the newly added categorical training architecture. The projection interface described below remains a distinct compatibility tool; neither change silently replaces the saved live artifacts.

No new architecture, blend-weight, ladder, hierarchy or profitability search was run in this pass. The retained model files and benchmark reports remain frozen. No demonstrated market edge is established.

## Station reconstruction mismatch: identified exactly

The original station dataset hash was `9a983c0cec82d918e7eb4f2319061cee70967c0469227638c2498908a9535330` (26,151 rows). The subsequent reconstruction hash was `ae8d26de15bd95a3665c338420675ae26fa6fb513cdbb91a3cd161307747005d` (26,133 rows).

The difference consists of six NYC temperature contracts for August 14, 2026, with three decision horizons each. Their primary rules name New York City and The Weather Company, while their secondary rules explicitly identify CLINYC/Central Park. The station-identity filter inspected only primary rules and rejected these 18 rows. An offline, attribution-only reconstruction restoring these rows reproduces the original hash exactly. No model scoring or fitting was needed.

All four raw observation hashes match the original report. Current reconstruction also matches every field of the later frozen feature file. This is a deterministic eligibility change, not evidence of stochastic feature reconstruction or mutated station readings. `supported()` now reads primary, secondary and legacy rule fields; it still requires the expected station, temperature maximum and settlement-source tokens. A regression test covers the secondary-rule case and wrong/missing stations. Historical reports remain unchanged, and their eligibility cohorts must not be silently mixed.

Evidence: `reports/correctness-review/station-reconstruction.json`, reproduced by `work/audit_station_reconstruction.py` against the local archived database. That database is not part of the earlier professor ZIP. The JSON records the pre-fix investigation; use its hashes and explicit counterfactual, not a claim that the current fixed code still excludes those contracts.

Post-fix verification: the unmodified production builder, with network requests blocked, reconstructs all 26,151 rows and exactly matches the original hash. See `reports/correctness-review/station-fixed-verification.json`.

## Event coherence: explicit, tested interface; deployment still separate

`SeedEnsemble.predict_event()` and `FusionBlend.predict_event()` now require a complete event partition at a single decision timestamp. Missing tails, gaps, overlaps, duplicated contracts, mixed event/station/timestamps, nonfinite probabilities and zero total mass are rejected. Labels are never read. Each parent is projected onto a categorical distribution by dividing its complete-event probabilities by their total. ConsensusBlend mixes the normalized parents, preserving the requested parent weights exactly.

This is an explicit inference projection, not a newly trained coherent remaining-temperature likelihood. It makes a complete event's probabilities sum to one; it does not prove calibration or better profits. The existing binary `predict()` remains unchanged so frozen historical results and incomplete, price-filtered datasets remain reproducible. The live single-contract path still uses that legacy interface: event-snapshot ingestion and a prospective evaluation of the new interface remain outstanding. Do not describe this as a fully deployed event-level model or silently attach the old benchmark to the new projection.

The physical temperature CDF already has a coherent form when all brackets share the same weather inputs. The independent market fusion corrections are where that property is lost. A future end-to-end categorical objective must use complete event partitions and consistent event features, without filtering away penny-price brackets before constructing the distribution. Eligibility filters belong after event prediction. That is a separate versioned model change requiring fresh evaluation, not another score on this validation window.

## Observation calibration default

The three physical research CLI commands now default to the frozen pre-training observation calibration in `reports/historical-forecast-replay/observation-calibration.json`. Missing calibration fails with an actionable error. Using the old assumption requires `--legacy-observation-assumption`, mutually exclusive with a supplied calibration file. Existing training-scope validation remains responsible for rejecting overlapping/future calibration targets. Low-level constructors and old pickles retain compatibility; this change specifically removes the silent CLI default.

The empirical discrepancy includes sampling, reporting and remaining-day transfer assumptions. It must not be described as identified pure sensor noise merely because late-day warming is usually small.

## Why probability metrics and profits diverge

The older station reconstruction shows approximately 74% of aggregate log-loss improvement in rows priced below 2%; 80% of its six-hour rows have zero bid and ask at most one cent. These are important evidence of score improvement concentrated near the tick floor, with limited immediate trade opportunities after costs. Quote-level profitability still requires side, price, fees and depth, so the concentration alone does not prove every such improvement untradeable.

For precision: moving a probability of the outcome that actually occurs from 99.7% to 99.9% changes log loss only by about 0.002. A confident wrong prediction incurs a large penalty, but many small tail-row improvements can dominate an aggregate. The concentration finding is empirical, not a claim that every small extreme-probability adjustment has a large individual loss effect.

The current three-model benchmark restricts validation to market probabilities 5–95%. The below-2% mechanism therefore cannot directly explain its gains or losses. For those models, the saved selected-bet decomposition, hard edge thresholds, small samples and delayed stale-signal execution remain the relevant measured explanations.

## Search accounting and selection rationale

`reports/correctness-review/search-inventory.json` inventories 40 top-level comparison slots in eight canonical reports, after excluding a byte-identical duplicate. These include baseline controls and repeated model configurations and are NOT 40 independent hypotheses. The inventory omits earlier adaptive grids, internal epoch/checkpoint looks, sizing grids, nested attempts and unrecorded UI training. An exact project-wide unique comparison count cannot be recovered from existing logs; stating an invented count or using 40 as a multiplicity correction would be misleading. Cohorts and date windows also differ across experiments.

The adaptive selector's primary log-loss criterion aligns with likelihood-based probability fitting. The Brier non-worsening condition guards against a deterioration in average squared probability error. It is an engineering multi-objective choice, not a significance test, a profitability criterion or uniquely optimal rule. There is no independent preregistration establishing why that priority was originally chosen; changing it now based on the same validation data would add selection bias. This rationale is now in the selector's docstring as well.

## Remaining limits before any new claim

- Seasonal breadth and small effective sample size are unchanged.
- A full audit cannot infer uniform embargo compliance merely from all documents saying chronological splitting is used.
- Fresh prospective, executable-price evidence is still required to establish net edge.
- Event-level live snapshot wiring and a future categorical training protocol remain unfinished; the new projection is a checked building block, not a replacement benchmark.
- Keep the existing professor ZIP as the immutable earlier snapshot. This addendum describes subsequent code changes; its earlier manifest must not be reused for changed source files.

## Verification

225 Python tests passed, including new partition, blend-weight, calibration-option and station-rule regressions. Frontend checks passed. All three retained artifacts still reproduce their 730 saved validation predictions to numerical precision; this is a compatibility check, not a new search or an independent test. No new profitability or model-training experiment was run.
