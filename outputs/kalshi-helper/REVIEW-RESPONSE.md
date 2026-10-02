# Response to the external review

September 23, 2026. This pass checks the critique, fixes signal control and packaging, and adds descriptive diagnostics. No model was retuned or promoted.

September 25 follow-up: the feature-hash mismatch described below has now been identified exactly as an 18-row settlement-rule eligibility difference. See `CORRECTNESS-REVIEW.md` and `reports/correctness-review/station-reconstruction.json`. The historical account and original numerical results below are preserved.

## Findings and measured follow-up

The review correctly identified stale documentation, an incomplete review ZIP, UI-label-dependent trading control, discrete training horizons hidden behind a continuous coverage range, and a material difference between forecast scoring and delayed trading simulation.

The original station feature dataset was hashed but not frozen as rows. Rebuilding features from the current cache gives a different hash: original `9a983c0cec82d918e7eb4f2319061cee70967c0469227638c2498908a9535330`, reconstructed `ae8d26de15bd95a3665c338420675ae26fa6fb513cdbb91a3cd161307747005d`. The precise source of the difference has not been established. The audit below therefore uses **frozen model weights on reconstructed data**, not an exact replication of the original experiment. The original report remains unchanged. Future station runs now save both the 30-minute and 90-minute feature datasets, rather than just hashes.

On the reconstruction, model log loss is 0.197696 versus market 0.199492. Additive contributions use the original daily denominators so disjoint buckets sum to the headline difference:

| Segment | Share of total improvement / other finding |
|---|---|
| 6-hour rows | About 61% of total log-loss gain; 80.0% have bid 0 and ask ≤1¢ |
| 12-hour rows | About 39% of total gain; 40.0% have bid 0 and ask ≤1¢ |
| 24-hour rows | No gain; predictions equal the market |
| Market probability below 2% | About 74% of total gain |
| Market probability 30–70% | About 2% of total gain |

Thus the tick-floor hypothesis is supported on this reconstruction. It is more precise to say that most gain is concentrated in low-price rows than to infer all gains are untradeable from a small sample.

## The execution question

The original scenario freezes the model's probability at decision time, then checks entry against a later quote. It tests a delayed, stale-signal strategy and can select a very different cohort from a decision-time strategy. It should not be treated as a clean estimate of the model's value at its original price. However, a market need not move away for every entry: sufficient original edge can survive an unchanged or moderately worse quote.

Two explicit modes now exist in `quote_scenario`: `later_quote` (default, preserving behavior) and `decision_quote` (same-quote counterfactual). The latter assumes immediate execution without depth or latency; it is optimistic, **not a mathematical upper bound** on every later-price outcome.

| Frozen model, reconstructed data, standard fees, zero slippage | Entries | P/L |
|---|---:|---:|
| Decision-quote counterfactual | 7 | +$2.57 |
| Later-quote stale-signal scenario | 311 | -$14.09 |

These are different selected cohorts. The original saved later-quote result was 312 entries/-$13.40; reconstruction is not exact. The new report also fixes cohorts by claimed decision-time net edge, then compares same-quote and later-quote per-row P/L without reselecting at execution. Only seven rows across seven events clear the 4¢ threshold at decision time. Neither those seven outcomes nor the aggregate delayed scenario establishes profitability or its impossibility.

`edge-audit.json` includes horizon × price buckets, claimed net-edge buckets, fixed-cohort outcomes and a complete-bracket diagnostic. For 786 complete event/timestamp partitions, normalized categorical log loss is 0.739980 versus normalized market 0.748319. 156 incomplete/nonpartition groups are excluded. Normalization is a diagnostic transformation, not the model's raw prediction. Mean raw probability-mass error is 0.028569, supporting the case for a coherent event-level model.

## Statistical qualifications

An aggregate Brier improvement cannot generally be converted to a typical probability disagreement by taking its square root. Writing model `q`, market `p` and outcome `y`, the exact identity is:

`E[(p-y)^2 - (q-y)^2] = E[(p-q)^2] + 2 E[(p-q)(q-y)]`.

The cross term vanishes under stronger conditional assumptions, not simply because a forecaster looks calibrated in marginal bins. Even a valid average disagreement estimate would not rule out profitable selected subsets. The relevant question remains whether a predeclared selection rule has positive net expected value at executable prices. The present evidence does not show that.

Power estimates based on independent binary trades are illustrative. Dependence, selection, payoff variation and multiple testing change the needed sample. A fixed event count is not a universal guarantee of adequate power.

## Fixes made

- `PredictionResult` carries probability, reason, signal eligibility and model ID. Live trading and explanations use the explicit signal flag, not label text. The old two-value `predict()` interface remains for read-only compatibility. Tests deliberately rename labels and put an “Experimental” string on a non-signal.
- Added explicit execution-scenario modes and tests that future quotes cannot affect the same-quote result.
- Added descriptive price/horizon, edge-cohort and complete-event diagnostics with no model/threshold tuning.
- Future station experiments freeze their actual feature rows. CLI training now uses the importable model class, avoiding new pickles bound to `__main__`.
- Updated README and review-package contents to include front-end assets, front-end checks, referenced documents, launcher and examples. The package is tested after extraction. No full research database is implied.

## Confirmed limitations still open

1. **Seasonal coverage:** the main training block lacks a full summer; validation/test are seasonally different. A new multi-season dataset and fixed rolling-origin protocol are needed before claiming robustness.
2. **Selection noise:** the candidate grid shares validation dates for checkpoint, family and blend selection. Close candidate scores and the Brier constraint deserve nested temporal evaluation, not retrospective removal of the gate to favor the best test result.
3. **Horizon/source shift:** main training samples six discrete horizons. The current coverage interval allows interpolation at untested horizons, and live best-book quotes differ from historical candle closes. These are not independently validated deployment conditions. This pass documents the issue; it does not claim to solve it by weakening gates or rounding time.
4. **Promotion lock:** overlap with a previously inspected holdout blocks promotion regardless of apparent gains. Careful Auto will not trade with that artifact; experimental paper mode is separate. Roughly one full test-window length of fresh dates would be needed under an unchanged rolling-window design, but the exact requirement depends on the data/split. The main neural interval also crosses zero, so removing the overlap lock alone would not make that result pass.
5. **Engine scoring:** first-prediction-per-market forecast metrics and latest-signal trading represent different estimands. They still require separate labeling/aligned scored cohorts.
6. **Execution:** the live engine still fills later, usually on the next polling cycle. No execution-time reforecast or passive-order queue model was added here.
7. **Storage and internals:** full market polling records and in-memory replay remain scalability concerns. Newton trees still mutate sklearn leaf values, an implementation dependency requiring version tests.
8. **Environment:** installed tzdata 2025.1 is below the declared minimum. The mismatch is recorded, not silently repaired or erased from historical provenance.

## Reproducing and sharing

The rebuilt professor ZIP is a source/test/evidence package. From its extracted root, run `python -m unittest discover -s tests -q` and `node tests/frontend_checks.cjs` with compatible dependencies. It now includes files required by those checks.

Historical report fields pointing into `../../work/browser-stage-data` record the original workstation layout; they do **not** identify included files. Existing signed/hash-associated reports and the frozen prospective protocol are preserved as evidence. For a new station run use `python -m helper.station_model --data PATH_TO_POPULATED_RESEARCH_DATA --report reports/new-station-model.json`. The package alone cannot supply that dataset. The prospective protocol remains a proposal awaiting new data, not a running collector or successful validation.
