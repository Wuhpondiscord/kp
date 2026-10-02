# Profitability test and execution improvements

September 23, 2026. Frozen models and hash-checked feature snapshots from the latest retraining were evaluated. Starting simulated balance is $1,000; scenarios use one contract per entry, estimated fees, shared cash and risk limits. There is no historical depth, and test dates were already inspected. These are exploratory hypothetical returns, not demonstrated executable profits.

## Results

| Model and policy | Entries | P/L |
|---|---:|---:|
| Station, original delayed selection, zero slippage | 311 | -$14.09 |
| Station, decision-qualified, immediate quote | 7 | +$2.57 |
| Station, decision-qualified, delayed quote | 5 | +$2.04 |
| Station, decision-qualified, delayed +2¢ slippage | 5 | +$1.94 |
| Station, decision-qualified, delayed +5¢ slippage | 5 | +$1.78 |
| Main neural, original delayed selection, zero slippage | 441 | -$11.67 |
| Main neural, decision-qualified, immediate quote | 49 | +$0.20 |
| Main neural, decision-qualified, delayed quote | 43 | +$1.95 |
| Main neural, decision-qualified, delayed +2¢ slippage | 34 | +$0.67 |
| Main neural, decision-qualified, delayed +5¢ slippage | 14 | -$0.57 |

All decision-qualified rows above use a fixed 4¢ net-edge threshold. Different rows can qualify under different cost assumptions. These comparisons are descriptive, not a search for the best test strategy. The station's five or seven outcomes cannot support an edge claim; its bootstrap interval can appear positive because sparse resamples do not represent the range of unseen future outcomes. Main-model intervals cross zero. No confidence interval here solves unavailable depth or reused holdouts.

## Validation-only strategy selection

A fixed grid of 2/4/6/8¢ thresholds was declared before this trading test, evaluated on validation dates using delayed execution with 2¢ slippage, and saved before scoring the test. Eligibility requires at least 30 trades across 30 event groups and a positive lower daily-P/L block-bootstrap bound adjusted for four comparisons. This minimum is a screening rule, not a statistical sufficiency guarantee. Model weights had already used validation labels; this is not fully nested selection.

**Neither model qualified.** Station validation produced only 0–6 entries across candidates. Main validation's 2¢ policy made 62 entries and lost $2.60; the higher-threshold candidates had too few entries. Therefore the validation-selected policy is **no trade** for both models, returning $0. That is abstention, not a claim of profitable forecasting. No live promotion gate was relaxed.

## What changed and why

The previous delayed scenario could create entries because a later price moved away from an old forecast even when no opportunity existed at the forecast time. A new `decision_qualified` option requires net edge at decision time, freezes the chosen side and rechecks edge at execution. The old scenario remains available for comparison.

Built-in live ML now applies the same conservative one-contract fee screen before creating a trading intent. The engine respects the intent's YES/NO side; an old forecast cannot silently become an opposite-side order after prices change. Newer rejected forecasts cancel older pending intents. Nonqualifying forecasts are still stored and scored, avoiding selection of only attractive-looking signals in the forecasting journal. External prediction imports retain their existing behavior unless they provide the new explicit flags.

The one-contract screen is conservative: it can skip a bulk order whose rounded per-contract fee would be lower. Live execution still uses actual configured size/depth limits, so its results are not identical to the one-contract historical scenario. Historical forecasts retain their original horizon/source assumptions; no execution-time reforecast was added. Later model updates, book latency and fee uncertainty still require prospective validation.

## Tests and reproducibility

139 Python tests and frontend checks passed during this pass. New regression cases cover prices creating an apparent edge only after the decision, fee-aware qualification, missing fees, fixed signal direction, saving rejected forecasts without creating orders, cancelling obsolete intents and keeping a no-trade policy independent of test labels. The final added sparse-event screen is also exercised by the policy tests.

`reports/profitability-protocol.json` records the strategy screen. `reports/station-policy-lock.json` and `main-policy-lock.json` record validation choices. `reports/profitability-test.json` contains all comparisons and daily block intervals. Code: `helper/profitability.py`, `evaluation.py`, `live.py`, `core.py`. Local reproduction driver: `work/test_profitability.py` from the workspace root, requiring the separately stored frozen datasets and artifacts. No day-long live paper test was run in this pass.

## Assessment

The change materially reduces stale-price-driven simulated trading, but the apparent improvement is mostly fewer trades. It does not add predictive information. Neither model currently has enough validated net edge to justify automatic trading.

The next likely modeling benefit remains an event-level remaining-day weather distribution trained on richer, timestamped forecast/observation pairs. It should be evaluated against both market prices and a price-only control with a fresh fixed protocol. Merely lowering thresholds, increasing stakes, or selecting the five historical winning trades would not be a credible improvement.
