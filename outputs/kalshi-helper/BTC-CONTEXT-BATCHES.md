# Two further BTC model batches: six candidates, mixed evidence

## Result and decision

Six new candidates were evaluated in two sequential batches with protocols frozen before each batch's scoring. One improves Brier slightly over the prior best candidate, but worsens log loss and loses in the primary trading scenario. Another earns +$1.20 on four primary-scenario trades but has worse forecast metrics. Neither is an overall improvement or grounds for promotion.

Keep the original fixed-anchor research candidate unchanged. Preserve the new methods, artifacts, all negative results and the mixed-result directional candidate for review. The July 21–31 holdout remains sealed. No new candidate was deployed.

## Batch 1: additional context with training-only regularization

The existing flow model omitted ten Coinbase features already timestamped and validated in the dataset: target distance in volatility units, normalized 1/5/15/60-minute returns, short-term volatility ratio, relative volume, current bar range, and time-of-day sine/cosine.

Three candidates use: those ten spot features alone; spot plus the four existing flow/basis features; and that combination plus three bounded interactions (distance × volatility ratio, one-minute return × five-minute imbalance, five-minute return × fifteen-minute imbalance). Inputs and interactions are standardized using training rows only. All candidates retain the fixed market-logit offset with no learned intercept or market slope.

Penalty values .1, 1 and 10 are compared inside each training window. The last max(2, floor(training days/4)) days form an inner validation slice with a two-hour embargo; earlier training outcomes must already have settled. Select the lowest inner log loss, breaking exact ties toward greater regularization, then refit the chosen model on the full training window. Outer evaluation labels and trading results do not select the penalty. This is nine configurations per fold, reported transparently, not a single prespecified parameter claim.

All three candidates selected .1, 10, 10 and .1 across the four windows. The coarse selections changed substantially with the inner period; nested validation prevents direct outer-label selection but does not guarantee a stable choice or useful model.

## Batch 2: physical zero and a directional constraint

After batch 1 failed to improve overall scores, a second batch tests a different constraint rather than adding more parameters. Three fixed-penalty (.1) models use:

- Signed flow: five/fifteen-minute taker imbalance and five-minute basis change. Excludes the absolute cross-exchange price gap.
- Price: distance in volatility units, normalized returns at four horizons, plus distance and one-minute return multiplied by (volatility ratio − 1).
- Joint: both groups above.

Features are scaled by their training root-mean-square without mean subtraction. Thus zero directional inputs produce no market correction. Complementing the market probability and reversing all directional inputs complements the output probability. This is a structural prior, not a claim that real markets obey perfect sign symmetry. Non-directional context can scale directional effects through the two price interactions, but cannot independently shift the direction.

## Same-cohort results

527 previously inspected contracts across 23 evaluation days, same four expanding splits. Smaller Brier/log loss is better. Trading uses the unchanged $1,000 bankroll, 1% sizing/shared caps, .04 entry threshold, fees and .02 primary slippage.

| Candidate | Brier | Log loss | Primary P&L | Entries |
|---|---:|---:|---:|---:|
| Raw market | .121078 | .377750 | $0 | 0 |
| Previous fixed anchor | .120761 | **.377420** | $0 | 0 |
| Batch 1: spot | .121385 | .379135 | −$9.82 | 1 |
| Batch 1: combined | .121261 | .379337 | −$6.91 | 6 |
| Batch 1: interactions | .121587 | .380191 | +$1.20 | 4 |
| Batch 2: directional flow | .120878 | .378105 | $0 | 0 |
| Batch 2: directional price | .121330 | .378870 | $0 | 0 |
| Batch 2: directional joint | **.120680** | .377929 | −$9.82 | 1 |

The directional joint model reduces Brier by about 0.067% relative to the previous fixed anchor. Its paired Brier difference is −.000081, with a day-cluster bootstrap 95% interval [−.001857, +.001764]. The interval crosses zero by a wide margin; it is descriptive and unadjusted for repeated experimentation. It cannot substantiate improvement. The joint model's log loss is worse than both fixed anchor and the raw market.

The interaction model's +$1.20 comes from four trades. Its worse forecast scores and small trade count do not justify calling it profitable. Sparse-trade profit intervals remain suppressed.

## What worked and what did not

**Additional information already encoded as features was not enough.** The spot-only and broadly combined candidates did not beat the baseline in aggregate. More inputs introduced more fitted directions without establishing incremental information beyond the market. The interaction candidate also underperformed on both scoring rules.

**Directional structure produced a limited metric gain.** The joint directional model improved Brier in some periods, but not consistently enough across both metrics. It still worsened both metrics in the first evaluation period relative to the single fixed anchor. This suggests the constraint is worth retaining as a research hypothesis, not promoting as a winner.

**Training-only selection is a workflow improvement, not proof of predictive gain.** The new inner validation and embargo isolate parameter selection from outer evaluation outcomes. They do not erase the fact that model designs have repeatedly been revised after inspecting these dates.

**Costs and selection remain decisive.** The directional joint model earns +$5.48 / +$8.90 / +$17.83 under 0/0.5/1-cent slippage scenarios, but loses $9.82 under the unchanged 2-cent primary case. Those scenarios select different trades. Choosing the best cost setting would invent favorable execution assumptions. The raw ledgers and all scenarios are retained in each report.

## Limits on further conclusions

These two batches provide partial metric improvements and additional negative results, not a verified new edge. Continuing to search the same dates until every headline looks positive would make that apparent success less trustworthy. The results do not justify risk increases, a lowered threshold, deployment, or opening the reserved holdout solely to look for a better score. Further substantive validation needs genuinely new information or evaluation evidence.

The same assumptions remain: completed-bar availability plus five seconds, no measured historical arrivals or order-book depth, hypothetical fills, assumed historical fees, finalized opening-target metadata, hourly contract sampling and limited independent event days.

## Code, artifacts and reproduction

From `outputs/kalshi-helper`:

```powershell
python -m helper.btc_context_batch
python -m helper.btc_directional_batch
python -m unittest discover -s tests
```

Both commands use bundled development data and saved baseline artifacts without network access. Protocols and source hashes are frozen before scoring, models are saved before each outer evaluation, and repeated evaluations are marked as reruns.

- `helper/btc_context_batch.py`: nested chronological regularization, context models, interactions, serialization and scoring.
- `helper/btc_directional_batch.py`: physically centered directional features and reversal-consistent models.
- `tests/test_btc_context_batches.py`: nested settlement purge/embargo, recorded selection consistency, model round trips, inference immutability, directional complement/neutral behavior and exclusion of absolute basis.
- `reports/btc-context-batch-v1/`: protocol, per-fold inner selection records, models, report, calibration and trade ledgers.
- `reports/btc-directional-batch-v1/`: protocol, per-fold models, report, calibration and trade ledgers.

Existing models and the original held-out partition are unchanged. No hosted UI or production settings were modified.
