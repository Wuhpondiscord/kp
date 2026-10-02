# Polymarket information and hierarchical calibration

Polymarket is now a read-only prospective data source. There are no wallet, account, signing or trading endpoints. All trade simulations remain Kalshi-only.

The collector queries Gamma by a deterministic supported city/date event slug, then retrieves each YES token's CLOB book. It parses outcome identifiers rather than assuming a token order; obtains best bid/ask independently of array order; retains metadata, rules, source payloads, receipt timestamps and hashes; and reports empty/unsupported books. Only explicit Fahrenheit brackets are parsed. Missing events and API failures are recorded without disabling Kalshi collection.

Integration deliberately labels every discovered event **related only**, not an exact contract match. Today's NYC Polymarket rules refer to LaGuardia hourly NOAA readings with a Weather Underground fallback, whereas the Kalshi station model uses Central Park. Different stations, observation definitions, fallback behavior and revision cutoffs prevent probability substitution. The source remains regional information even when a station happens to match; exact equivalence requires a separate rules audit.

Complete bracket books yield a normalized implied median interval and mean spread. This is a noisy market-derived feature, not measured temperature or an executable Kalshi probability. Incomplete books, incoherent total probability, stale snapshots, future receipts and mismatched city/dates withhold the feature. Tail medians are recorded but excluded from the initial model ablation. All quotes are collected sequentially, not atomically.

## With versus without

`PolyMaximum` adds the regional median's difference from the NWS forecast and the Polymarket spread to the remaining-day weather model. The paired comparison trains both arms on identical Kalshi examples, date partitions and outcomes. Both select blends on validation only, including zero weather contribution. All contracts across both feeds stay grouped by Kalshi event date. No additional independent outcome count is attributed to Polymarket.

The comparison was executed and returned **insufficient paired data: 0 eligible settled rows**. The existing archive predates this collector. Today's standalone NYC API check returned six usable two-sided books and five empty-book errors, so its regional feature was withheld. There is no valid with/without profitability or accuracy result yet. Synthetic fixtures verify parsing and masking only; they are not reported as performance evidence. Historical Gamma metadata cannot recreate point-in-time books or historical rule revisions, so no fabricated backfill was used.

Use **Research → Improve my weather strategy → Compare with Polymarket data** after accumulating recorded and settled examples, or:

```powershell
python -m helper --data ../../work/browser-stage-data compare-polymarket --output reports/my-poly-comparison
```

## Hierarchy experiment

Implemented and fitted a two-level statistical model: shared probability/horizon calibration plus city-specific intercept and probability-slope deviations. City deviations receive stronger regularization, drawing weakly supported city estimates toward the common model. Unknown cities use the shared component. This is penalized partial pooling, not a fully Bayesian hierarchy and not a new deep-learning framework.

Fixed experiment on the same previously inspected 70-day Kalshi validation cohort:

| Model | Log loss ↓ | Brier ↓ |
|---|---:|---:|
| Shared calibration | 0.512770705 | 0.170349992 |
| Shared + city deviations | 0.512781006 | 0.170322927 |
| Market | 0.512304623 | 0.170244165 |

The hierarchy slightly improves Brier but slightly worsens log loss; both trail market. Neither generated entries in the fixed 2% edge, 2-cent slippage, 1% bankroll scenario. There is no evidence to promote it. The final test was not evaluated, and the live model was not replaced.

A hierarchy is still a sensible research structure: separate category models; shared weather behavior with station-specific adjustments; one coherent event distribution converted to brackets; event-level risk aggregation. The result here argues for testing that structure with informative weather/cross-market inputs rather than expecting city labels alone to create an edge.

## Verification and artifacts

166 Python tests passed following implementation. New tests cover order-endpoint rejection, shuffled books, bracket units, future timestamps, wrong dates, incomplete data, paired-data gating and unseen-city fallback. Source smoke evidence, the paired-data report and the hierarchy protocol/results are under `reports/polymarket-source-smoke.json`, `reports/poly-ablation-v1/`, and `reports/hierarchy-v1/`. Research runners and artifacts do not automatically replace active weights.

API references: [market discovery](https://docs.polymarket.com/market-data/discover-markets) and [prices/order books](https://docs.polymarket.com/market-data/prices-order-books).
