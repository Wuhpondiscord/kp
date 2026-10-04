# BTC forecast and trading experiment — October 4, 2026

Implemented and evaluated two separate components without a Kalshi API key:

* **BTCRegime forecast:** nonlinear, shallow histogram gradient-boosted trees on
  the existing ten Coinbase inputs. Fixed 80 iterations, four leaves per tree,
  minimum 60 examples per leaf, L2=10 and learning rate 0.05. Automatic random
  validation/early stopping is disabled. It predicts the probability of the Kalshi
  YES outcome, not an unconditional dollar-price forecast.
* **Learned trade filter:** standardized ridge regression predicting realized
  per-contract net return from win probability, estimated edge after costs, spread,
  absolute target distance and volatility ratio. Both YES and NO examples come
  from strictly forward, out-of-fold forecasts inside the training period. It
  permits a trade only when its predicted return is positive and the original
  four-cent edge requirement passes. Its estimates are not confidence bounds.

The third policy adds the existing quarter-Kelly sizing and daily/session loss
stops to the filter. This is a separate comparison, not a newly validated risk rule.
No original test rows are loaded, no parameters were selected from these results,
and no weather models or deployed app behavior changed. All dates below are reused
development data; none of these results is a new independent holdout finding.

## Forecast results on September 19–24 validation

| Forecast | Log loss (lower better) | Brier (lower better) |
|---|---:|---:|
| Market midpoint | 0.413306 | 0.133611 |
| Existing BTCSignal | 0.433219 | 0.140467 |
| Existing BTCMarketGuard | 0.411535 | 0.133377 |
| Price-only control | 0.411718 | 0.133211 |
| New BTCRegime | 0.436612 | 0.139694 |

The new forecast improves Brier by about 0.55% relative to BTCSignal but worsens
log loss by about 0.78%. Its earlier forward-fold log losses are 0.460616,
0.458882 and 0.472317, each worse than the corresponding market baseline. More
flexible feature interactions have not solved the forecasting gap. Do not replace
the existing forecast with this candidate on the basis of its one better metric.

## Trading results on the same validation period

All scenarios start at $1,000, recompute probabilities with next-minute spot and
market inputs, assume 2-cent slippage and the existing quadratic fee, and retain
the existing bankroll/exposure caps. Fills are hypothetical: depth, queue position
and actual post-publication quote availability are unknown. The underlying forecast
was trained at the earlier decision horizon, a remaining limitation.

| Policy | Entries | Net P&L | Return on deployed capital | Cost-equity drawdown |
|---|---:|---:|---:|---:|
| BTCMarketGuard + existing fixed rule | 2 | -$3.85 | -20.42% | $9.30 |
| BTCRegime + existing fixed rule | 118 | -$234.51 | -25.67% | $281.07 |
| BTCRegime + learned filter | 0 | $0 | Undefined | $0 |
| Filter + quarter-Kelly and stops | 0 | $0 | Undefined | $0 |

The unfiltered BTCRegime policy deployed $913.51 cumulatively and lost 3.20 cents
per contract. The filter removed these trades, but a policy that never trades is
not profitable and has no realized ROI. For context, abstaining unconditionally
also produces $0. There is insufficient evidence to show the learned filter beats
that simple baseline. The risk-policy variant had no opportunities to demonstrate
an incremental benefit.

Earlier forward folds, each resetting the hypothetical bankroll to $1,000:

| Development fold starts | Unfiltered BTCRegime P&L | Filter P&L | Filter trades |
|---|---:|---:|---:|
| September 7 | -$138.57 | +$3.20 | 1 |
| September 11 | -$399.06 | $0 | 0 |
| September 15 | -$190.29 | $0 | 0 |

Do not add these independently reset accounts together as a continuous portfolio.
One historical winning filtered trade is far too little evidence of edge.

## How training leakage is controlled

Each outer fold fits the forecaster on earlier settled events only. Inside that
training partition, the first 40% of dates warm up the forecaster and three expanding
blocks generate out-of-fold predictions, separated by a two-hour embargo. Only
those predictions and their eventual outcomes train the trade filter. It never
trains on in-sample forecast accuracy or outer validation outcomes. Both sides of
one contract are dependent examples, not two independent events.

The filter receives refreshed execution-time quotes and features, and fees/slippage
enter its realized-return targets. It cannot alter the probabilities reported in
forecast metrics; rejected trades receive the market midpoint only in the execution
copy, which causes the existing edge rule to abstain. Input-length and finite-value
checks prevent silent truncation of predictions.

The machine-readable report contains forecast calibration bins, complete ledgers,
opportunity counts, bootstrap intervals, gate coefficients and inner-fold timestamp
audits. Bootstrap intervals are exploratory day-block ratios, not full bankroll
path resimulations. Tiny numbers of active trading days cannot establish reliability.

## Decision

Keep the forecast as a research challenger, not an upgrade. Keep the separately
trained filtering mechanism for further evaluation, while explicitly comparing it
to always abstaining. Do not weaken the filter to manufacture activity. Better
data alignment and additional independent dates remain more promising than adding
complexity to these same Coinbase features. BRTI remains optional; public-source
research continues without credentials.

Reproduce from outputs/kalshi-helper:

```powershell
python -m helper.btc_regime
python -m unittest discover -s tests
```

Artifacts: `reports/btc-regime-gate-v1/{protocol,report,manifest}.json`. Fitted models
are rebuilt by the deterministic experiment rather than installed in the app.

Implementation references: [histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
and [ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
