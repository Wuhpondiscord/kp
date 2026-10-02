# Training architecture update — September 21, 2026

Version 0.6 changes **Train new model** to a market-correction neural network. The original classifier remains available from the Learning approach selector for comparisons. Auto-tune still runs its separate six-candidate comparison; it does not search this network's architecture.

## Why change the structure?

The old network learned YES probabilities from scratch, using market price, spread, horizon and city, then blended its predictions with the market. Much of its capacity went toward reproducing a strong existing forecast. The new model starts at market odds and learns an additive correction in log-odds space:

`prediction = sigmoid(logit(market_probability) + neural_correction(features))`

The output layer starts at zero, so the initial prediction matches the market. Hidden layers use tanh activations, trained with Adam and L2 regularization. Scaling is fitted only on training rows. Each calendar date receives equal total loss weight, matching the date-averaged validation metric. No oversampling of YES outcomes distorts the original probabilities.

This offset construction follows the general approach documented for [boosting from existing predictions](https://xgboost.readthedocs.io/en/stable/python/examples/boost_from_prediction.html), where a probability baseline is supplied on the logit scale. Our implementation is a small NumPy neural network, not XGBoost; the source supports the offset principle, not a claim that this particular network will win bets.

Validation selects the checkpoint and then the final shrinkage toward market odds, including zero contribution. The final test never selects parameters. The familiar epoch budget, patience, network size, learning rate and saved checkpoints still work. Historical classifier artifacts remain readable.

## Actual default-run results

Run `train-880997cc5db6`, seed 1729, 32/16 hidden units. Training stopped after 27 epochs and retained epoch 13. Validation selected full use of the market-correction output (which already includes market odds). Browser-triggered repeat `train-738dd22ecde6` completed in 9.35 seconds with identical results; see [browser verification](reports/residual-browser-check.json).

| Metric | Market baseline | New neural model |
|---|---:|---:|
| Validation log loss | 0.20378125 | 0.20255301 |
| Validation Brier | 0.06482275 | 0.06487188 |
| Final-test log loss | 0.20018944 | 0.19942174 |
| Final-test Brier | 0.06459957 | 0.06491357 |

The earlier fixed-half-blend neural model scored 0.2011496 test log loss. The new network improves on that result by about 0.86%; it improves on market log loss by about 0.38%. Brier remains worse. The paired seven-day-block 95% log-loss-difference interval is [-0.00227846, +0.00073197], including zero. These are reused test dates, so automatic promotion remains blocked. The previous beta model's reported test loss (0.19919105) was slightly lower; that comparison used a different fitting/validation procedure and does not establish that the neural model is the best available model.

The same 365-date real archive is used: 26,007 quotes, 8,758 contracts, **1,460 events**. Final test: 73 dates, 292 events, 1,752 contracts and 5,200 quotes. Multiple quotes and sibling contracts are correlated. The UI now reports independent event counts and descriptive city/horizon slices. Chicago, Miami and the 12-hour slice did not improve in log loss; Denver, NYC and the other horizon slices did. These slices are not separately validated strategies and are not used to decide which bets to permit.

## What this says about data

There is still no independent meteorological feature in this price-correction network. More copies of nearby quotes cannot supply the missing weather information. The separate GFS challenger already showed that poorly aligned/older forecasts can be worse than the current market.

The most useful next data work is:

1. Archive exact forecast initialization, release and retrieval times, and match each prediction cutoff to a genuinely available run.
2. Join official settlement-station observations and provider-specific maximum-temperature rules; retain revisions rather than overwriting them.
3. Add multiple forecasts, ensemble spread and recent forecast residuals, with missing-source masks and chronological comparisons against the price-only model.
4. Accumulate untouched future dates for one confirmatory evaluation after the approach is frozen. Additional independent seasons/events matter more than denser repeated quotes.

Mentions require a different dataset—timestamped speaker/event transcripts, duration, exact contract vocabulary and listing times—and should not share temperature labels. See [the earlier research audit](MODEL-UPGRADE.md) for the weather and linguistic-count research and original-notes alignment.

## Verification

103 Python tests passed. New checks cover initial market identity, numerical finite-difference validation of backpropagation through every layer, equal total date weighting, learning a known bias, and prediction-preserving artifact reloads. Existing tests cover timestamp leakage, future-test-label isolation, train-only scaling, cancellation and model-loading gates. JavaScript syntax checks passed.

[Saved results and segment metrics](reports/residual-training-check.json) contain the full numerical evidence. Run folders under `data/training/` contain exact split manifests, configuration, epoch CSV, checkpoint and report. No real-money orders are enabled.
