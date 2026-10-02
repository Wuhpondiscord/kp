# Post-priority retraining comparison

Fresh run `train-103cd0ff5f8a` completed using the updated pipeline. It selected the neural checkpoint at epoch 36, stopped after 56 epochs, and selected blend weight 0.75. Configuration remained adaptive selection, 32/16 hidden layers, seed 1729, maximum 150 epochs, patience 20.

The frozen feature hashes match the previous run `train-a3765d7dccfa`. The comparison uses identical chronological partitions: 30,806 training rows, 9,671 validation rows, 10,438 test rows and 832 purged rows. The test spans 74 dates, 296 city/day events and 1,776 contracts. These dates are reused research data, not a fresh holdout.

| Measure | Previous model | Fresh model | Market |
|---|---:|---:|---:|
| Test log loss (lower better) | 0.208090656 | 0.208090656 | 0.208974284 |
| Test Brier (lower better) | 0.067651398 | 0.067651398 | 0.067400294 |
| Fixed-$10 hypothetical P/L | +$22.98 | +$22.98 | — |

Maximum absolute prediction difference is exactly zero. Reloading the newly saved model reproduced its reported test log loss within 1e-12. All 145 Python tests passed.

The fixed-$10 comparison deliberately preserves the previous sizing experiment: $1,000 bankroll, 4-cent decision edge threshold, delayed quotes, 2-cent slippage, assumed fees, 100-contract quantity cap, and existing event/portfolio caps. Both models produce 34 entries, 565 contracts, $331.02 total deployed capital and +4.07 cents net per contract. Return on deployed capital is 6.94%, with an exploratory interval from -25.64% to +34.35%. This scenario does not implement the new live-engine daily/session entry stops and is not a complete simulation of that updated execution policy. Historical depth is unverified.

The forecast loss-difference interval remains [-0.0024143, +0.0005903], crossing zero. Brier remains worse than market. Promotion remains blocked; profitable deployment has not been established.

The result is expected: the recent additions chiefly changed evidence recording and risk controls. The coherent weather challenger was rejected by validation and was not incorporated into this live model. Newly recorded NWS snapshots do not yet constitute a settled training dataset. This run verifies reproducibility, not improved forecasting skill. A further skill improvement requires informative new training features/data or a challenger that beats the incumbent under validation, rather than simply rerunning the same deterministic training.

The fresh model is saved in the training/model library and marked as unproven. Full numeric comparison is in `reports/post-priority-comparison.json`; the pretraining configuration is in `reports/post-priority-protocol.json`. The reproducible runner is `work/train_post_priority.py` relative to the workspace root.
