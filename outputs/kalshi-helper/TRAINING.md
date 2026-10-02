# Training and market explanations

Open **Model room → Train a neural model**. Version 0.6 defaults to a market-correction network; the original classifier remains selectable for comparison. See [RESIDUAL-TRAINING.md](RESIDUAL-TRAINING.md) for the current method and measured results. This button always initializes and trains fresh weights; it does not return a cached experiment. **Auto-tune · choose best model** runs the separate expanding-window comparison and may reuse an identical saved evaluation.

## Training workflow

The included real archive is partitioned by calendar date into nominal 60% training, 20% validation and 20% final test. The current dataset yields 4,002 training samples, 1,215 validation samples and 1,432 test samples. Another 427 samples are purged around the boundaries. Dates stay together, and all labels in a training partition must have settled at least 48 hours before the next partition's first prediction.

The neural classifier trains with Adam in minibatches. Scaling is fitted only to training data. Each epoch updates weights and reports training/validation daily log loss and Brier score. Validation log loss chooses the best saved checkpoint. Early stopping monitors validation improvement. The final test is evaluated after checkpoint selection; it cannot choose an epoch. Reusing previously inspected test dates blocks automatic promotion.

Controls include epoch budget, 16/8, 32/16 or 64/32 hidden layers, batch size, learning rate, L2 regularization, patience and seed. Runs are reproducible for a fixed dataset, software stack and configuration. Loss curves display the deployed fixed blend of 50% neural probability and 50% market midpoint; raw optimizer loss is logged separately.

Each run writes `data/training/train-<id>/config.json`, `split-manifest.json`, `best.pkl`, `results.csv` and `report.json`. The active model bundle and its integrity hash are stored under `data/models/`. Cancellation keeps the previously active model. A restarted server marks interrupted training clearly rather than presenting it as complete.

The model inputs are price-derived probability, spread, time remaining and city identity. This is actual supervised neural training, but it does not yet learn from independent weather forecast vintages, transcripts or other category data. A bigger network or lower training loss is not evidence of better final-test performance.

## Inspect a market

Click a market title or **Prediction & why** on the browse board. The app refreshes that individual contract and shows:

- Model YES probability versus market midpoint.
- YES and NO expected edge per $1 contract after conservative one-contract fees.
- A **Pass**, **Watch**, experimental direction, or validated paper-candidate decision.
- Validation/test log loss, Brier score and sample coverage for the actual model used.
- Hypothetical input sensitivities: change market price, spread or horizon and observe the probability response. These are model responses, not causal explanations.
- The settlement rules, a link to the Kalshi event, and a button to select the contract for a paper session.

Unsupported categories do not receive invented model probabilities. Missing fees, stale quotes, closed markets or a failed validation gate prevent a validated paper-entry recommendation. Sensitivity results and aggregate metrics do not establish confidence for an individual contract, and quote-based edge is not proof of executable profit.
