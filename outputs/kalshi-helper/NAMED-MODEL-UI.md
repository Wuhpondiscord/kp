# Named models in the app

Model Room and practice setup now offer WeatherSignal, MarketGuard, and ConsensusBlend. ConsensusBlend accepts a WeatherSignal percentage; MarketGuard automatically receives the remainder. These are probability weights, not bankroll allocations. The retained baseline artifacts are unchanged.

## Using the controls

1. Open Model Room, choose a family and saved weights, and adjust the mix for ConsensusBlend.
2. Choose **Use this model** to save the selection for board checks and copy it into practice setup. Practice can use its own selection; each session freezes that configuration.
3. Choose **Test on saved market history** to evaluate that exact selection against the existing validation data. The test uses the wallet balance from practice setup and reports forecast scores plus immediate and delayed quote scenarios.
4. For further training, expand **Train further from these weights**. The default freezes the weather encoder and trains the market head. Uncheck it to fine-tune all layers. Training starts from saved parameters and preserves the fitted normalization, at a fixed learning rate of 0.001. Epochs are limited to 1–100.
5. New checkpoints appear under **Saved weights** after successful training. They are not selected automatically. Choose one explicitly, test it, then use it if desired. A ConsensusBlend checkpoint retains both parents and allows its mix to change.

Older price-only training and data tools remain under an advanced disclosure. Their results and checkpoints are separate from these named families.

## What the new live path does

Experimental live practice uses the selected named model with an explicit GFS/ECMWF 06Z forecast run and station observations, through the same feature builders as the historical research. Settlement station/strike support is checked. Current observations use the existing 30-minute availability allowance; stale or missing inputs prevent a signal. The adapter refreshes cached observation data periodically, and the source cache can reuse observations for up to an hour.

Training covered the nominal 12-hour candle, approximately 13 hours before close. The live path allows an explicitly experimental 12–14-hour window, not a claim that every minute within it was independently validated. Other times collect prices without generating named-model entries. This restriction is shown in the UI, and all markets remain visible by default. Careful mode still does not trade an unproven named model; choose Experimental for model-driven paper signals.

At execution, the engine rechecks the selected model against the new book, using the named weather feature path. A model switch cannot silently reuse another family's pending signal. Real money, account keys, and order placement are not implemented.

## Training and evaluation limits

The built-in continued-training workflow uses the existing 87 training events across 22 dates. Evaluation uses 257 events across 67 already-inspected validation dates. It does not discover a new training corpus, select hyperparameters, use the reserved test, or establish profitability. Further training can worsen performance. Historical profit reports still assume fills without depth and retain stale predictions in delayed scenarios; they are distinct from the refreshed live engine.

Named inference requires the existing optional PyTorch environment and named artifact files under `reports/named-model-families/`. Historical testing/training additionally requires the forecast cache and replay files. New checkpoints live under the active data directory's `named-checkpoints/`, with parent metadata and SHA-256 verification. Baseline artifacts are never overwritten.

## Network recovery

The old server failed with Windows socket permission error 10013. A fresh Python process could reach Kalshi, and restarting the verified local server outside its restricted process environment restored access. The app now identifies this error immediately with a useful restart message instead of repeating it three times. **Check Kalshi connection** tests the exchange-status endpoint directly from the app process.

`Start Helper.cmd` now detects this workspace's existing `work/browser-stage-data` directory and starts the app on port 8766 with those data. Standalone copies without that directory retain the default local data/port behavior. No firewall protection was disabled and no API key was added.

## Verification

- 219 Python tests passed, including input/weight validation, frozen-backbone preservation, warm-start normalization, live coverage gating, and actionable socket-error handling.
- Frontend checks and JavaScript syntax checks passed.
- Browser verified the live Kalshi connection, market refresh, custom 60/40 selection, and saved-history evaluation reproducing log loss 0.489759 and the earlier -$21.14 delayed +2-cent scenario on a $1,000 wallet.
- A 50-epoch market-head continuation completed and produced checkpoint `0b5ba7760ac2`; the checkpoint was selectable and independently evaluated from the browser.
- A short live paper run used `ConsensusBlend:0.6:base`, produced supported Denver predictions, and explicitly withheld signals outside the experimental window. The verification session stopped cleanly with zero pending orders, one simulated fill, $993.37 cash and a $6.63 open position cost. That open paper position is not a realized loss or a profit result. A short smoke test cannot establish profitability or long-running reliability.

The original market-data failure is resolved in the checked app process. Keeping the server running and the computer awake remains necessary for live practice.
