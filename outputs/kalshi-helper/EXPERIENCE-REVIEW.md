# End-to-end experience review — September 22, 2026

Version 0.7 adds a saved-model selector, recommended-settings training button, automatic refresh of stale markets on page load, and plain-language filters: **Worth a look**, **Pass for now**, and **Not enough evidence**. Advanced training, history download and weather experiments remain under an expandable section. The model library states which model is in use; training and Auto-tune select their result automatically, and users can switch back to a saved model.

## Browser test performed

An existing user paper session was running on port 8765. To avoid disturbing it, I used a separate instance on **8766**, with a consistent SQLite snapshot of the real archive and copies of its saved model files. These are real archived outcomes and live Kalshi prices, not synthetic model-training data.

1. Opened the page and let it refresh markets automatically. A session network restriction initially blocked requests; after restoring access, 48 real contracts loaded.
2. Used **Model room → Train with recommended settings**. Run `train-c6e16af713e3` completed in **8.36 seconds**, training 27 epochs and retaining epoch 13.
3. Selected beta calibration in the model library and verified the “In use” label. Selected the newly trained network again and verified it changed back. Model reports follow the selected model; training history remains the latest training run.
4. Started $1,000 of experimental paper practice through the browser, using automatic market discovery. The new session appeared correctly.
5. Inspected two polling cycles, then stopped the session through the browser. Run `86c5b1c548b7` recorded **24 predictions, 24 pending signals, 12 skips on later books and 12 cancellations at shutdown**. Every prediction referenced the newly trained model. There were **no source errors, fills or realized profit**. Twelve of the 24 checked markets had supported ML estimates; the others were outside the trained window.
6. Used the **Worth a look** filter. No contracts qualified. Its empty state now explains that the helper is passing, instead of implying a broken filter.

This approximately two-minute test checks the real data → model → decision workflow. It does not establish full-day uptime, settlement behavior or profitability. Those remain separate requirements.

## Issues found and fixed during testing

- A tall sticky practice sidebar made Start unreachable in a shorter browser window. It now scrolls normally with the page; the exact previously failing click succeeded.
- Concurrent refreshes could leave the selected-model label stale or show an older practice session. Refresh requests are now queued and awaited; a new session keeps its selection while older responses finish.
- The final console check caught an audit-panel insertion error after training controls moved under Advanced. It was fixed and a frontend regression check now exercises that nested layout; the visible training flow was rerun afterward.
- Model selection had no explicit UI. Saved models now appear with dates and labels. Activation verifies the recorded artifact hash and preserves the original validation gate. Switching models is blocked during an active session or research job.
- New paper sessions snapshot their starting model so subsequent training does not silently change their predictions.
- A second browser session (`0c93be70cff6`) verified the saved model snapshot matched its prediction records and the model-switch button was disabled while running. It was then stopped; [pinned-model evidence](reports/pinned-model-browser-check.json).
- Late training cancellation, inconsistent input probabilities/spreads and duplicate Windows server startup now have regression coverage. Details are in [BROWSER-REGRESSION.md](BROWSER-REGRESSION.md).
- Very small probabilities display <1% rather than 0%, and negative fractional-cent edges remain visible.

## Ratings

These are my subjective ratings of the tested build, not independent certification.

| Area | Rating | Assessment |
|---|---:|---|
| Open → browse → understand a bet | 7/10 | Current markets load without tickers; opportunity/pass labels and filters help. Unsupported categories and time windows still require explanation. |
| Train → select → practice | 8/10 | One-click defaults, explicit saved-model selection, clear completion and a working paper flow. Advanced controls remain optional. |
| UI clarity | 7/10 | Main decisions are simpler; model details remain dense and could benefit from more progressive disclosure. |
| Software/data checks | 8/10 | 107 Python tests, real-archive validation, browser workflows, plus frontend formatter and concurrency checks passed. No full-day soak was completed. |
| Predictive performance | 4/10 | The residual model slightly improves log loss versus market odds, but Brier is worse and the uncertainty interval includes no improvement. |
| Evidence of profitable betting | 1/10 | No demonstrated executable edge. The smoke test placed zero trades; historical fill scenarios remain assumptions. |

Model scores are unchanged by these UI/reliability changes: test log loss **0.19942** versus market **0.20019**; Brier **0.06491** versus **0.06460**. Test dates have been reused, so automatic promotion stays blocked. “Worth a look” means an experimental price discrepancy, not a guaranteed good bet.

## Using the result

The verified instance is at **http://127.0.0.1:8766/** and uses isolated test data. The original session on 8765 was not interrupted or switched. The project source is updated; after that original session finishes, restart the normal helper to load version 0.7 against the original data.

Evidence: [browser-model-selection.json](reports/browser-model-selection.json). Research and architecture: [RESIDUAL-TRAINING.md](RESIDUAL-TRAINING.md).
