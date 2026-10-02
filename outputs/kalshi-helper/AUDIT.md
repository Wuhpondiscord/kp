# BetCheck v0.4 audit

September 19, 2026. The app is more usable and the tested data/training paths work, but there is no demonstrated profitable edge. Real markets are used only with simulated money.

## Results

- **89 automated tests passed.** Includes isolated HTTP integration, transport failures, history import, model fitting and persistence, chronological splits, leakage controls, accounting, execution constraints, cancellation, restart handling and browser form defaults.
- **Seven live/data/training checks passed.** Actual Kalshi markets and order books, an archived settlement and its candles, NWS forecast timestamps, SQLite integrity, 30 sampled archived-response hashes, saved-model inference and a fresh neural fit. Details: `reports/system-audit.json`.
- Historical dataset: **8,758 contracts, 365 dates, 26,007 samples**. After date-boundary purging, the fresh training check used **15,478 training / 4,909 validation / 5,200 test samples**. Test covered 73 dates and 1,752 contracts.
- Fresh neural training completed 65 epochs and selected epoch 61 using validation loss. Test labels did not choose the checkpoint. The audit fit did not replace the active model. A subsequent browser-triggered full-year run (`train-66fc488490c3`) completed in 15.54 seconds with the same checkpoint and metrics, and saved the current experimental model.

| Final test metric (lower is better) | Neural/market blend | Market baseline |
|---|---:|---:|
| Log loss | 0.201150 | 0.200189 |
| Brier score | 0.064744 | 0.064600 |

The neural blend was slightly worse than the market baseline. Passing the software audit means the training process ran correctly; it does not mean the model won its forecasting comparison. The active model remains unproven and the careful-mode gate stays closed.

## Defects found and fixed

- Socket timeouts were not retried; bounded retries now include them.
- Null order-book sides could cause errors; they now become empty depth, never invented liquidity.
- Repeated markets across API pages were not deduplicated.
- Event-specific fee overrides were not consistently applied in discovery and series resolution. Unknown fees disable automatic fills.
- Future-dated weather cache entries could be accepted as fresh.
- Non-object JSON requests could disconnect instead of returning a useful error.
- Unsupported contract types/notional values and contradictory settlements could enter historical samples.
- A long history download blocked checking a market. Market checks now use a separate job channel.
- The default total-risk input violated its own browser step constraint, silently preventing practice from starting. Fixed and covered by a regression test over numeric defaults.
- Background polling unnecessarily replaced market cards, interrupting clicks. Unchanged cards now keep their DOM nodes.

## Easier everyday use

The home screen now presents a betting board with YES/NO prices, plain-language pass/watch/experimental labels, saved bets, filters and sorting. Clicking a market refreshes its prices and displays the model's explanation, fees, evidence and a play-stake payout preview. The wallet offers simple balance presets. Advanced model controls remain in the Model room.

Browser checks verified price refresh, shortlist filtering, fresh market explanations, changing a stake from $10 to $25, selecting a contract for practice and account navigation. A browser-started live practice session (`79188a3c1aa0`) checked 24 markets, recorded no errors and zero fills, and stopped cleanly. Browser-triggered neural training completed and displayed the exact full-year split. A 390px viewport check found no horizontal overflow, and the tested page logged no JavaScript errors. Evidence: `reports/browser-workflow-check.json`. No real-money orders are available or placed.

## Remaining limits

- This is a broad regression suite, not exhaustive testing or quant-firm certification. Parent-process line coverage is recorded in `reports/line-coverage`; it excludes the separate HTTP server process and is not branch coverage. Live-execution code still has substantial untested paths.
- Source checks sample actual responses at a point in time; they cannot certify every external record or continuous availability.
- Current learned features are market prices, spreads, city and time remaining. Independent weather, mentions, post-count and election features/models are still needed. NWS rain periods are context, not validated full-weekend probabilities.
- Historical dollar scenarios lack historical order-book depth and queue position. They cannot establish executable returns.
- No full 24-hour soak has been completed. Stopping a practice session leaves unsettled positions open and does not continue settlement monitoring.
- Repeatedly inspecting the same final test makes it unsuitable as fresh confirmation. Promotion blocks reused holdouts; further model research needs a new prospective test period.

## Repeat the checks

From this project folder:

```powershell
python -m unittest discover -s tests -v
python -m helper.audit
```

The audit requires network access and the local archive. It saves a new audit report without activating its freshly fitted model.
