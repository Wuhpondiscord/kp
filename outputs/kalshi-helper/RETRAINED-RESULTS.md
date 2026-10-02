# Updated model training and test results

Completed September 23, 2026 against the populated browser-stage research store. Both models were trained to completion; no test-driven parameter changes were made. Historical test dates were previously inspected, so these are exploratory results and neither model qualifies for automatic promotion.

| Run | Model log loss | Market log loss | Model Brier | Market Brier |
|---|---:|---:|---:|---:|
| Main Auto: 10,438 rows / 74 days | 0.208091 | 0.208974 | 0.067651 | 0.067400 |
| Station: 5,256 rows / 74 days | 0.197696 | 0.199492 | 0.064299 | 0.064409 |

Compare models only against their same-row baseline. The station universe differs from the earlier 5,274-row experiment. Its present inputs are now frozen, hash-verified and exactly reproducible; this does not resolve why the older reconstruction differed.

Main run `train-a3765d7dccfa`: 56 epochs, best checkpoint 36, validation-selected neural weight 0.75. Auto compared calibrators and updated tree candidates but still selected the neural checkpoint. Its results match the prior main run: slightly better log loss, worse Brier, and a market-relative loss interval crossing zero. This run is now selected in the stage model library; careful-mode validation restrictions remain.

Station run `station-2026-09-23T223805-001006-0000`: four-leaf Newton trees, 120 iterations, weight 1. The paired log-loss difference is -0.001796 with an exploratory seven-day-block interval [-0.002545, -0.000950]. At a 90-minute observation delay, log loss is 0.197786 and Brier 0.064351. This remains a separate research artifact, not a live price-only predictor.

| Station simulation, $1,000 starting cash | Entries | Net P/L |
|---|---:|---:|
| Same decision quote, standard fees, zero slippage | 7 | +$2.57 |
| Later quote, standard fees, zero slippage | 311 | -$14.09 |
| Later quote, standard fees, 2¢ slippage | 240 | -$19.22 |
| Later quote, double fees, 5¢ slippage | 152 | -$18.33 |

These assume fills without historical depth. The first and later-quote strategies select different cohorts. Seven favorable outcomes are not evidence of profitability. The later-quote strategy continues to lose in every tested cost scenario.

Validation performed: 133 software tests passed; station artifact and feature hashes verified; the importable station model loaded in a fresh Python process; both station and main test scores reproduced from frozen feature files and saved weights to within 1e-12. This was CLI training and offline evaluation, not a new browser-driven training run or a day-long live paper test.

Reports: [main training](reports/adaptive-retrained.json), [station training](reports/station-retrained.json), [station reproduction and edge diagnostics](reports/retrained-verification.json).

The recent changes improve robustness and reproducibility. They have not produced a measured improvement over the prior model on an identical dataset. A coherent event-level weather model and fresh prospective evaluation remain future work.
