# Attempt to improve validation performance

Trained two focused candidates: a regularized context calibrator and a smaller 16/8 neural network. Training targets the predefined 12–24-hour, 5–95-cent cohort. The original training partition was split again: 4,373 fitting rows and 1,170 inner-validation rows, with a 48-hour settlement embargo. Inner validation selected the context calibrator at weight 1. The neural run stopped after 24 epochs. Neither candidate was selected using the later 1,682-row strategy-validation period.

| Later strategy validation | Previous model | Focused challenger | Market |
|---|---:|---:|---:|
| Log loss | 0.513878 | 0.513423 | 0.512305 |
| Brier | 0.170572 | 0.170463 | 0.170244 |
| 2% edge, 1% bankroll scenario P/L | -$23.96 | +$5.09 | — |
| Traded city/day groups | 31 | 10 | — |

These are the same later validation cohort and execution assumptions, not whole-dataset headline scores. Scenario costs include 2-cent slippage, assumed fees, execution-time price-model refresh, event/portfolio caps and loss stops. Historical fills remain unverified.

The gain is small and the challenger still loses to the market on both forecast metrics. Only 10 independent events traded at the 2% threshold, below the unchanged 30-event minimum; thresholds of 4%, 6% and 8% produced no trades. No strategy passes. The active model was not replaced, and no final test was evaluated. All periods were previously inspected, so this remains exploratory development, not a newly untouched holdout.

Saved artifacts and the predeclared protocol are in `reports/focused-challenger-v1/`. The runner is `work/train_focused_challenger.py` at the workspace root. Strategy reports now provide explicit rejection reasons for sparse events, nonpositive profit and an uncertainty bound that includes losses.

The remaining-day weather model still needs settled prospective features. Neither synthetic labels nor relaxed thresholds were used to force a pass.
