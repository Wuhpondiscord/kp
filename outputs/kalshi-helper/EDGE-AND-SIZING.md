# Edge and bankroll sizing evaluation

September 24, 2026. Evaluated frozen retrained models and hash-verified snapshots. No model weights were changed. All results use previously inspected historical dates, assumed quote fills and fees, and no verified historical depth. They do not demonstrate attainable live profits.

## Does the strategy have an edge?

Primary scenario: $1,000 starting balance, decision-time net edge of at least 4¢, same side at a later quote, 2¢ slippage and estimated fees. Opportunities are unique qualifying contracts, not every repeated forecast. The test covers 74 dates.

| Model / sizing | Qualifying contracts | Filled city-days | Contracts bought | Total stakes incl. fees | Net P/L | Net per contract | Return on deployed stakes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Main / one contract | 49 | 34 | 34 | $20.33 | $0.67 | 1.97¢ | 3.30% |
| Main / fixed $10 cap | 49 | 34 | 565 | $331.02 | $22.98 | 4.07¢ | 6.94% |
| Main / 1% bankroll | 49 | 34 | 564 | $330.36 | $22.64 | 4.01¢ | 6.85% |
| Station / one contract | 7 | 5 | 5 | $1.06 | $1.94 | 38.8¢ | 183.0% |
| Station / fixed $10 cap | 7 | 5 | 317 | $47.11 | $114.89 | 36.24¢ | 243.88% |
| Station / 1% bankroll | 7 | 5 | 316 | $47.00 | $114.00 | 36.08¢ | 242.55% |

The very large station percentage comes from a few cheap contracts with favorable historical outcomes. It must not be interpreted as a reliable expected return. Buying 317 contracts still supplies only five city/day outcomes. Rounding fees and differing contract prices change per-contract returns when quantity changes; sizing is not simply a constant multiplier.

Uncertainty is estimated by a seven-day block bootstrap over daily totals, keeping same-day contracts together. The statistic is a ratio of resampled total profit to contracts or stakes, rather than an average of individual trade percentages. Zero-trade resamples have undefined ratios and are counted separately. These intervals describe the observed clustered sample; sparse samples, historical reuse and missing execution data remain serious limitations.

- Main, fixed $10: net-per-contract interval **−15.33¢ to +19.70¢**; deployed-return interval **−25.64% to +34.35%**.
- Station, fixed $10: net-per-contract interval **−6.67¢ to +86.74¢**; deployed-return interval **−46.90% to +666.38%**. Five groups are too sparse for dependable interval interpretation.

**Neither establishes an edge.** Validation contains only 21 filled main-model city-days and four station city-days at the primary threshold. No sizing policy satisfies the predeclared minimum of 30 city-days with a positive lower validation edge bound, adjusted for four sizing comparisons. Validation-selected action remains **no trade**. The test comparisons are reported for diagnosis, not used to choose a winner. Thirty groups is a screening floor, not a guarantee of adequate power; forecast weights already used validation labels, so this is not nested validation.

## How much should it risk?

Compared one-contract, fixed $10, 1% current cost-valued bankroll and quarter-Kelly policies. All policies share these limits:

- At most **1% of initial bankroll per bet**: $10 on $1,000.
- At most **3% open cost per city/day**, shared across brackets even if event IDs differ in the historical input: $30.
- At most **20% total open cost**: $200.
- Whole contracts, shared available cash, fees included in budgets, and one successful entry per contract.
- Main scenario assumes at most 100 contracts can fill at the quoted price. Sensitivities limit fills to 10 or one contract. These are assumptions, not observed depth.

The bankroll policy can shrink after losses; it does not grow beyond the initial per-bet ceiling. Open holdings are valued at acquisition cost for sizing. This is not liquidation equity or mark-to-market wealth. Quarter-Kelly uses the estimated win probability and conservative one-contract cost, then applies the same caps. It matches fixed $10 results here because the cap binds: a 4¢ minimum edge already implies at least a 1% bankroll allocation under this quarter-Kelly formula. There is no observed benefit from added Kelly complexity under these settings.

Main fixed-$10 strategy's cost-valued realized drawdown is **$49.83**, versus $3.62 for one-contract entries. The 1% bankroll variant draws down $50.17. These figures omit adverse movements of still-open positions; they are not full mark-to-market drawdowns. Main fixed-$10 maximum simultaneous open cost was under $20 in this sample, so most bankroll was idle despite $331.02 cumulative deployment. Reused cash contributes repeatedly to cumulative deployed stakes.

Liquidity sensitivity is material: main fixed-$10 P/L falls from $22.98 at an assumed 100-contract maximum to $7.66 at 10 and $0.67 at one. Station falls from $114.89 to $19.58 and $1.94 respectively. The larger profits cannot be asserted without actual quantity/depth evidence. Slippage scenarios at zero and 5¢ are also saved in the JSON report.

## Changes based on the findings

1. Added an explicit **1% per-bet cap** to the live simulator's default settings. Existing 3% event and 20% portfolio limits remain; temperature brackets belonging to the same Kalshi event share that budget. The historical sizing evaluator groups directly by series and settlement-date group, not contract ID.
2. Added **total capital deployed**, **settled capital deployed**, **settled contracts**, **realized profit per settled contract**, and **realized return on settled deployed capital** to paper reports. Open positions are excluded from realized-return denominators.
3. Added deployed capital and settled-stake return to the practice UI. Older reports show unavailable values rather than fabricated zero returns.
4. Added a reusable sizing evaluator, price/quantity sensitivities, clustered ratio intervals and a saved validation sizing decision. The deployed engine retains depth-based execution under the new cap; quarter-Kelly is a research comparison, not enabled automatically.
5. Kept the validation gate closed. Bigger stakes change financial exposure; they do not strengthen the statistical evidence.

These settings are conservative defaults, not an optimized or proven risk allocation. A running server must reload updated code before using the new engine default; existing reports retain their original settings. No real-money trading is supported.

## Validation and reproduction

142 Python tests and frontend checks passed. New tests verify shared city/day limits across different event IDs, per-bet cost caps, accounting and deployed-return identities, quantity not inflating event counts, sparse-data labeling and live settled-only denominators. The final comparison adjustment changes only reported interval quantiles; it does not change fills or P/L.

Artifacts: `reports/sizing-protocol.json`, `reports/sizing-results.json`, `reports/main-sizing-lock.json`, `reports/station-sizing-lock.json`. Code: `helper/sizing.py`; local driver `work/test_sizing.py` requires the frozen model/data files outside the review ZIP. This is offline evaluation, not a fresh live paper session or browser interaction test.
