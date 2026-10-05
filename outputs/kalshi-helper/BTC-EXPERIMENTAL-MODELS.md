# Experimental BTC models: pipelines, benchmarks and limits

Published comparison: October 5, 2026. All three are **EXPERIMENTAL, historical paper replay only**. No model is approved for live trading or automatic promotion. The user interface exposes saved chronological fold predictions, not a newly fitted model evaluated on its own training set.

## Why these three

There is no established overall winner and no proven profitable BTC strategy. These are three useful representatives of different observed strengths on the **same 527 contracts / 23 UTC days**, not three statistically validated winners:

| Model / replay ID | Selection reason | Brier ↓ | Log loss ↓ | Net simulated P&L | Entries |
|---|---|---:|---:|---:|---:|
| Raw market midpoint (forecast control) | Reference | 0.1210778762 | 0.3777504177 | — | — |
| FixedAnchor / `fixed_anchor` | Best balanced forecasting baseline; lowest log loss among this comparison | 0.1207608524 | 0.3774200524 | $0.00 | 0 |
| DirectionalAnchor Joint / `directional_joint` | Lowest Brier among these recent same-cohort candidates | 0.1206802797 | 0.3779286278 | -$9.82 | 1 |
| Delayed MicroFlow / `delayed_micro` | Highest primary simulated P&L in the recent comparison; a timing stress case | 0.1216386059 | 0.3803002349 | +$34.59 | 12 |

All P&L starts from $1,000 and uses the same standard 2-cent slippage scenario. Zero trades is abstention, not evidence of a profitable strategy. Selecting representatives after inspecting results introduces selection bias. The delayed candidate is included to make the forecast/profit discrepancy inspectable, **not because deliberately using stale data is a validated improvement**.

The existing BTC Volatility, BTC Signal and BTC Regime September replays remain available as legacy comparisons. Their 2,830-contract cohort is different; their scores and dollars must not be ranked directly against this table. This document does not claim an exhaustive all-time model ranking.

## Shared data and timing pipeline

1. Collect public Kalshi `KXBTC15M` historical markets, settlement outcomes and quote candles. Validate supported rules, the 15-minute interval, target and available official outcome. These contracts resolve on a final settlement statistic; they are not generic touch-at-any-time contracts.
2. Sample contracts closing on whole UTC hours. Construct 727 usable development records from 732 sampled markets (one unsupported rule/outcome and four missing Coinbase lookbacks excluded). The development span is June 20–July 20, 2026; training warm-up and embargo mean 527 records are scored.
3. Obtain Binance BTCUSDT minute bars, including taker-buy base volume, and Coinbase BTCUSD minute bars. Binance archive checksums and local raw-response hashes protect byte integrity. They do not prove historical feed availability or source equivalence to Kalshi's settlement index.
4. Start the decision around five minutes before expiry. Use only completed bars with a fixed five-second assumed availability delay; refresh market midpoint and spot/features at the execution quote. This avoids using a forecast anchored to a stale earlier market price against a later execution price.
5. Preserve timestamp, feature availability and provenance fields. Reject future inputs, duplicate contracts, malformed or missing inputs; do not silently fill a missing source with a different feed.
6. Generate predictions separately within four expanding chronological folds. Fit only earlier settled labels, with a two-hour embargo at each evaluation boundary. The preprocessing statistics belong to the training fold. Evaluation is June 28–July 20 inclusive: training counts 188/327/466/610, evaluation counts 136/136/141/114. Fold starts are June 28, July 4, July 10, July 16; ends July 4, July 10, July 16, July 21 (exclusive).
7. Concatenate out-of-fold predictions for one continuous paper bankroll simulation. Do not add separately reset fold accounts and call that a continuous return.

Kalshi market prices contain substantial information already. All three models estimate a small correction to that price, rather than replacing it with a prediction of BTC's unconditional dollar price. Binance is a USDT venue and Coinbase uses USD; neither is the exact settlement index. Hourly subsampling, assumed quote availability and unmeasured order-book depth limit deployment relevance.

Sources and code: `helper/btc_research.py` (public Kalshi/Coinbase collector and execution refresh), `helper/btc_new_history.py` (Binance minute flow features), `helper/btc_long_history.py` (archive validation), `helper/btc_inputs.py` (execution validation), `helper/btc_flow_followup.py` (dataset and chronological splits). Source URLs and archive hashes are in the saved manifests. Public upstream references: [Binance public data](https://github.com/binance/binance-public-data), [Kalshi historical markets](https://docs.kalshi.com/api-reference/historical/get-historical-markets).

## FixedAnchor: conservative market correction

Implementation: `helper/btc_anchor_study.py`, base optimization in `helper/btc_flow_model.py`.

Inputs are (1) five-minute signed taker-buy volume imbalance, (2) fifteen-minute imbalance, (3) Binance/Coinbase log price ratio in basis points, and (4) that basis's five-minute change. Imbalance is signed net buy volume divided by total volume. Each feature is centered and scaled using training-only statistics, with a standard deviation floor of 1e-6 and standardized values clipped to ±5.

The probability is `sigmoid(logit(market_midpoint) + z @ weights)`. There are four learned coefficients, no intercept and no fitted rescaling of the market logit. Mean log loss plus L2 penalty 0.1 is minimized with bounded L-BFGS-B; each coefficient is constrained to ±0.25. Neutrality means the training feature mean, not necessarily physical zero. Weights and scaling are numerical JSON, with no executable pickle loading.

Why it appears stronger: anchoring preserves useful market information, while the small correction limits extreme confidence errors. Removing an intercept and market slope reduces opportunities to learn a transient calibration shift. These are plausible explanations consistent with the ablations, not established causal proofs. Its pooled Brier gain over market is only about 0.000317; the exploratory day-block 95% interval for model-minus-market Brier is [-0.001620, +0.000999], crossing zero. It wins both forecast measures in only two of four folds.

Why it does not earn money here: improvements are too small to exceed executable spread, fees and the default slippage assumption. The fixed-anchor opportunity audit goes from 35 raw-edge candidates to 23 after spread, seven after fees and zero after 2-cent slippage. At zero assumed slippage it makes $3.14 over seven trades; 0.5-cent and one-cent scenarios lose $29.20 and $9.82 respectively. Cost changes alter which trades qualify, not merely the price of an unchanged basket. These are sensitivity checks, not measured execution.

## DirectionalAnchor Joint: sign-symmetric price and flow correction

Implementation: `helper/btc_directional_batch.py`, optimization inherited through `helper/btc_context_batch.py`.

Ten correction features combine seven price features and three flow features. Price inputs are target distance normalized by volatility, normalized returns over 1/5/15/60 minutes, and two interactions: target distance times (short/long volatility ratio minus one), and one-minute return times that same ratio adjustment. Flow inputs are five/fifteen-minute imbalance and five-minute basis change. Absolute exchange basis is omitted.

Features have physical zero preserved: there is no empirical mean subtraction. Each is divided by its training root-mean-square, floored at 1e-6 and clipped to ±5. The market-logit offset, L2 0.1, coefficient bounds ±0.25 and optimizer remain the same. Reversing signed inputs reverses the correction. This is an inductive assumption, not a universal financial law.

Why Brier improves but log loss does not: Brier measures squared probability error, whereas log loss punishes confidently wrong outcomes more sharply. The additional signed price/volatility context changes where corrections fall; aggregate improvement in squared error can coexist with worse tail errors. Its Brier difference versus FixedAnchor is only about -0.00008057, with exploratory paired interval [-0.001857, +0.001764]. This does not establish superiority. It produces one losing eligible trade, far too little trading evidence. Do not interpret the extra features as a proven source of independent information beyond market price.

## Delayed MicroFlow: explicitly a timing stress case

Implementation: `helper/btc_microdata.py`, `MicroAnchor(combined=True)` with `attach(..., delay=60)` at evaluation. It shares the **same trained weights** as contemporaneous Flow + Micro, rather than being separately trained on stale inputs.

Its eight inputs are FixedAnchor's four minute features plus four one-second-derived micro features: fifteen-second signed volume imbalance, fifteen-second log return in basis points, five-second minus sixty-second imbalance, and five-second return. Training uses contemporaneous completed data with the same five-second publication allowance. In the delayed replay only the four new micro features are shifted back sixty seconds; market quotes and original minute features stay current. The model uses training mean/std scaling, clipping and the same four-parameter-model objective/bounds extended to eight coefficients.

The source is 32 daily Binance one-second archives (June 19–July 20), 2,764,800 validated seconds. Only required windows are retained. An independent second-to-minute aggregation audit matches all 727 windows for closing price, volume and taker-buy volume within explicit floating-point tolerance. Granularity is not evidence of lower feed latency, and taker flow is not a reconstructed order book.

The current-input Flow + Micro counterpart scores Brier 0.1210027733, log loss 0.3782762558 and loses $13.11 over ten trades. Delaying the new features worsens both forecast metrics yet produces +$34.59 over twelve trades / nine active days. Independently reset delayed fold P&Ls are +$8.83, +$45.24, -$9.62 and -$9.86; profit is concentrated rather than consistently replicated. These fold dollars do not sum to the continuously sized aggregate.

Why the apparent trading improvement is suspect: a small probability change changes eligibility, side and size near the threshold; twelve trades expose a very different subset than the 527-event forecast score. Delayed features can accidentally select winning outcomes. We have not demonstrated that this is the mechanism, or that staleness is helpful. The sparse-trade uncertainty rule refuses an inferential profit interval below 30 entries or ten active days. Selecting this positive timing variant after inspecting it further weakens any profitability claim.

## Trading policy: shared, deterministic, not a second learned model

All three use `helper/sizing.py:simulate` and `helper/core.py` fee/money arithmetic. There are no separate learned trading weights for these three. Older learned gate experiments are not substituted into their benchmarks.

For each eligible execution quote, compare the forecast with the executable YES/NO side, require the four-cent edge threshold under the simulator's spread/cost checks, charge the existing quadratic fee with assumed coefficient 0.07 and cent rounding, and add the chosen slippage. The standard replay uses $1,000, 2-cent slippage, at most 100 contracts per entry, up to 1% per bet, 3% shared open BTC series/day exposure and 20% overall exposure, with available cash constraints. Settlements are processed chronologically and bankroll changes affect later sizes. The standard scenario does not enable optional daily/session loss stops.

The UI permits $10–$1,000 paper balances and fixed 0/2/5-cent slippage scenarios. It reruns sizing rather than multiplying a stored return. No model is retrained when replaying. No orders are submitted. Assumed fills have no historical depth, queue, cancellation or measured latency model. The 0.07 historical fee coefficient is an assumption, including June overrides not independently verified. Gross or optimistic scenario profits cannot establish executable net returns.

## Uncertainty and reproducibility

These dates have been inspected across many experiments. Four chronological folds prevent training on the future within each evaluation, but do not turn repeated model selection into an independent test. Day-block intervals are exploratory and do not correct for the full research search. July 21–31 holdout remains unopened by these models and excluded from this release's new files. No new untouched performance claim is made.

The complete ledgers and calibration bins are in each report's `aggregate` entries; `trading` contains cost sensitivities. Saved `models-YYYY-MM-DD.json` files give each fold's weights and preprocessing. `reports/btc-experimental-manifest.json` hashes runtime data and weights; runtime rejects changed artifacts. `helper/btc_experimental.py` loads fold models and reproduces the benchmark, while `helper/btc_lab.py` exposes them to the existing HTTP API. Historical protocol fields saying deployment is disallowed refer to live model promotion; this publication permits only opt-in historical replay, with no change to those frozen gates.

From `outputs/kalshi-helper`:

```sh
python -m unittest discover -s tests
node tests/frontend_checks.cjs
python -c "from helper.btc_experimental import replay; print(replay('fixed_anchor',1000,.02))"
python -c "from helper.btc_experimental import replay; print(replay('directional_joint',1000,.02))"
python -c "from helper.btc_experimental import replay; print(replay('delayed_micro',1000,.02))"
```

Core files for review:

- `helper/btc_experimental.py`, `helper/btc_lab.py`: experimental catalog and safe replay integration.
- `helper/btc_anchor_study.py`, `helper/btc_directional_batch.py`, `helper/btc_microdata.py`: the three model pipelines.
- `helper/btc_flow_model.py`, `helper/btc_context_batch.py`: common fitting implementation.
- `helper/btc_research.py`, `helper/btc_new_history.py`, `helper/btc_flow_followup.py`, `helper/btc_inputs.py`: provenance and timing.
- `helper/sizing.py`, `helper/core.py`: shared trading simulator and fees.
- `reports/btc-anchor-study-v1/`, `reports/btc-directional-batch-v1/`, `reports/btc-microdata-v1/`: weights, protocols, results and diagnostics.
- `tests/test_btc_experimental.py`: exact score and full-ledger reproduction, hash rejection and risk caps. Other BTC tests cover future-input exclusion, label independence, splits, fitting and serialization.

Recommended research baseline remains FixedAnchor. The next useful evidence would be reproducible gains on untouched dates with measured execution and genuinely independent, correctly timed information. The present release does not justify increasing risk or moving to real money.
