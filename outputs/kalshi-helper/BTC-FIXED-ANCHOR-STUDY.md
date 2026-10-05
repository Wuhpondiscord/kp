# Fixed market anchor and input-age study

## Finding

Removing learned market recalibration gives a small development-score improvement. It does not establish a tradable edge: no new-model signal survives the unchanged 4-cent net-edge threshold after costs, and a one-minute delay removes the forecast advantage. Keep the candidate experimental and the July 21–31 holdout sealed.

## Implemented changes

- Added `FixedAnchor`, which predicts `sigmoid(logit(market_probability) + standardized_external_features @ weights)`. Unlike the previous full-flow model, it has no fitted intercept or extra market-log-odds coefficient. When external features equal their training means, it returns the market probability. This does not guarantee unbiased probabilities generally; standardization still defines the reference feature state.
- Kept the four external inputs, L2 penalty .1, coefficient bounds ±.25, chronological training windows, two-hour embargo, bankroll, risk limits, fees, and trading threshold unchanged.
- Reconstructed external inputs at 0, 60, 120 and 300 seconds of additional delay from checksum-verified local Binance/Coinbase archives. Models train on current inputs; delay is applied at inference. Market quotes, labels, and evaluated contracts stay fixed. These are discrete minute-bar stress scenarios, not measured historical arrival times.
- Added per-period feature drift, forecast calibration bins, signal filtering counts, and a fixed-trade cost decomposition. Expected return uses model probabilities and is explicitly distinguished from realized return.
- Added a fixed development gate requiring both Brier and log loss to beat the raw market and market-only calibration in at least three of four periods and in aggregate, positive net profit, at least 30 trades and 10 trade days. Passing it would not automatically deploy a model or release the holdout.

## Results: 527 contracts, 23 evaluation days

All dates are previously inspected development data. Smaller scores are better. This is an exploratory comparison; there is no new untouched-test claim.

| Model | Brier | Log loss | Net simulated P&L | Trades |
|---|---:|---:|---:|---:|
| Raw market | .121078 | .377750 | $0.00 | 0 |
| Market-only calibration | .122014 | .381645 | $0.00 | 0 |
| Previous full flow | .121173 | .379743 | −$0.46 | 2 |
| **Fixed market anchor** | **.120761** | **.377420** | **$0.00** | **0** |

The fixed anchor reduces Brier by approximately 0.26% relative to the raw market and 0.34% relative to full flow. Its log-loss improvement over the raw market is approximately 0.087%. These small differences are uncertain:

- Brier difference versus market: −.000317; day-cluster bootstrap 95% interval [−.001620, +.000999].
- Brier difference versus market-only calibration: −.001253; interval [−.003724, +.001032].

Both intervals cross zero. They are descriptive, do not adjust for repeated model exploration, and do not capture all possible dependence between days. Zero trades means abstention, not a profitable strategy.

| Evaluation starts | Market Brier | Fixed-anchor Brier | Market log loss | Fixed-anchor log loss |
|---|---:|---:|---:|---:|
| June 28 | .124759 | .126594 | .388313 | .393277 |
| July 4 | .139583 | .138557 | .434558 | .430455 |
| July 10 | .108872 | .108658 | .344688 | .345546 |
| July 16 | .109706 | .107540 | .338273 | .334657 |

The candidate beats both controls on both metrics in only two of four periods. The development gate fails on period consistency, positive P&L, trade count and trade days. It passes only the pooled-score requirement.

## Input age

| Additional delay | Fixed-anchor Brier | Fixed-anchor log loss | Trades |
|---|---:|---:|---:|
| 0 seconds | .120761 | .377420 | 0 |
| 60 seconds | .121976 | .380382 | 0 |
| 120 seconds | .121668 | .379381 | 0 |
| 300 seconds | .122193 | .380146 | 0 |

Every delayed scenario scores worse than the raw market. Results are not monotonic in delay; this is noisy development evidence, not a fitted latency-decay law. The comparison identifies sensitivity to freshness, but it cannot recover actual historical exchange publication or network latency. The baseline already assumes completed bars plus five seconds of publication allowance. Actual arrival-time telemetry is still needed before live claims.

For completeness, the old full-flow model's P&L at 0/60/120/300 seconds is −$0.46 / $0 / −$9.55 / +$14.63, with 2/0/1/2 trades. The five-minute scenario has worse forecast scores despite its two winning trades. It must not be selected as a profitable delay setting.

## Why trades disappear

All stages require the same four-cent estimated edge. Counts refer to contracts, not quantities:

| Stage | Full flow | Fixed anchor |
|---|---:|---:|
| Model–midpoint difference | 108 | 35 |
| After bid/ask spread | 80 | 23 |
| After conservative one-contract fee | 19 | 7 |
| After two-cent slippage | 2 | 0 |
| Simulated fills after sizing | 2 | 0 |

The fixed anchor makes smaller actionable deviations from price. Removing recalibration improved aggregate forecasts without making its remaining deviations large enough for the existing entry rule. This does not prove the rule is optimal, and it is not a reason to lower it retrospectively.

For the previous model's same two filled trades, the cost accounting reconciles exactly (amounts rounded here):

| Component | Amount |
|---|---:|
| Realized gross profit at midpoint | +$1.285 |
| Spread cost | −$0.205 |
| Slippage | −$0.820 |
| Fees | −$0.720 |
| Realized net | **−$0.460** |

The model predicted +$1.806 net for those fills. The realized result was −$0.460. Two trades cannot distinguish systematic miscalibration from outcome noise. This is a fixed filled-cohort decomposition, not a claim that removing costs would leave trade selection unchanged.

## Drift diagnosis

The absolute Binance/Coinbase basis is the most visible distribution shift: the July 4 evaluation mean is 8.79 basis points versus 14.06 in training, a −2.05 training-standard-deviation shift. Later windows remain roughly one standard deviation below their training means. No external feature is clipped at the ±5-standard-deviation model limit in these evaluation windows. This points to a changing input distribution rather than clipping as a plausible concern; it does not identify a causal market mechanism. Earlier basis-removal ablations already failed to improve aggregate scores, so this study does not launch another feature search.

## What to retain

Retain the fixed-anchor candidate and the new diagnostics as research tools. Keep the deployed app and original model artifacts unchanged. The next bottleneck is whether external information can produce a stable, timely correction large enough to survive costs. Current data do not support increasing risk, promoting a model, or treating the small forecast gain as an edge.

Limits remain: sampled hourly contracts, no historical depth or actual fills, USDT/USD basis effects, finalized target metadata assumptions, assumed June fees, reused development dates, and unavailable historical arrival timestamps. The simulator's cent fee approximation is retained for matched comparisons.

## Reproduce and review

From `outputs/kalshi-helper`:

```powershell
python -m helper.btc_anchor_study --cache <existing-public-cache-directory>
python -m helper.btc_anchor_study
python -m unittest discover -s tests
```

Preparation uses the existing raw cache offline and fails on missing data; it cannot silently access the network or drop contracts with missing delayed features. Subsequent evaluation uses the derived age-feature artifact plus the existing development file. Reruns are labeled in the report.

- `helper/btc_anchor_study.py`: fixed-anchor model, age reconstruction/validation, attribution, drift and gate.
- `tests/test_btc_anchor_study.py`: neutral-feature preservation, model artifact validation/round trip, delay chronology and unchanged labels/quotes, YES/NO cost reconciliation, and gate evidence requirements.
- `reports/btc-anchor-study-v1/protocol.json`: choices fixed before the first evaluation.
- `reports/btc-anchor-study-v1/age-features.json` and `inputs.json`: derived timestamped inputs and source hashes.
- `reports/btc-anchor-study-v1/models-*.json`: four periods of fitted candidate/control parameters, saved before scoring.
- `reports/btc-anchor-study-v1/report.json`: complete scores, confidence intervals, calibration, trade ledgers, filtering counts, cost attribution, drift, gate checks and code hashes.

All 316 Python tests pass. The original controls reproduce prior headline scores and complete trade ledgers; calibration-bin floating point differences are below 1e-15. No holdout evaluation or deployment was performed.
