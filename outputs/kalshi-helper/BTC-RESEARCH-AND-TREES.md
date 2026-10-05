# Research review and nonlinear benchmark

## Decision

Reject the new boosted-tree models as replacements. Training-period selection modestly reduces their aggregate loss but does not make them competitive with the market or fixed-anchor baseline. Keep the app and previous models unchanged, preserve every result, and keep the July 21–31 holdout sealed. There is no basis to call this project state of the art or profitable.

## Research reviewed and how it applies

### Tabular benchmarks: test nonlinear baselines, do not assume neural superiority

Grinsztajn et al. compare neural and tree models on tabular benchmarks and find strong tree performance in their studied settings. This motivates a nonlinear benchmark for our small numerical feature table. It does not imply trees must win on this financial dataset or that their benchmark is the current universal state of the art. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html).

### Deep market models depend on different inputs

DeepLOB learns from high-frequency limit-order-book sequences using convolutional and recurrent components. Our hourly-selected contract observations and minute candles lack that order-book history, so copying the architecture would not reproduce its information set or task. [DeepLOB](https://arxiv.org/abs/1808.03668).

Research on stationary order-book features likewise studies representations derived from order-book data. Our normalized returns and signed imbalances are related modeling ideas, but they are not equivalent datasets or reproductions of that work. [Stationary order-book feature paper](https://arxiv.org/abs/1810.09965).

### Pretrained tabular models are a credible separate candidate

TabPFN's Nature paper reports a foundation-model approach for small/medium tabular datasets. That makes it relevant to our sample size, but it does not demonstrate market-relative probability accuracy or net returns on these contracts. It would require a separately frozen local benchmark with the exact checkpoint, feature schema, chronological splits, runtime and calibration recorded. [Nature paper](https://www.nature.com/articles/s41586-024-08328-6).

The official implementation distinguishes code licensing from weight licensing and supports multiple model versions. A future experiment must pin the checkpoint and review its applicable terms rather than silently downloading an evolving default. Local inspection found PyTorch installed and TabPFN absent. **TabPFN was researched, not installed, trained or benchmarked in this batch.** [Official implementation](https://github.com/PriorLabs/TabPFN).

### Repeated backtests need explicit accounting

Bailey et al. explain how repeated strategy selection can produce backtest overfitting. An inventory of experiments helps audit what was tried, but it is not itself a probability-of-overfitting estimate or statistical correction. We still need new evaluation evidence before confirming a candidate chosen after many development comparisons. [Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

Research above was checked for this implementation on October 5, 2026. These sources motivate methods and limitations; none establishes a profitable Kalshi strategy.

## Implemented nonlinear experiment

Use scikit-learn's public `GradientBoostingClassifier(init=...)` interface to initialize the score from the current market probability. Trees learn residual corrections. This avoids the older pattern of overwriting learned tree values in place. Tree structure is read for JSON serialization and checked against the library's predicted probabilities. [Official API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

Inputs are market probability, ten timestamped Coinbase features and four Binance flow/basis features. Fixed settings: depth 2, minimum 30 rows per leaf, learning rate .05 and seed 1729. Compare an unconditional 32-stage challenger with a training-selected challenger choosing 0, 8 or 32 stages. Stage zero returns the unmodified market.

Selection uses the previous chronological inner split with embargoed, already-settled labels. A candidate must have inner Brier no worse than market; among eligible candidates, choose minimum inner log loss with simpler models breaking ties. Refit on the full outer training window. Outer evaluation outcomes and P&L never select the stage count.

The same 527 development contracts, 23 evaluation days, four chronological periods, fees, 4-cent threshold, 2-cent primary slippage, $1,000 bankroll and 1% risk/shared caps are retained. Protocol and input hashes were frozen before scoring. No evaluation threshold was relaxed.

## Results

| Predictor | Brier | Log loss | Primary net P&L | Trades |
|---|---:|---:|---:|---:|
| Market | .121078 | .377750 | $0 | 0 |
| Fixed-anchor baseline | **.120761** | **.377420** | $0 | 0 |
| Fixed 32-stage trees | .124407 | .391689 | −$56.93 | 108 |
| Training-selected trees | .124176 | .390189 | −$47.95 | 70 |

Selection chooses 32 / 0 / 0 / 32 stages across the periods. It rejects learning in two windows, but still chooses trees in the first and final windows, both of which lose on outer evaluation. The selected model's Brier difference versus market is +.003098, with day-cluster bootstrap 95% interval [−.000767, +.008095]. This descriptive interval is not adjusted for our full history of experiments.

| Evaluation starts | Fixed-tree P&L | Selected-tree P&L | Selected stages |
|---|---:|---:|---:|
| June 28 | −$36.30 | −$36.30 | 32 |
| July 4 | −$15.69 | $0 | 0 |
| July 10 | +$6.17 | $0 | 0 |
| July 16 | −$12.05 | −$12.05 | 32 |

Aggregate P&L comes from a continuous-bankroll replay; per-period results reset the bankroll. Their totals therefore need not match exactly. For example, selected-tree aggregate −$47.95 differs from the sum of its independently reset period losses.

## Interpretation

Nonlinearity generated many more qualifying probability deviations, but those deviations did not improve forecast accuracy or net returns. The weak first-period performance dominates much of the loss. The short inner window did not reliably identify the periods in which nonlinear corrections generalize; selection can reject a later useful correction and accept a later harmful one.

These results reject this tree configuration, not all boosted trees or all deep learning. A broader search might find a better historical score, but would not establish dependability on repeatedly inspected dates. Adding parameters is not a substitute for a richer, correctly timed information set or independent evaluation.

## Concrete next research requirements

1. **Comparable information:** for a DeepLOB-style experiment, obtain order-book sequences rather than treating candles as depth. Record event time, local receipt time, sequence gaps and executable sizes. Keep spot BTC price direction distinct from the exact Kalshi settlement target.
2. **A pinned pretrained benchmark:** evaluate a specified local TabPFN checkpoint on the same chronological market-relative task, with immutable input/output hashes and runtime. This is a separate planned experiment, not a completed benefit or a deployment recommendation.
3. **Dependable evaluation:** retain raw-market and simple-model controls, untouched dates, fee/depth/latency assumptions, period consistency and net returns. Previously inspected dates remain development data even when a new architecture is used.
4. **No SOTA label without comparable evidence:** papers' headline metrics use other targets and datasets. A claim here requires an explicit benchmark against relevant alternatives and execution evidence; profitability cannot be inferred from tabular accuracy alone.

## New audit ledger

`reports/btc-trees-v1/experiment-ledger.json` inventories BTC report files that expose aggregate scores, including hashes, named models, event/day counts, headline metrics, P&L and holdout/deployment flags. It preserves report grouping so different cohorts are not accidentally ranked together. Repeated controls remain visible.

The ledger intentionally does not claim to count every historical optimization run or compute a global multiple-comparison correction. It is an audit index. Earlier research that uses another report schema is outside its stated scope.

## Reproduction and files

```powershell
# From outputs/kalshi-helper; existing dependencies/data, no network needed.
python -m helper.btc_tree_research
python -m unittest discover -s tests
```

- `helper/btc_tree_research.py`: public-API market-initialized boosting, JSON inference/validation, training-only stage selection, evaluation and ledger generation.
- `tests/test_btc_tree_research.py`: exact market fallback, JSON prediction equivalence, label-independent inference, malformed/cyclic tree rejection and cohort-preserving inventory.
- `reports/btc-trees-v1/protocol.json`: fixed settings and input/control hashes.
- `reports/btc-trees-v1/models-*.json`: inner choices and fitted trees per period.
- `reports/btc-trees-v1/report.json`: all forecast, calibration, uncertainty, trading and period results.
- `reports/btc-trees-v1/experiment-ledger.json`: report index described above.

Original artifacts, the sealed holdout and hosted settings remain unchanged. No improvement or production deployment is claimed.
