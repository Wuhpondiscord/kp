# Pinned pretrained model: reproducible benchmark and output-calibration repair

## Outcome

The next research direction was actually implemented, not merely proposed: local TabPFN-v2 inference on the same chronological BTC task. It underperformed. A subsequent constrained market-relative calibration greatly reduced its scoring regression but did not beat the existing fixed-anchor baseline or establish profitability. Preserve the results, keep the previous model unchanged, and do not deploy either candidate.

## Exact pretrained setup

- `tabpfn==2.2.1`, CPU, four PyTorch threads, four estimator configurations, seed 1729; remaining classifier defaults unchanged.
- Official `Prior-Labs/TabPFN-v2-clf` checkpoint `tabpfn-v2-classifier-v2_default.ckpt` at revision `f851f2a3c941544733b712d8c0f96dfae9b28862`.
- SHA-256: `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`; 29,009,539 bytes.
- Inputs: market probability, ten already timestamped Coinbase features and four Binance flow/basis features. Targets: real settled Kalshi contract outcomes.
- The pretrained weights are **not fine-tuned**. `fit` supplies the labeled training context and preprocessing for in-context prediction. This is not a newly gradient-trained neural network.
- Isolated research virtual environment; production dependencies untouched. The complete 45-package dependency closure was checked and recorded in `research-requirements.txt`.
- Public checkpoint download only; inference is local. Package telemetry is explicitly disabled, Hugging Face offline mode enabled, and package state redirected to the research directory.

This is a reproducible v2 benchmark associated with established research, not a claim to use the latest or strongest available checkpoint. Four estimator configurations were fixed for CPU practicality before outcomes were inspected, not selected for their scores. The official model card describes the v2 checkpoint and separate weight terms. [Model card](https://huggingface.co/Prior-Labs/TabPFN-v2-clf), [Nature paper](https://www.nature.com/articles/s41586-024-08328-6).

## Experiments

The main experiment tests raw TabPFN probabilities and an inner-trained convex probability blend with the market. The blend weight minimizes log loss on the earlier chronological inner validation slice and is reset to zero if its inner Brier is worse than market. It chooses weights 1 / 0 / 0 / 1 across the four folds. The choice is saved before scoring the outer period. No outer label selects it.

After diagnosing large confidence errors, a separately frozen repair retains the market log-odds offset and adds a scalar times clipped TabPFN-minus-market log odds. The scalar is constrained to [0,.25], penalized with L2 .1, and fitted using the same cached inner validation predictions only. Its four weights are .217865 / 0 / 0 / .108912. These fixed regularization choices deliberately limit the model's ability to override price; there was no grid search over them.

The repair is post-diagnostic exploratory work, not an independent confirmation. All outer dates were previously inspected development data. Both experiments keep the 527-contract/23-day cohort, embargoes, $1,000 bankroll, 1% risk/shared caps, fees, 4-cent edge requirement and primary 2-cent slippage unchanged. The July 21–31 holdout remains sealed.

## Results

| Predictor | Brier | Log loss | Primary net P&L | Trades |
|---|---:|---:|---:|---:|
| Raw market | .121078 | .377750 | $0 | 0 |
| Previous fixed anchor | **.120761** | **.377420** | $0 | 0 |
| Raw TabPFN | .124776 | .397358 | −$57.71 | 105 |
| Inner market blend | .124187 | .392689 | −$51.17 | 65 |
| Constrained log-odds repair | .121394 | .378740 | $0 | 0 |

The repaired model greatly reduces the pretrained regression but remains worse than both the raw market and the previous fixed-anchor candidate. Its zero trading result is abstention, not profitable selection. Raw TabPFN's model-minus-market Brier interval is [−.000465, +.008594]; the repair's interval is [−.000377, +.001283]. These day-cluster intervals are descriptive, cross zero, and are not adjusted for repeated experiments.

## Why this failed here

**Confidence errors, not simply direction accuracy.** Raw TabPFN makes 92 classification errors versus 95 for the market and 91 for fixed anchor. Yet it makes 17 confidently wrong predictions (probability ≤.10 or ≥.90) versus seven for each control. Better direction counts therefore coexist with worse proper probability scores. Its mean absolute probability deviation from market is 4.74 percentage points, versus 1.48 for fixed anchor. These are dataset diagnostics, not a universal conclusion about TabPFN.

**Short calibration windows select the wrong temporal behavior.** The inner blend fully adopts TabPFN in the first and final periods, both of which subsequently lose. It rejects TabPFN in the third period, where the raw model improves forecast scores and earns +$10.33 in the separately reset period replay. A selection procedure can obey chronological boundaries and still generalize poorly. Nested validation prevents direct leakage; it cannot guarantee that a small earlier window represents the next period.

**Model capacity did not add missing information.** The checkpoint had access to the market feature but was not structurally constrained to preserve its probability. The repair addresses that overreach but cannot create an informative external signal. The benchmark provides evidence against this particular input/checkpoint/configuration combination, not against all pretrained models or all fine-tuning.

**Task mismatch remains important.** Strong tabular benchmark performance does not establish incremental information over an already informative market price, exact BRTI-linked settlement probabilities, or profitable execution after spread/fees/latency. Our data still lack historical depth and measured source arrival times. The earlier literature on sequence/order-book models uses a materially different information set.

## Reproduction evidence

The original runner refits the final-period model with the same seed and training context. Maximum prediction difference is **0.0**. A second independent process repeats the same fit while blocking `socket.connect` and `socket.create_connection`; it also reproduces with maximum difference **0.0**. This confirms local reproducibility of that period under the recorded environment, not generalization to new outcomes.

The four inner-plus-outer fits took approximately 3.55, 3.95, 5.21 and 6.85 seconds in the recorded run; these are machine-specific measurements, not performance guarantees. No GPU was available. The repeated final fit took about 3.84 seconds in the main runner.

Installation initially needed dependency-resolution cleanup and a writable package-state path. These setup failures occurred before model scoring and did not lead to altered data, checkpoints or hyperparameter choices. Package state is separate from enabled telemetry; telemetry remains disabled.

## Decision and next evidence needed

Reject raw TabPFN and its two calibrated variants as replacements. The noticeable reduction in the repair's losses versus the raw model is not an improvement over our best baseline. Calling that a win would change the comparison after seeing results.

The research-guided pretrained test was worthwhile: it rules out “a stronger off-the-shelf tabular learner will fix the existing feature table” for this configuration, and the confidence-error audit identifies how it failed. More checkpoint/weight searching on these same dates would not establish a dependable edge. The substantive next step is a richer and correctly timed information set, or new evaluation evidence, while keeping the strong price baseline and exact contract target. A SOTA claim requires comparable independent benchmark results; it cannot be obtained by renaming this experiment.

## Files

- `helper/btc_tabpfn_research.py`: optional-dependency local benchmark, checkpoint validation, chronological blend selection, output caching and repeated fit.
- `helper/btc_tabpfn_anchor.py`: fixed market-relative repair using cached predictions, with ticker alignment checks.
- `tests/test_btc_tabpfn_research.py`: blend endpoints and Brier gate, invalid inputs, checkpoint rejection before optional imports, bounded correction behavior.
- `reports/btc-tabpfn-v1/`: frozen protocol, per-period training-context identities and predictions, environment/metrics/report, dependency lock and network-blocked reproduction result.
- `reports/btc-tabpfn-anchor-v1/`: separately frozen repair protocol, learned scalar weights and report.

The 29 MB checkpoint and isolated virtual environment are local research dependencies outside the Git checkout; checkpoint weights are not silently bundled into the hosted app. All pretrained output probabilities needed for reviewing or reproducing the calibration repair are saved in the report directory.

## Commands

Use an isolated environment containing the recorded dependencies and obtain the exact pinned checkpoint from its official repository. Verify its SHA-256 above before inference. From `outputs/kalshi-helper`:

```powershell
<research-python> -m helper.btc_tabpfn_research --checkpoint <verified-local-checkpoint>
python -m helper.btc_tabpfn_anchor
python -m unittest discover -s tests
```

The repair and ordinary unit suite do not require TabPFN installation. The main benchmark intentionally fails if its package version or checkpoint hash is wrong. Repeated report generation is explicitly marked as a rerun. No hosted code, production requirements, prior model weights or sealed evaluation data were changed.
