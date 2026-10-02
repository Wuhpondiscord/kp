# Targeted refinement of the combined model

The revised three-seed ensemble is more consistent across training seeds and slightly better on validation log loss, but slightly worse on Brier score. There is no statistically established improvement over the original ensemble. Both remain research candidates; the live app model is unchanged.

## Diagnosis and intervention

The original combined MLP improved results in Chicago, Denver, and 20–80 cent contracts, but hurt New York and 5–20 cent contracts. May was also weaker than the market. These are exploratory slices, not independently confirmed trading rules.

Two changes were tested in a fixed factorial comparison: reduce auxiliary weather interval-loss weight from 1 to 0.1, and add a penalty of 1 times the weighted squared correction to market log odds. The first reduces the shared encoder's dependence on a weather objective that generalized poorly; the second favors smaller departures from market prices. These constants were recorded before candidate training, not optimized on validation. No city exclusions or price-dependent routing were added.

## Validation results

Lower is better. All ensembles average the same three seeds, trained for 200 fixed epochs. All rows, splits, and preprocessing match the original comparison: 87 training events across 22 dates, and 257 validation events across 67 dates (730 selected bracket rows).

| Model | Log loss | Brier |
|---|---:|---:|
| Market benchmark | 0.495074 | 0.164566 |
| Original combined ensemble | 0.491363 | **0.162410** |
| Reduced weather weight only | 0.495399 | 0.164207 |
| Correction penalty only | 0.491475 | 0.162634 |
| Reduced weather weight + penalty | **0.491068** | 0.162524 |

The combined revision improves log loss by 0.000295 (about 0.06%) against the original, while worsening Brier by 0.000114. Its descriptive, three-comparison-adjusted seven-date block-bootstrap interval for the log-loss difference is **[-0.010745, +0.009621]**. This interval includes zero by a wide margin. Relative to the market, the revised ensemble's observed log-loss reduction is about 0.81%, not evidence of profit.

Individual-seed log losses for the revised model are **0.489920, 0.491997, 0.493570**, all below the market benchmark on these dates. Original individual seeds scored 0.497329, 0.494373, and 0.500616. This is improved seed consistency in this sample, not proof of future robustness. Reducing the weather loss alone hurt all three seeds; the correction penalty is important to this result.

## Where the tradeoff appears

| Validation subset | Market | Original | Combined revision |
|---|---:|---:|---:|
| New York | 0.47063 | 0.49332 | 0.47579 |
| Miami | 0.52957 | 0.52541 | 0.52271 |
| Chicago | 0.50555 | 0.48331 | 0.49650 |
| Denver | 0.52198 | 0.50259 | 0.51145 |
| 5–20 cents | 0.26295 | 0.26985 | 0.26506 |
| 20–50 cents | 0.61965 | 0.60802 | 0.61082 |
| 50–80 cents | 0.64000 | 0.62946 | 0.63250 |
| 80–95 cents | 0.28746 | 0.28718 | 0.29255 |

Entries are subset log loss, weighted within each subset. They do not add up to the overall score. The revision reduces the New York and cheap-contract weaknesses but gives back some Chicago, Denver, and mid-price gains. New York and cheap contracts still underperform the market. The 80–95 cent subset contains only 28 rows. No further architecture or parameter search was performed after observing these results.

## Saved implementation and checks

`helper/joint_research.py` now supports explicit auxiliary-loss weighting and correction regularization, retaining the original defaults. `work/refine_joint_models.py` reproduces the experiment offline from the existing forecast cache and training/validation files. It verifies original model hashes and reproduces the original predictions to 1e-12 before comparing revisions.

`reports/joint-refinement/` contains the pre-training protocol, report with all city/month/price diagnostics and seed scores, row-level predictions, nine trained model artifacts, and verification metadata. `candidate.json` identifies the combined revision's three equally weighted members as a research candidate on the basis of lowest validation log loss. The original ensemble remains saved because it has the better Brier score.

All **207 Python tests pass**, including a behavioral check that stronger correction regularization reduces departures from market prices on a controlled fixture. Fourteen input, source, and model hash checks passed. Reserved test data were not loaded. No live promotion, execution simulation, or profitability evaluation was performed.

From the workspace root, with the optional PyTorch environment and historical cache available:

```powershell
python work/refine_joint_models.py --output outputs/kalshi-helper/reports/joint-refinement-reproduction
```

Use a fresh output directory. Existing validation has now informed this additional comparison, so it is increasingly unsuitable as independent evidence. A frozen candidate needs new, seasonally broader paired data before claiming improvement or an edge. Adding more variants to these dates would not resolve that limitation.
