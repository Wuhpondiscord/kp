# Model names, disparity diagnosis, and revision log

This is the reference record for the three current research model families. Historical reports keep their original labels so past results remain reproducible. The renamed research artifacts do not replace the app's active model.

| Current name | Former label | Distinguishing behavior |
|---|---|---|
| **WeatherSignal** | Original | Joint weather/market network; full auxiliary weather loss, no explicit market-correction penalty; three-seed ensemble |
| **MarketGuard** | Revised | Same small architecture; weather loss weighted 0.1, quadratic market-logit correction penalty of 1; three-seed ensemble |
| **ConsensusBlend** | New blend | Fixed 50/50 probability average of WeatherSignal and MarketGuard; no independently fitted gate or weights |

These names describe construction, not proven forecasting skill or safety.

## Why forecast scores and profits differ

### 1. The forecast differences are small; the strategy has a hard threshold

WeatherSignal, MarketGuard, and ConsensusBlend have validation log losses 0.491363, 0.491068, and 0.489681 respectively; Brier scores are 0.162410, 0.162524, and 0.161883. These are small score differences with unresolved uncertainty, not three dramatically different levels of established skill.

Their average absolute probability corrections versus market are much more different: **5.51, 1.76, and 3.55 percentage points**. The fixed strategy requires an estimated net edge of 4 cents after spread and fees. Small changes around this boundary can remove most of the trades, even when average forecast scores improve.

At the same quote, these models produce **234, 6, and 95 entries**, respectively. At the next hourly quote plus 2-cent slippage, they produce **119, 1, and 42**. Thus a lower dollar loss for MarketGuard is largely a much smaller betting sample, not evidence it has identified better trades. One market event can contain several correlated bracket positions.

### 2. The scores and the trading objective weight different things

Log loss penalizes confident mistakes particularly strongly; Brier is squared probability error. A change can improve one and worsen the other without any implementation error. The forecast report weights dates/events, across the entire selected validation cohort. P/L counts only qualifying contracts and weights them by quantity, entry cost, and settlement payout. Bankroll sizing creates additional path dependence and gives cheap contracts larger quantities under a dollar cap.

Forecast scores compare predictions with eventual outcomes, not with the available purchase price plus fees. A calibrated probability is not automatically an exploitable discrepancy. More contracts increase exposure, not the number of independent events.

### 3. The blend benefits from complementary errors, but trading is nonlinear

For each observation, the equal blend's Brier error equals the mean of the two parent errors minus one quarter of their squared probability disagreement. Averaging therefore reduces error relative to the average parent score, and can beat both parents when their errors complement each other. Log loss is also convex in probabilities. Neither property guarantees that a blend beats the better parent in general.

The entry threshold, side choice, fees, and sizing are nonlinear. Averaging two probabilities does not average the parents' portfolios or profits. ConsensusBlend can remove profitable parent bets while retaining other losing ones. Its best aggregate forecast score does not guarantee its best tail-selected trading score.

### 4. The delayed scenario mostly removes bets; it does not simply make every fill worse

The following exact accounting decomposition uses **one contract per entry** to isolate selection from quantity changes. All delayed entries are a subset of the original decision-qualified entries. For surviving contracts, side and quantity are identical.

| Model | Immediate P/L of bets removed by delay | Cost benefit on surviving bets | Total P/L change |
|---|---:|---:|---:|
| WeatherSignal | +$19.82 | +$8.70 | -$11.12 |
| MarketGuard | +$2.37 | +$0.03 | -$2.34 |
| ConsensusBlend | +$11.65 | +$3.05 | -$8.60 |

Identity: delayed P/L minus immediate P/L equals **minus removed bets' immediate P/L plus surviving bets' price/cost change**. Each identity is checked in the runner. The surviving bets were cheaper in aggregate even after the assumed 2-cent slippage. Losing the historically profitable removed subset outweighed that benefit.

This is consistent with the edge recheck retaining contracts whose prices moved away from the fixed prediction and excluding contracts whose prices moved toward it. It is an observed selection pattern, not proof of why the market moved. The forecast is stale in this stress test; full weather inference is not refreshed at execution. It would be incorrect to describe all the deterioration as higher entry costs or claim it estimates a refreshed live strategy.

### 5. Sparse data and repeated selection remain limiting

The parent models train on 87 events across 22 spring dates and are evaluated on 257 events across 67 already-inspected dates (730 selected bracket rows). Several bracket rows share the same outcome. Seasonality, seed variation, and subset composition can dominate small score differences. Forecast uncertainty still includes no improvement versus market prices. Historical quantity/depth and fee overrides are unverified, and immediate fills are optimistic.

The diagnosis supports structural investigation; it does not identify a profitable city, price bucket, or threshold to exploit. No such routing rules were fitted here.

## Fixed improvement attempts

Before training, the runner recorded one attempt per parent and a conservative retention rule: both log loss and Brier must not worsen, and delayed +2-cent bankroll P/L must not worsen. This is an exploratory engineering rule on reused validation, not a statistical promotion gate. No subsequent settings were searched after seeing results.

**WeatherSignal attempt:** average network weights across the last 50 of the fixed 200 training epochs, separately within each seed, to reduce dependence on the final training checkpoint. The three seeds remain equally weighted. No validation-selected checkpoint is used.

**MarketGuard attempt:** retain regularization but give the first 0.1 logit units of correction no penalty; penalize only the squared excess. This is at most about a 2.5-percentage-point unpenalized departure near 50%, smaller toward the extremes. The original +/-0.5 logit output bound remains. The motivation is to allow modest learned corrections without removing the strong-correction penalty, not to lower the trading threshold.

**ConsensusBlend attempt:** remain exactly the 50/50 mixture, using the two attempted parent revisions. No independent third-model training or changed blend weight.

| Family | Version | Log loss | Brier | Delayed +2-cent P/L |
|---|---|---:|---:|---:|
| WeatherSignal | Retained baseline | 0.491363 | 0.162410 | -$68.70 |
| WeatherSignal | Weight-average attempt | 0.492174 | 0.162685 | -$90.87 |
| MarketGuard | Retained baseline | 0.491068 | 0.162524 | -$10.00 |
| MarketGuard | Deadzone attempt | 0.490835 | 0.162635 | -$53.92 |
| ConsensusBlend | Retained parents | 0.489681 | 0.161883 | -$91.49 |
| ConsensusBlend | Attempted parents | 0.490296 | 0.162213 | +$21.53 |

Trading results use $1,000 starting capital and the same 4-cent edge, fees, 1%/$10 per-bet cap, $30 city/day cap, and $200 portfolio cap. The attempted blend trades 70 entries across 62 city-days, deploying $611.47 for a +3.52% return on deployed cash. Its positive return is an exploratory observation; it does not override worse forecast scores and failed parent retention criteria. Selecting this blend solely because it turned profitable would change the rule after seeing outcomes.

**Neither parent attempt was retained.** ConsensusBlend therefore stays linked to the retained parents. The attempted predictions, scores, and ledgers are kept in the audit report; their additional trained model files are not saved. Failed attempts are part of the result and should not be repeated as if untested.

## Current artifacts and reproduction

The current research family directory is `reports/named-model-families/`. It contains exactly three trained-model files: `WeatherSignal.pkl`, `MarketGuard.pkl`, and `ConsensusBlend.pkl`. Existing historical artifacts are preserved for reproducibility. The new directory also contains the protocol, full before/attempt/retained report, prediction audit, and registry/verification metadata.

`work/update_named_families.py` performs the fixed training attempts, verifies parent artifact hashes, reproduces earlier predictions, evaluates candidates, applies the recorded retention rule, and serializes only the three retained families. All saved models are round-trip checked; all scenarios reconcile P/L and satisfy risk caps. Pickle artifacts must only be loaded from trusted sources.

```powershell
python work/update_named_families.py --output outputs/kalshi-helper/reports/named-model-families-reproduction
```

Run from the workspace root with the existing PyTorch environment, source reports, and offline forecast cache. Use a fresh directory. The reserved test and active app model were not changed.

## Revision priorities justified by this diagnosis

1. Collect synchronized decision and execution books with known available depth. Refresh all weather and price inputs at execution before comparing prospective trading policies.
2. Expand paired training data across seasons. More spring-date fitting cannot establish summer generalization.
3. Preserve separate forecast, opportunity, and executable-return reports, including the exact selection/cost decomposition above.
4. Freeze these families and any future structural experiment before new-date evaluation. The rejected attempts do not justify another search for profitable subsets on these same dates.
