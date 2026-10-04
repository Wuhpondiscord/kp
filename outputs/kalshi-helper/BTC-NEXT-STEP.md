# BTC follow-up: settle the data question before increasing complexity

The next decision is to prioritize settlement-source and timing alignment. A
controlled calibration experiment is now implemented and run, with no deployment,
strategy search, weather changes or reevaluation of the original test set.

## Implemented experiment

`helper/btc_calibration.py` adds **BTCAverageRaw** and **BTCAverage**. The former
adjusts the original Gaussian volatility baseline for averaging over the final
minute. Under a constant local Brownian model, if T minutes remain from the latest
observed price, terminal-price variance is sigma² T, whereas the continuous
final-minute-average variance is sigma² (T - 2/3). The implementation scales the
standardized target distance accordingly and rejects decisions where the averaging
window is already partially observed.

This is a log-price, continuous-average approximation to an arithmetic average of
60 discrete BRTI prices. It does not fix Coinbase/BRTI basis, price jumps or receipt
latency. BTCAverage fits only a bias and positive scale on the resulting probit
score, with a fixed regularizer. It learns two parameters rather than adding a
large architecture or searching a strategy grid.

Three expanding chronological folds fit before September 7, 11 and 15 and score
the following four days with a two-hour embargo. A final fit uses the original
1,704 training contracts and scores the 559 validation contracts. The source
loader opens train.jsonl and validation.jsonl only, verifies their manifest hashes,
rejects overlapping tickers and out-of-period rows, and checks input availability.
The earlier test dates remain exposed historical evidence, not a fresh holdout;
this module does not open test.jsonl. Development dates were already inspected,
so these results are exploratory, not confirmatory.

## Results

| Model | Forward-fold log loss | Forward-fold Brier | Validation log loss | Validation Brier |
|---|---:|---:|---:|---:|
| Market | 0.438147 | 0.141640 | 0.413306 | 0.133611 |
| Original BTCVolatility | 0.445456 | 0.143576 | 0.427762 | 0.138863 |
| BTCSignal | 0.455181 | 0.147968 | 0.433219 | 0.140467 |
| BTCMarketGuard | 0.443349 | 0.143863 | 0.411535 | 0.133377 |
| Price-only control | 0.438506 | 0.141867 | 0.411718 | 0.133211 |
| BTCAverageRaw | 0.446109 | 0.143423 | 0.426528 | 0.138546 |
| BTCAverage | 0.443327 | 0.143343 | 0.431090 | 0.140330 |

The averaging correction slightly improves validation log loss (0.29% versus the
original volatility baseline), but slightly worsens forward-fold log loss. Fitted
calibration improves aggregate forward scores but worsens later validation. None
of these spot-only methods consistently beats the market. The market-offset
model's advantage is also unstable: its three fold log losses are 0.428527,
0.437709 and 0.463934, versus market scores 0.436436, 0.425533 and 0.452747.

The diagnostic report breaks scores down by market-price bucket. On validation,
BTCAverageRaw beats the market in the <10% and >=90% buckets but loses in all
three intermediate buckets. This is descriptive, with small cohorts; it is not
authorization to trade only the historical winning buckets. Bin counts and
calibration means are recorded alongside losses. Proper scores combine calibration
and discrimination, so a lower Brier score is not by itself proof that calibration
alone improved ([scikit-learn explanation](https://scikit-learn.org/stable/modules/calibration.html)).

No new P&L optimization was run. The candidate has not earned a new profitability
claim, and using the old test to choose a mixture or threshold would turn it into
development data. Original BTC results remain unchanged in btc15m-pilot-v1.

## The more consequential data opportunity

Kalshi now documents a [CF Benchmarks REST passthrough](https://docs.kalshi.com/cfbenchmarks/rest-passthrough):
`GET /trade-api/v2/cfbenchmarks/history/values?id=BRTI&timespan=HOUR&timestamp=...`.
It requires signed Kalshi API requests and appropriate account entitlement.
No separate CF API key is required by that interface, but access for this project
has not been verified. The existing anonymous public collector cannot fetch it.
Do not put a private key in the shared Hugging Face demo or send it in chat.

The [authenticated WebSocket feed](https://docs.kalshi.com/websockets/cfbenchmarks-value)
also supplies source timestamps, receipt timestamps and averaging fields. This
provides a plausible route to testing the actual settlement signal rather than
assuming Coinbase is interchangeable. A documented endpoint is not evidence that
we already collected its data. No BRTI feed integration or authenticated request
was performed in this update.

The next data experiment should:

1. Verify authorized BRTI access locally; preserve raw responses and hashes.
2. Compare historical BRTI and Coinbase at matching timestamps and quantify basis,
   target-side disagreements and label disagreements by horizon. Keep official
   Kalshi outcomes as ground truth.
3. For prospective records, save source time, local receipt time, target publication,
   and contemporaneous executable bid/ask depth. Historical source times alone do
   not prove live availability.
4. Declare the dataset window, exclusion rules, architecture and trading rule before
   a new untouched-period comparison. Test the same models with Coinbase versus
   BRTI inputs to isolate source quality before adding complexity.

The feed's final-minute accumulation has documented boundary and sample-count
semantics. Any future importer must validate completeness rather than treating
partial rolling averages as official final settlements. Entitlement failure
must remain explicit; there should be no silent substitution with Coinbase.

## Reproduce

From outputs/kalshi-helper:

```powershell
python -m helper.btc_calibration
python -m unittest discover -s tests
```

Outputs are in `reports/btc-average-calibration-v1`: a frozen protocol, two fitted
parameters, forward-fold/validation diagnostics and hashed inputs/code/results.
All model and data changes remain local research work. No website changes are needed
until there is a model that merits being offered to users.
