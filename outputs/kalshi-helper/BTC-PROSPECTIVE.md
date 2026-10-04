# Prospective BTC recorder and depth-limited paper replay

This adds a **local experiment runner**, separate from the hosted app. It discovers
public KXBTC15M markets automatically, uses Coinbase for spot features, and requires
no API key. No order endpoints are implemented.

## Frozen before collection

`init` fits the existing BTCMarketGuard, BTCRegime and out-of-fold trade filter
on the original training partition, then saves the models and a 30-day protocol.
It hashes training inputs, model bytes and execution code. A changed protocol,
model or relevant code fails verification rather than silently altering the run.
The market midpoint supplies the forecast benchmark; always-abstain supplies the
zero-profit strategy benchmark. No thresholds or models are selected from incoming
results. Joblib is a Python serialization format: use locally initialized experiment
directories only, never model files received from untrusted sources.

## What is recorded

The SQLite log stores request start times, local response receipt times, source
responses and hashes, book levels, published market targets, predictions, rejection
reasons, source failures and official outcomes. A unique index permits only one
prediction per contract, including after restarting the recorder. Prediction
decisions are restricted to 285–315 seconds before close. Coinbase features use
completed minute bars with the existing five-second allowance and must have
arrived before the prediction is recorded. Books older than 20 seconds at prediction
are rejected. The actual source publication times remain unknown for REST data.

All forecasts are logged even if the trade filter rejects them. Settlement polls
check previously predicted contracts. Retrieval failures are logged; the next cycle
can retry. Raw payloads are append-only through the recorder, with hashes checked
on replay; this is not cryptographically signed or tamper-proof archival storage.

## Paper execution

Replay waits for a book request started at least two seconds after the prediction,
received within 30 seconds and before close. It commits to the decision-time side,
rechecks price/cost against later depth, consumes at most displayed whole-contract
size across price levels, and includes a two-cent slippage stress and the existing
0.07 quadratic fee assumption. YES asks are derived from NO bids and vice versa.

Each strategy has its own hypothetical $1,000 account. Caps are 1% of cost-valued
capital and $10 per entry, $30 concurrent exposure per BTC UTC day, $200 total
concurrent exposure, and 100 contracts. The original four-cent edge requirement
remains. Rejected, late and insufficient-depth opportunities do not produce fills.
These caps are not daily loss stops. Multiple strategy accounts are comparisons,
not a combined tradable portfolio.

Funds become available only when the recorder receives settlement, not retroactively
at its source timestamp. Unsettled positions stay separate from realized P&L.
Reports include market/model scores, settled ledgers, net return per contract,
return on deployed capital and exploratory day-cluster intervals. Empty and sparse
results cannot establish edge. Intervals do not replay bankroll paths.

Displayed REST depth is not guaranteed execution, and exchange timestamps/queue
position are unavailable. Historical candles and live books also differ from the
original training source. This runner improves observability and execution bounds;
it does not eliminate those limitations or verify the assumed fee schedule.

## Start a formal run locally

On Windows, double-click **Start BTC Paper Experiment.cmd**. It initializes once,
then resumes the same directory on subsequent launches. The terminal remains open.
Python and the project dependencies must already be installed.

From outputs/kalshi-helper:

```powershell
python -m helper.btc_forward init --directory data/btc-forward
python -m helper.btc_forward record --directory data/btc-forward --polls 172800 --interval 15
```

The second command runs in the terminal; the computer and process must stay awake.
Stop with Ctrl+C, then resume using the same record command and directory. Do not
run two recorders against the same directory. Collection is not an installed service
or an automation. Keep this checkout's frozen code unchanged during the experiment;
use another checkout for development. Preserve the entire data directory privately.

Read the latest report with:

```powershell
python -m helper.btc_forward replay --directory data/btc-forward
```

New-prediction collection stops at the frozen end time. Continue retrieving pending
settlements afterward without creating new predictions:

```powershell
python -m helper.btc_forward settle --directory data/btc-forward --polls 20 --interval 15
```

Monitor missing data and errors while collecting. Do not tune the strategy in
response to interim P&L. The 30-day end is a fixed checkpoint, not a promise of
enough trades or statistical power.

## Verification

Tests exercise complementary prices, fractional-depth flooring, quantity and
cash caps, delayed execution, stale/pre-requested snapshots, settlement accounting,
abstention, malformed books and changed protocol/data detection. A separate
`data/btc-forward-smoke` directory is used for bounded live checks; its records
must not be pooled into a formal evaluation.

The full 30-day experiment is **not** running in the background. The smoke test is
a functionality check, not a profitability result. Nothing is deployed to HF.

The bounded public-API smoke check recorded an actual BTC prediction and subsequent
books with zero fetch errors. Neither strategy qualified for a fill. Later execution
guard fixes intentionally invalidate that smoke directory's code hash; retain its
records as engineering evidence and initialize a new directory for the final runner.

Final runner verification: **277 Python tests passed**. A second, newly initialized
smoke directory completed a public-data poll with the final code. No prospective
profitability claim is made. The launcher is provided but has not been left running.

Order-book conventions follow the [Kalshi order-book API](https://docs.kalshi.com/api-reference/market/get-market-orderbook).
