# BetCheck · Kalshi betting helper

Version 0.9.0. Local Kalshi market research and simulated-money practice. Current ML covers daily high temperatures in NYC, Chicago, Miami and Denver; other browser categories do not have trained models.

The combined-model families are **WeatherSignal**, **MarketGuard**, and **ConsensusBlend**. Their [disparity diagnosis and revision log](MODEL-DISPARITY-LOG.md) explains the research results. The [named-model UI guide](NAMED-MODEL-UI.md) describes app selection, custom blends, historical practice, continued training, and experimental live coverage. Older price-only models remain separate advanced tools.

The [ConsensusBlend weight comparison](CONSENSUS-WEIGHT-COMPARISON.md) tests seven fixed parent mixtures. The retained 50/50 mix leads forecast scores; alternative weights have different but unproven trading results.

Read [ML-DL-REVIEW.md](ML-DL-REVIEW.md) for the architecture/data audit and [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md) for external-review findings, new diagnostics and remaining limitations. Historical reports describe specific runs, not a proven betting edge.

The [historical forecast replay setup](FORECAST-ARCHIVE-SETUP.md) documents the Hugging Face/Kaggle source audit, installed GFS/ECMWF single-run downloader, offline reproduction command and first real-data physical-model comparison. This separate research workflow currently underperforms market probabilities and does not replace the active live model.

The [physical-model diagnosis](PHYSICAL-MODEL-DIAGNOSIS.md) tests a missing current-observation correction, documents why the historical model struggles, and separates weather-model development from a future learned execution model.

## Start here

The [weather/market neural comparison](JOINT-MODEL-COMPARISON.md) compares seven approaches on identical historical validation data. A small combined ensemble scores best, but uncertainty includes no improvement; it remains research-only.

The [targeted combined-model refinement](JOINT-MODEL-REFINEMENT.md) tests auxiliary weather-loss weighting and correction regularization. The revision improves seed consistency and slightly improves log loss, with a small Brier tradeoff and no established edge.

The [combined-model profitability check](JOINT-MODEL-PROFITABILITY.md) compares fixed-risk trading scenarios. Optimistic same-quote profits do not survive delayed execution consistently; the revision produces too few bets to establish an edge.

The [three-model comparison](THREE-MODEL-COMPARISON.md) adds a fixed blend of the original and revised ensembles. It has the best validation forecast scores, but loses in delayed-execution scenarios.

Use Python 3.11 or later. Install dependencies once with `python -m pip install -r requirements.txt` (check your actual environment; the captured review environment has an older tzdata version). Double-click **Start Helper.cmd**, then open **http://127.0.0.1:8765**. Alternatively run `python -m helper serve` in this folder. Keep the app and computer running during paper sessions.

1. Paste a Kalshi market URL into **Find a bet**, or press **Try the weekend rain market**. Event and series links expand into contracts; individual tickers work too.
2. Review buy prices, market-implied odds, forecast sources, and the settlement rules. Market prices are never labeled independent model probabilities.
3. Choose a starting balance and duration, then **Start practicing**. You can use the analyzed link or automatic daily-weather discovery.
4. Follow **My practice bets** for cash, realized P/L, open bets, data health, and reasons for not trading.
5. Open **Model room** and press **Train new model** to watch actual epoch training. Use **Compare model families** for the cached expanding-window comparison. Click any market title for a fresh model explanation.

## Strategies and current coverage

- **Auto** trades only with a model that passes the predefined forecast validation gate. The current model did not pass; auto collects prices without fabricating signals.
- **Experimental** uses the selected named family and blend in the new practice setup. Named models use issued forecasts and station observations in an experimental 12–14-hour pre-close window; outside it they collect prices only. Legacy price-only tools have their own wider saved coverage and remain separate. Neither path establishes a profitable edge.
- **Watch prices** records real data without buying. Advanced users can supply timestamped JSONL forecasts; doing so disables built-in predictions for that session.

For the example weekend-rain URL, the analyzer finds 23 contracts and retrieves NWS rain-probability periods for 22 mapped stations. West Palm Beach's `CLIDJT` mapping is left unsupported rather than guessed. NWS probabilities are for measurable precipitation during their stated periods; they are **not calibrated full-weekend contract odds**. The contracts settle from The Weather Company, periods cross day boundaries, and the weekend may already be underway. The app displays these facts and does not automate rain bets from an invalid mapping. Other categories can be discovered and observed; independent mentions/post-count/election models remain future work.

## Real-data evaluation

The latest workstation archive produced 51,747 rows from 8,806 contracts, 1,468 events and 367 dates. It is not included in the professor review ZIP. Main training uses chronological 60/20/20 date partitions and a 48-hour settlement embargo: 30,806 train, 9,671 validation, 10,438 test and 832 purged rows. The older comparison path uses expanding validation folds.

The main test spans 74 days: model log loss 0.208091 versus market 0.208974; Brier is slightly worse and the paired interval crosses zero. Reused test dates separately block automatic promotion. The station-informed experiment is separate from live picks. Its original historical loss is 0.197390 versus 0.199210 on a different 5,274-row test. See REVIEW-RESPONSE for reconstruction differences and evidence that most scoring gain is concentrated at very low market prices. **No proven edge.**

Dollar backtests use later hourly quotes with explicitly assumed quantity, fees, and slippage. Historical depth is unavailable, so these are **indicative scenarios, not verified executable returns**. The separate event replay uses recorded books when available. Synthetic demos remain clearly labeled and are not included in real-data evaluation.

## Paper execution

Every custom reference is resolved before the worker starts. Missing/closed markets fail clearly instead of endlessly requesting a series as a contract. Fee metadata is read from series/event overrides; unknown fees disable fills. The worker records per-market health and model gating reasons. Auto-mode market baselines cannot generate orders, even if quotes subsequently move. API `finalized` and `settled` outcomes are recognized.

Entries use a later order-book snapshot, conservative fees, depth limits, and shared event/portfolio risk caps. Sessions stop at their deadline. **Unsettled positions remain open in the report; stopping does not liquidate them or continue settlement monitoring.** Restart marks interrupted sessions; it does not silently resume them. Use one server per data directory. A full 24-hour soak has not yet been completed.

CLI with externally generated predictions remains available:

```powershell
python -m helper paper --tickers "KALSHI-REFERENCE" --bankroll 1000 --hours 24 --predictions "C:\path\predictions.jsonl"
```

## Historical replay

```powershell
python -m helper import examples/synthetic-replay.jsonl --dataset example-replay
python -m helper replay example-replay --bankroll 500 --export reports/example-replay.json
```

The example records carry a `synthetic: true` provenance flag, so they retain the synthetic badge even under a different dataset name. Any run containing synthetic rows is labeled synthetic. Imported data provenance remains your responsibility.

To replay just a time window:

```powershell
python -m helper replay example-replay --bankroll 500 --start 2025-01-01T12:00:00Z --end 2025-01-02T08:00:00Z
```

Replay uses the **same engine** as live paper sessions. It does not train models, perform walk-forward tuning, or independently verify timestamp truthfulness. Train on earlier data and generate out-of-sample predictions before importing them. It cannot manufacture missing historical depth. The separate Backtest & models workflow downloads recent and archived Kalshi hourly candles. It never invents historical order-book depth.

### JSONL format

One JSON object per line. Every object includes `type`, `ticker`, and timezone-aware `available_at` (when the information became usable). `examples/synthetic-replay.jsonl` provides a complete working example.

| Type | Required additional fields |
| --- | --- |
| `market` | `event_ticker`, `close_time`, `status`; `fee_rate` + `fee_version` required to enable fills |
| `book` | `observed_at`, `yes`, `no`: arrays of `[dollar_price, contract_quantity]` bid levels |
| `prediction` | `made_at`, `features_available_at`, `expires_at`, `probability`, `model_id` |
| `outcome` | `settled_at`, `yes_payout` from 0 to 1 |
| `weather_forecast` | Prediction timestamps plus `mean`, `std`, `lower`, `upper`, `station`, `rule_reference`, `model_id`; converted on import |

Use decimal strings for prices and amounts. For example, `"0.4055"` means $0.4055, not 40.55 dollars. All timezone offsets are normalized internally. A dataset's exact duplicate rows are deduplicated. Raw input rows and imported-at timestamps remain stored separately. Imports are atomic.

Fee policy is supplied on the market record, allowing different assumptions and versions through historical time. Unknown fees disable entries. Input metadata updates must arrive with their own availability time. At a shared timestamp, market/outcome/book records precede predictions, preventing scoring a prediction against an already-known settlement.

## Weather model

`examples/weather-inputs.json` demonstrates reviewed normal-distribution inputs:

```powershell
python -m helper forecast examples/weather-inputs.json --export reports/weather-predictions.jsonl
```

The model computes `P(lower <= temperature < upper)`. Null bounds mean an unbounded endpoint. A bracket such as reported 70–74 °F maps to `[69.5, 74.5)` **only if the real reporting chain uses nearest-integer rounding**. Verify exact station, observation period, rounding, units, and source report before configuring a real market. The required station and rule-reference strings document your mapping; they do not prove it is correct.

The URL analyzer now loads real NWS precipitation forecasts for supported weekend-rain station rules. Experimental Gaussian forecast post-processing and station-informed trees exist, separately from live picks; full ensemble EMOS and calibrated rain-contract prediction are not implemented. The demo exercises this model using invented inputs. The next model milestone is a small set of verified daily-weather contracts with archived forecast vintages and official settlement reports.

## Accounting and execution assumptions

- Buy at the opposite outcome's implied ask; walk up to 20 recorded levels. Default participation is 10% of visible depth per price level, rounded down to whole contracts.
- Require a **new** snapshot observed at least 5 seconds after signal arrival. With 60-second polling, actual simulated latency is usually at least one polling cycle. A signal does not fill against its original book.
- Compare probability to average entry debit, including fees, at each selected price level. Default minimum estimated edge is 4 cents per contract.
- Default caps: 3% of initial bankroll per event, 20% total open cost. Fees count toward caps. Correlation across separate event IDs is not modeled yet.
- At most one successful entry per market per run; no selling before settlement, shorting, leverage, market making, or Kelly sizing yet.
- Fee model: `rate × quantity × price × (1 − price)`, with each price-level total debit rounded up to cents. This is deliberately conservative and **does not reproduce the exchange's newer per-order rounding accumulator, rebates, or direct-member precision**.
- Settlements require `settled` or `finalized` status. YES receives the supplied payout; NO receives its complement. Settlement corrections are rejected rather than silently rewriting a balance.
- Cash + estimated liquidation proceeds is an **equity lower bound**: missing/stale depth and quantities exceeding the participation cap are valued at zero. It is not a midpoint valuation. Equity drawdown may reflect missing data as well as trading losses.
- Realized P&L is recognized on settlement and includes entry costs and fees. Open position cost is shown separately. A positive paper result is not evidence of attainable live execution.

## Collection and files

```powershell
python -m helper collect --series EXACT-SERIES-TICKER --limit 12
python -m unittest discover -s tests -v
```

- `helper/core.py`: shared event-driven simulator and accounting.
- `helper/kalshi.py`: public GET requests, rate spacing, retries, raw archives, schema normalization.
- `helper/live.py`: timed collection and paper-trading worker.
- `helper/replay.py`: event replay, JSONL validation, synthetic examples.
- `helper/models.py`: weather distribution baseline.
- `helper/storage.py`: SQLite input journal, run reports, fetch index.
- `helper/server.py`, `helper/static/`: loopback dashboard.
- `data/research.sqlite3`: generated input journal and saved runs.
- `data/raw/`: immutable downloaded responses, indexed by hash and arrival time.
- `reports/`: exported reports.

The server listens only on `127.0.0.1`; it validates Host and Origin for browser mutations. It should not be exposed to the public internet. Nothing is sent to an LLM or external analytics service. Public Kalshi, NWS, Open-Meteo forecast and IEM station data are fetched by the relevant workflows. Responses are archived locally.

The current prototype loads a replay into memory and saves full report snapshots. Start with small, targeted datasets; large-scale storage compaction and bounded streaming replay are future work.

## Next milestones

1. Verify a few weather settlement contracts; archive forecast vintages and official observations.
2. Fit a station/lead-time baseline, walk-forward calibration, and category benchmark reports.
3. Extend the existing historical candle backfill with forecast vintages and trade/depth history where available.
4. Add live settlement refresh/resume, per-market fee schedules, data-health alerts, and longer-run storage compaction.
5. Add cumulative weather, mentions, and post counts through the existing prediction interface.

## API references checked during development

- [Market schema and filters](https://docs.kalshi.com/api-reference/market/get-markets)
- [Order-book structure and reciprocal prices](https://docs.kalshi.com/getting_started/orderbook_responses)
- [Historical-data endpoints and cutoffs](https://docs.kalshi.com/getting_started/historical_data)
- [Fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
- [Fee rounding and accumulators](https://docs.kalshi.com/getting_started/fee_rounding)

API schemas and fee schedules change. The client rejects unsupported book formats rather than guessing price units.
