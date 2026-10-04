# Experimental BTC paper lab

Open `/btc` or choose **BTC paper lab** in the main navigation. Select BTC
Volatility, BTC Signal or BTC Regime, enter $10–$1,000 of simulated money, and
replay September. All inputs are bundled; no API key or forward collection is needed.

This release exposes historical experimentation only. It does not promote a model
into weather practice, automatic picks, live BTC trading or real-money orders.
The older experiment's `deployment_allowed: false` remains unchanged: that gate
blocks operational model promotion, while this separate lab makes its negative
results reproducible through the UI with the user's authorization.

The lab's BTC Signal is the external-trained logistic variant, not the older
September-trained BTCSignal research run. The UI states its training provenance.
BTC Regime likewise uses January–April Binance training. First replay rebuilds
the deterministic models from hash-checked numerical arrays, then caches their
September execution-time predictions in memory. No uploaded pickle is accepted.
Weather checkpoint selection and training remain independent.

Changing bankroll runs the sizing engine again, including integer contracts and
fee rounding. Changing the fixed 0/2/5-cent scenario is an assumed-cost sensitivity,
not a new calibration or evidence of achievable fills. The standard scenario
reproduces −$122.52, −$319.83 and −$222.99 from $1,000. Forecast scores are matched
to the same execution rows and market benchmark. All dates were previously inspected.

Requests are serialized with a nonblocking lock to bound shared hosted CPU work.
Results are returned to the requesting browser rather than changing a global active
model or storing a shared wallet. Reports can be downloaded as JSON. Reloading the
page clears the displayed result. The page states that no live collection is started.

Validation: 289 Python tests passed, existing frontend checks passed, JavaScript
syntax passed. Browser replay reproduced the $1,000 volatility loss and a separate
$100 logistic replay (−$27.95). Hosted smoke checks now run all three standard BTC
replays as well as existing weather checks.

See [BTC-LOSS-AUDIT.md](BTC-LOSS-AUDIT.md) for the evidence behind the experimental
status and [helper/btc_lab.py](helper/btc_lab.py) for the adapter.
