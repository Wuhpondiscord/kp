# Browser and regression checks — September 21, 2026

Tested the actual local page at a roughly 1268×714 window, including neural training, cancellation, live Kalshi discovery, the model-available filter, contract recommendation details and paper-practice startup.

## Issues reproduced and fixed

1. **Practice Start button unreachable on a shorter screen.** The tall sticky sidebar held the lower controls below the viewport; two clicks failed. The practice panel now scrolls normally with the page. The same browser flow then successfully started a real-market, simulated-money session.
2. **Cancellation during evaluation could publish a new model.** A regression test reproduced this. Cancellation is now checked after blend selection and again before report/model publication. Early cancellation in the browser preserved the previous active model; a subsequent normal training run completed successfully.
3. **Malformed training features could pass validation.** Tests supplied probabilities outside [0,1], probabilities inconsistent with quote midpoint, inconsistent spreads and negative horizons. They previously passed. They now raise explicit errors. All 26,007 existing real archive rows pass the stricter checks.
4. **Duplicate Windows server startup could share the port.** Two listeners were observed during restart. The server now claims an exclusive socket before recovering jobs or sessions. A duplicate-launch test proves it cannot bind or modify the running job record.
5. **Rounding overstated certainty and hid small losses.** A live contract showed 0% instead of a small positive model probability, and 0¢ for a slightly negative edge. Probability displays now use <1% / >99% for near-boundary values, and price/edge displays retain tenths of a cent. Nine JavaScript assertions verify the formatter behavior.

## Results and limits

- 106 Python tests passed; both JavaScript files passed syntax checks; nine frontend formatting assertions passed.
- Live discovery initially exposed a session network restriction. After granting network access and restarting the app, the browser displayed “Prices updated” with current real markets.
- A live Denver contract opened successfully, displaying the residual model's validation/test scores, fees, recommendation and input sensitivity. The recommendation was to pass. No browser console errors were recorded in that flow.
- Browser cancellation preserved `train-738dd22ecde6`; another default training run completed with 27 epochs and best checkpoint 13.
- Live paper startup was tested with automatic daily-weather discovery, experimental ML and $1,000 simulated money. The smoke test is stopped after inspection; it is not a full-day profitability or settlement test. Exact session results are in the accompanying JSON evidence.

The changes strengthen reliability and presentation. They do not improve the previously reported model scores or establish a tradable edge. Late-day contracts can have one-sided books or be outside the trained time window, so a functioning practice run can legitimately produce no fills.

Reproduce software checks from this folder:

```powershell
python -m unittest discover -s tests -v
node tests/frontend_checks.cjs
node --check helper/static/main.js
node --check helper/static/training.js
```
