# BRTI capture setup — local research only

The project now has a read-only, authenticated historical BRTI capture command.
It supports RSA and Ed25519 signatures, fixes the destination to Kalshi's BRTI
history endpoint, refuses redirects, and saves response hashes and local retrieval
timestamps. It cannot place orders. No credentials or signatures are persisted.

**An API key is optional for the project.** Public Kalshi quotes/outcomes, Coinbase
proxy features, saved-data training and paper research work without one. To rerun
the development comparison now, use `python -m helper.btc_calibration`; this needs
neither network access nor credentials. The original `helper.btc_research` collector
also uses public data, though online downloads need network access.

Without credentials, the BRTI command now exits successfully with
`optional_source_not_configured`: it makes no request and writes no fake data.
Partial configuration is reported explicitly. Add `--require-brti` only for a job
that must fail when this optional source is unavailable. Once both credential
variables are configured, the same command attempts BRTI capture automatically;
authentication/network failures remain explicit.

**Current status:** no Kalshi environment credentials were configured in the agent
session. Real entitlement,
response schema and coverage have **not** been tested. No BRTI data has been fetched,
and no new model score or profitability result is claimed.

## Configure locally

Use your Kalshi API key ID and downloaded private key, stored outside this repository.
Set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` in the terminal that will run
the command. The second variable contains a local PEM **file path**, not the key
contents. Do not paste the key into chat or put it in the shared Hugging Face demo.
The capture command requires the optional Python `cryptography` package; the
ordinary app and offline model research do not depend on it.

Install that optional dependency with `python -m pip install -r requirements-brti.txt`.

From outputs/kalshi-helper, after configuration:

```powershell
python -m helper.brti_capture --hour 2026-09-20T12:00:00Z
```

The requested timestamp must be exactly an hour boundary. This initial development
hour avoids the original September 25–30 test period. The command makes one GET
request with a 40-second timeout, with no retries or silent proxy substitution.
Output defaults to ignored `data/brti-private`. Each capture has a unique raw JSON
and metadata JSON; metadata contains no authentication headers. Inspect errors
locally: 401/403 means credentials or entitlement need attention; an upstream 503
can have several causes and is not automatically evidence of missing entitlement.

Verify a saved capture without credentials or network access:

```powershell
python -m helper.brti_capture --verify data/brti-private/CAPTURE.meta.json
```

Replace CAPTURE with the generated filename. The verifier checks the byte length
and SHA-256 hash. Private keys and raw capture extensions are also git-ignored.

## Why raw capture is not yet an importer into training

The documented response wraps CF data inside `data`. Until a real entitled response
is available, the code only validates that envelope and records
`status=captured_unvalidated` and `usable_for_training=false`. An empty payload
does not count as valid tick data. This avoids guessing an upstream schema and
silently training on incomplete or misinterpreted values.

Next, inspect a genuine response and add a schema-specific tick adapter with real
redacted fixtures. Validate index identity, units, timestamps, duplicates, revision
handling, coverage and final-minute boundaries. Then measure Coinbase/BRTI basis
on matching development timestamps before retraining. Do not interpolate away
missing settlement ticks or treat retrieval time as historical availability.

The historical API can delay recent values by up to 15 minutes. It is suitable for
retrospective source comparisons, **not** proof those values were available for a
five-minute live trade. Prospective inference will need the live feed and measured
receipt timestamps.

References, checked October 4, 2026:

* [Kalshi passthrough and entitlement](https://docs.kalshi.com/cfbenchmarks/rest-passthrough)
* [Kalshi authentication](https://docs.kalshi.com/getting_started/quick_start_authenticated_requests)
* [CF history endpoint and delay](https://docs.cfbenchmarks.com/api/rest/historical-values/)

Tests verify RSA and Ed25519 signatures with generated ephemeral keys, UTC hour
normalization, credential absence, redirect refusal, entitlement errors, raw-file
tamper detection and the unvalidated-data status. Network responses are mocked;
passing these tests does not certify live Kalshi access.
