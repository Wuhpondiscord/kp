"""Read-only public-data client. No signing keys, balances or order endpoints."""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from .core import utcnow, validate_record


BASE = "https://external-api.kalshi.com/trade-api/v2"


class KalshiError(RuntimeError):
    def __init__(self, status, path):
        self.status = status
        super().__init__(f"Kalshi GET failed: HTTP {status} for {path}")


class Kalshi:
    def __init__(self, store, interval=0.3):
        self.store, self.interval, self.last_request = store, interval, 0.0

    def get(self, path, **params):
        url = BASE + path + ("?" + urlencode(params) if params else "")
        for attempt in range(3):
            time.sleep(max(0, self.interval-(time.monotonic()-self.last_request)))
            self.last_request = time.monotonic()
            try:
                request = Request(url, headers={"User-Agent": "KalshiHelperResearch/0.1", "Accept": "application/json"})
                with urlopen(request, timeout=15) as response:
                    body = response.read()
                self.store.raw(url, body)
                return json.loads(body)
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise KalshiError(exc.code, path) from exc
                time.sleep(min(10, 2**attempt))
            except (URLError, TimeoutError, ConnectionError) as exc:
                reason=getattr(exc,'reason',exc)
                if getattr(reason,'winerror',None)==10013 or '10013' in str(reason):
                    raise RuntimeError('Windows blocked this app’s network access (10013). Restart BetCheck from Start Helper.cmd in a normal terminal, then retry Update prices. No API key is required. If it persists, check the Python app’s network permissions; do not disable your firewall.') from exc
                if attempt == 2:
                    raise RuntimeError(f"Kalshi network request failed: {getattr(exc,'reason',str(exc))}") from exc
                time.sleep(2**attempt)
        raise RuntimeError("Request retry budget exhausted")

    def markets(self, series, limit=20):
        result, cursor, seen, tickers = [], "", set(), set()
        while len(result) < limit:
            params = dict(series_ticker=series, status="open", limit=min(1000, limit-len(result)))
            if cursor:
                params["cursor"] = cursor
            response = self.get("/markets", **params)
            for market in response['markets']:
                if market['ticker'] not in tickers:
                    result.append(market);tickers.add(market['ticker'])
            cursor = response.get("cursor", "")
            if not cursor:
                break
            if cursor in seen:
                raise RuntimeError("Repeated pagination cursor")
            seen.add(cursor)
        return result[:limit]

    def market(self, ticker):
        return self.get("/markets/" + quote(ticker, safe=""))["market"]

    def book(self, ticker):
        response = self.get("/markets/" + quote(ticker, safe="") + "/orderbook", depth=20)
        return normalize_book(ticker, response, utcnow())


def normalize_book(ticker, response, at):
    book = response.get("orderbook_fp")
    if book is None:
        raise ValueError("Expected orderbook_fp; refusing to guess legacy price units")
    return validate_record(dict(type="book", ticker=ticker, available_at=at, observed_at=at,
                                yes=book["yes_dollars"] or [], no=book["no_dollars"] or []))


def normalize_market(market, at, fee_rate=None):
    if market.get("market_type", "binary") != "binary":
        raise ValueError("Only binary $1 contracts are supported")
    if str(market.get("notional_value_dollars", "1.0000")) not in ("1", "1.0", "1.00", "1.0000"):
        raise ValueError("Only $1 notional contracts are supported")
    return validate_record(dict(type="market", ticker=market["ticker"], available_at=at,
        event_ticker=market["event_ticker"], title=market.get("title", market["ticker"]),
        close_time=market["close_time"], status=market["status"], fee_rate=fee_rate,
        fee_version="user-supplied-flat-rate/conservative-cent-rounding-v1" if fee_rate is not None else None,
        rules=market.get("rules_primary", ""), rules_secondary=market.get("rules_secondary", ""),
        raw_metadata=market))


def normalize_outcome(market, at):
    if market.get("status") not in ("settled", "finalized"):
        return None
    payout = market.get("settlement_value_dollars")
    if payout is None:
        if market.get("result") not in ("yes", "no"):
            return None
        payout = "1" if market["result"] == "yes" else "0"
    return validate_record(dict(type="outcome", ticker=market["ticker"], available_at=at,
                                settled_at=market.get("settlement_ts") or at, yes_payout=payout))


def collect_once(store, series, limit=20, dataset="collected"):
    if not series or not 1 <= limit <= 100:
        raise ValueError("Provide a series and a market limit between 1 and 100")
    client, records, errors = Kalshi(store), [], []
    for market in client.markets(series, limit):
        try:
            records.append(normalize_market(market, utcnow()))
            records.append(client.book(market["ticker"]))
        except (RuntimeError, ValueError, KeyError) as exc:
            errors.append(f"{market.get('ticker')}: {exc}")
    count = store.append(records, dataset) if records else 0
    return dict(records=count, markets=sum(r["type"] == "market" for r in records), errors=errors,
                dataset=dataset, collected_at=utcnow())
