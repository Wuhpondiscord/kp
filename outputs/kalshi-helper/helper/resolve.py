"""Resolve public links, event names and series before starting a paper worker."""
import re
from urllib.parse import urlparse, unquote
from .core import stamp, utcnow
from .kalshi import Kalshi, KalshiError, normalize_market
from .discovery import category_of, series_fee, event_fee
from .research_store import ResearchStore


def parse_reference(value):
    value = str(value).strip()
    if '://' in value or '/' in value:
        parsed = urlparse(value if '://' in value else 'https://' + value)
        if parsed.scheme != 'https' or parsed.hostname not in ('kalshi.com', 'www.kalshi.com') or parsed.port or parsed.username:
            raise ValueError('Paste a https://kalshi.com/markets/… link or a market ticker.')
        parts = [unquote(x) for x in parsed.path.strip('/').split('/')]
        if len(parts) not in (2, 3, 4) or parts[0] != 'markets':
            raise ValueError('This is not a Kalshi market link. Open a market and copy its address.')
        value = parts[3] if len(parts) == 4 else parts[1]
    value = value.upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9_.-]{0,119}', value):
        raise ValueError('The market link or ticker is invalid.')
    return value


def resolve_markets(store, reference, client=None):
    client = client or Kalshi(store)
    ticker = parse_reference(reference)
    event = {}
    if '-' not in ticker:
        series = client.get('/series/' + ticker)['series']
        markets = client.markets(ticker, 49)
        kind = 'series'
    else:
        try:
            response = client.get('/events/' + ticker, with_nested_markets='true')
            event = response['event']
            markets = event.get('markets') or response.get('markets') or []
            kind = 'event'
        except KalshiError as exc:
            if exc.status != 404:
                raise
            markets = [client.market(ticker)]
            event = client.get('/events/' + markets[0]['event_ticker'])['event']
            kind = 'contract'
        series = client.get('/series/' + event['series_ticker'])['series']
    if not markets:
        raise ValueError('No open contracts were found for this series/event. Try a current market link.')
    if len(markets) > 48:
        raise ValueError('This series has more than 48 contracts. Paste a specific event link instead.')
    fee_metadata = dict(series)
    if event.get('fee_type_override'):
        fee_metadata['fee_type'] = event['fee_type_override']
    if event.get('fee_multiplier_override') is not None:
        fee_metadata['fee_multiplier'] = event['fee_multiplier_override']
    rate = series_fee(fee_metadata)
    at = utcnow()
    cards = []
    event_cache={m['event_ticker']:event for m in markets} if event else {}
    for market in markets:
        rate=event_fee(client,series,market['event_ticker'],event_cache)
        # Persist raw, verified metadata; missing quotes stay missing, not 0/100%.
        store.append([normalize_market(market, at, rate)], 'market-discovery')
        bid, ask = market.get('yes_bid_dollars'), market.get('yes_ask_dollars')
        bid, ask = (float(bid), float(ask)) if bid is not None and ask is not None else (None, None)
        valid = bid is not None and 0 <= bid <= ask <= 1
        cards.append(dict(ticker=market['ticker'],event_ticker=market['event_ticker'],
            title=market.get('title',market['ticker']),subtitle=market.get('yes_sub_title',''),
            series=series['ticker'],category=category_of(series),status=market['status'],
            close_time=market['close_time'],at=at,bid=bid if valid else None,ask=ask if valid else None,
            midpoint=(bid+ask)/2 if valid else None,no_ask=1-bid if valid else None,
            spread=ask-bid if valid else None,fee_rate=rate,rules=market.get('rules_primary',''),
            rules_secondary=market.get('rules_secondary',''),sources=event.get('settlement_sources') or series.get('settlement_sources',[]),
            tradeable=market['status'] in ('open','active') and stamp(market['close_time'])>stamp(at)))
    result = dict(at=at,reference=reference,resolved_as=kind,title=event.get('title') or series.get('title',ticker),
                  markets=cards,tickers=[m['ticker'] for m in cards if m['tradeable']])
    return result
