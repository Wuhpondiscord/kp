"""Automatic category discovery and a persisted, human-readable market board."""
from datetime import timedelta

from .core import dec, stamp, utcnow
from .kalshi import Kalshi, normalize_market
from .research_store import ResearchStore


WEATHER_SERIES = ['KXHIGHNY', 'KXHIGHCHI', 'KXHIGHMIA', 'KXHIGHDEN']
CATEGORIES = {
    'weather': 'Daily weather', 'climate': 'Seasonal weather', 'mentions': 'Mentions',
    'posts': 'Post counts', 'economics': 'Economy & travel', 'entertainment': 'Entertainment',
}


def category_of(series):
    categories = [str(x).lower() for x in (series.get('categories') or [])]
    text = ' '.join([series.get('title') or '', series.get('ticker') or '', *(series.get('tags') or [])]).lower()
    if 'sports' in categories:
        return None
    if 'mentions' in categories or 'mention' in text or 'say during' in text:
        return 'mentions'
    if any(word in text for word in ('number of posts', 'post count', 'tweets', 'how many posts')):
        return 'posts'
    if 'climate and weather' in categories:
        return 'weather' if series.get('frequency') == 'daily' else 'climate'
    if any(word in text for word in ('tsa', 'gas price', 'inflation', 'cpi', 'gdp', 'jobs report')):
        return 'economics'
    if any(word in text for word in ('box office', 'rotten tomatoes', 'spotify', 'billboard')):
        return 'entertainment'
    return None


def series_fee(series):
    if series.get('fee_type') not in ('quadratic', 'quadratic_with_maker_fees'):
        return None
    multiplier = series.get('fee_multiplier')
    if multiplier is None:
        return None
    return str(dec('0.07')*dec(multiplier))


def event_fee(client,series,event_ticker,cache):
    if event_ticker not in cache:
        cache[event_ticker]=client.get('/events/'+event_ticker)['event']
    event=cache[event_ticker]
    metadata=dict(series)
    if event.get('fee_type_override'):
        metadata['fee_type']=event['fee_type_override']
    if event.get('fee_multiplier_override') is not None:
        metadata['fee_multiplier']=event['fee_multiplier_override']
    return series_fee(metadata)


def refresh_board(store, categories=None, progress=None):
    categories = categories or ['weather']
    if not set(categories) <= set(CATEGORIES):
        raise ValueError('Unknown market category')
    rs, client = ResearchStore(store), Kalshi(store)
    catalog = rs.get('series_catalog')
    if not catalog or (stamp(utcnow())-stamp(catalog['at'])).total_seconds() > 21600:
        listed = client.get('/series')['series']
        catalog = dict(at=utcnow(), series=listed)
        rs.set('series_catalog', catalog)
    selected = [s for s in catalog['series'] if category_of(s) in categories]
    selected.sort(key=lambda s: (s['ticker'] not in WEATHER_SERIES, -(float(s.get('volume_fp') or 0))))
    # Bounded discovery per category; search keeps the complete series catalog available.
    counts, chosen = {}, []
    for series in selected:
        category = category_of(series)
        if counts.get(category, 0) < (4 if category == 'weather' else 30):
            chosen.append(series)
            counts[category] = counts.get(category, 0)+1
    cards, errors, events = [], [], {}
    loaded, scanned = {}, 0
    for index, series in enumerate(chosen):
        category = category_of(series)
        if loaded.get(category, 0) >= (4 if category == 'weather' else 3):
            continue
        scanned += 1
        before = len(cards)
        if progress:
            progress(f"Loading {series['title']}", index, len(chosen))
        try:
            for market in client.markets(series['ticker'], 18):
                if stamp(market['close_time']) <= stamp(utcnow()):
                    continue
                at = utcnow()
                bid, ask = market.get('yes_bid_dollars'), market.get('yes_ask_dollars')
                if bid is None or ask is None:
                    continue
                bid, ask = float(bid), float(ask)
                if not 0 <= bid <= ask <= 1:
                    continue
                try:
                    rate = event_fee(client,series,market['event_ticker'],events)
                except (RuntimeError,ValueError,KeyError) as exc:
                    rate=None
                    errors.append(f"{market['event_ticker']}: fee metadata unavailable: {exc}")
                store.append([normalize_market(market, at, rate)], 'market-discovery')
                cards.append(dict(ticker=market['ticker'], event_ticker=market['event_ticker'],
                    title=market.get('title',market['ticker']), subtitle=market.get('yes_sub_title',''),
                    series=series['ticker'], series_title=series['title'], category=category_of(series),
                    close_time=market['close_time'], status=market['status'],tradeable=market['status'] in ('active','open'),bid=bid, ask=ask, midpoint=(bid+ask)/2,
                    no_ask=1-bid, spread=ask-bid, volume=float(market.get('volume_fp') or 0),
                    at=at, fee_rate=rate, fee_version=f"series-{series['ticker']}-{catalog['at']}",
                    rules=market.get('rules_primary',''), rules_secondary=market.get('rules_secondary',''),
                    sources=series.get('settlement_sources',[])))
        except (RuntimeError,ValueError,KeyError) as exc:
            errors.append(f"{series['ticker']}: {exc}")
        if len(cards) > before:
            loaded[category] = loaded.get(category, 0) + 1
    result = dict(at=utcnow(), categories=categories, markets=cards, errors=errors,
                  scanned_series=scanned, available_series=len(selected))
    if not cards:
        if errors:
            raise RuntimeError('No markets loaded: '+'; '.join(errors[:2]))
    rs.set('board', result)
    return result
