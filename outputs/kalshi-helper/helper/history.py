"""Real settled markets and archived hourly quotes, with no fabricated depth."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .core import stamp, utcnow
from .discovery import WEATHER_SERIES
from .kalshi import Kalshi
from .research_store import ResearchStore, digest


HORIZONS = (24, 12, 6)


def valid_quote(candle):
    try:
        values=[]
        for side in ('yes_bid','yes_ask'):
            part=candle[side]
            if 'close_dollars' in part:
                values.append(float(part['close_dollars']))
            else:
                # Historical endpoint uses decimal-dollar strings named close.
                # Reject numeric/integer legacy values instead of guessing cents.
                value=part['close']
                if not isinstance(value,str) or '.' not in value:
                    return None
                values.append(float(value))
        bid,ask=values
        if 0 <= bid <= ask <= 1:
            return bid, ask
    except (KeyError,TypeError,ValueError):
        pass
    return None


def extract_samples(series, market, candles, horizons=None):
    """All features use completed candles at/before the decision cutoff.

    Next-hour quotes are execution scenarios only, never model features.
    Final metadata and downloads occur later; original publication time is
    inferred from the exchange candle boundary, not falsely called ingested_at.
    """
    if market.get('result') not in ('yes','no') or market.get('status') not in ('settled','finalized'):
        return []
    if market.get('market_type','binary')!='binary':
        return []
    try:
        if float(market.get('notional_value_dollars','1'))!=1:
            return []
        if market.get('settlement_value_dollars') is not None and float(market['settlement_value_dollars'])!=int(market['result']=='yes'):
            return []
    except (TypeError,ValueError):
        return []
    if not market.get('settlement_ts'):
        return []
    if market.get('settlement_value_dollars') is not None and float(market['settlement_value_dollars']) not in (0,1):
        return []
    close = stamp(market['close_time'])
    settled = stamp(market['settlement_ts'])
    ordered = sorted(candles, key=lambda c: c['end_period_ts'])
    rows = []
    for horizon in (horizons or HORIZONS):
        cutoff = int((close-timedelta(hours=horizon)).timestamp())
        eligible = [c for c in ordered if c['end_period_ts'] <= cutoff and valid_quote(c)]
        if not eligible:
            continue
        candle = eligible[-1]
        timestamp = candle['end_period_ts']
        if cutoff-timestamp > 3600 or timestamp >= int(settled.timestamp()):
            continue
        bid,ask = valid_quote(candle)
        next_candles = [c for c in ordered if timestamp < c['end_period_ts'] <= timestamp+7200 and valid_quote(c)]
        execution = next_candles[0] if next_candles else None
        execution_quote = valid_quote(execution) if execution else None
        previous = [c for c in eligible[:-1] if timestamp-c['end_period_ts'] <= 7200]
        previous_mid = sum(valid_quote(previous[-1]))/2 if previous else (bid+ask)/2
        decision = datetime.fromtimestamp(timestamp,timezone.utc).isoformat()
        rows.append(dict(ticker=market['ticker'], event=market['event_ticker'], series=series,
            group=close.date().isoformat(), at=decision, feature_time=decision,
            settled_at=settled.isoformat(), close_time=close.isoformat(), horizon=horizon,
            hours_left=(close.timestamp()-timestamp)/3600, bid=bid,ask=ask,p=(bid+ask)/2,
            spread=ask-bid, prior_move=(bid+ask)/2-previous_mid,
            y=int(market['result']=='yes'), quote_volume=float(candle.get('volume_fp') or 0),
            execution_at=datetime.fromtimestamp(execution['end_period_ts'],timezone.utc).isoformat() if execution else None,
            execution_bid=execution_quote[0] if execution_quote else None,
            execution_ask=execution_quote[1] if execution_quote else None,
            source='kalshi_hourly_candle', synthetic=False))
    return rows


def download_history(store, days=100, series_list=None, progress=None):
    if not 45 <= int(days) <= 365:
        raise ValueError('Historical window must be 45–365 days')
    rs, client = ResearchStore(store), Kalshi(store)
    series_list = series_list or WEATHER_SERIES
    since = stamp(utcnow())-timedelta(days=days)
    cutoff = stamp(client.get('/historical/cutoff')['market_settled_ts'])
    existing = rs.history_keys()
    discovered, errors, listing_limits = {}, [], []
    for series in series_list:
        for endpoint in ('/markets','/historical/markets'):
            cursor = ''
            for page in range(3):
                params = dict(series_ticker=series, limit=1000)
                if endpoint == '/markets':
                    params['status']='settled'
                if cursor:
                    params['cursor']=cursor
                response = client.get(endpoint,**params)
                for market in response['markets']:
                    if stamp(market['close_time']) >= since and market.get('result') in ('yes','no'):
                        discovered[market['ticker']] = (series, market)
                cursor=response.get('cursor','')
                if not cursor:
                    break
                # Series lists are returned newest-first by the current API;
                # retain a visible bounded-listing caveat instead of asserting census completeness.
                if response['markets'] and max(stamp(m['close_time']) for m in response['markets']) < since:
                    listing_limits.append(f'{series}: stopped after an entirely older page')
                    break
            if cursor and page == 2:
                listing_limits.append(f'{series}: listing reached three-page limit')
    pending = [(s,m) for s,m in discovered.values() if m['ticker'] not in existing]
    recent, archived = defaultdict(list), []
    for series, market in pending:
        if market.get('settlement_ts') and stamp(market['settlement_ts']) >= cutoff:
            recent[market['close_time'][:10]].append((series,market))
        else:
            archived.append((series,market))
    done = 0
    for day, pairs in sorted(recent.items()):
        for offset in range(0,len(pairs),60):
            batch=pairs[offset:offset+60]
            try:
                minimum=min(int(stamp(m['close_time']).timestamp()) for _,m in batch)-27*3600
                maximum=max(int(stamp(m['close_time']).timestamp()) for _,m in batch)-3*3600
                response=client.get('/markets/candlesticks',market_tickers=','.join(m['ticker'] for _,m in batch),
                                    start_ts=minimum,end_ts=maximum,period_interval=60)
                by_ticker={x['market_ticker']:x['candlesticks'] for x in response['markets']}
                for series,market in batch:
                    candles=by_ticker.get(market['ticker'],[])
                    if candles:
                        rs.add_history(series,market,candles)
                    else:
                        errors.append(f"{market['ticker']}: no recent candles returned")
                    done+=1
            except (RuntimeError,ValueError,KeyError) as exc:
                errors.append(f'{day}: {exc}')
                done+=len(batch)
            if progress:
                progress(f'Downloaded {done} of {len(pending)} markets',done,len(pending))
    for series,market in sorted(archived,key=lambda x:x[1]['close_time'],reverse=True):
        try:
            close=int(stamp(market['close_time']).timestamp())
            response=client.get('/historical/markets/'+market['ticker']+'/candlesticks',
                                start_ts=close-27*3600,end_ts=close-3*3600,period_interval=60)
            if response['candlesticks']:
                rs.add_history(series,market,response['candlesticks'])
            else:
                errors.append(f"{market['ticker']}: no historical candles returned")
        except (RuntimeError,ValueError,KeyError) as exc:
            errors.append(f"{market['ticker']}: {exc}")
        done+=1
        if progress:
            progress(f'Downloaded {done} of {len(pending)} markets',done,len(pending))
    rows=sample_rows(store)
    manifest=dict(at=utcnow(),requested_days=days,series=series_list,discovered_markets=len(discovered),
        stored_markets=len(rs.history_keys()),samples=len(rows),days=len({r['group'] for r in rows}),
        errors=errors,listing_limits=listing_limits,data_hash=digest(rows),synthetic=False,
        limitations=['Hourly bid/ask quotes have no historical depth or queue position.',
          'Original quote availability is inferred from candle end times; downloaded_at is stored separately.',
          'Archived rule revisions and the completeness of the listing cannot be independently established.'])
    rs.set('history_manifest',manifest)
    return manifest


def sample_rows(store,horizons=None):
    rows=[]
    for item in ResearchStore(store).history():
        rows.extend(extract_samples(item['series'],item['market'],item['candles'],horizons))
    return sorted(rows,key=lambda r:(r['at'],r['ticker']))
