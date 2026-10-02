"""Nominal candle horizons and physical forecast hours are distinct quantities."""
PRICE_COHORT='Nominal 12h, 18h and 24h candles; midpoint 5–95 cents'

def price_primary(row):
    return row.get('horizon') in (12,18,24) and .05<=row['p']<=.95
