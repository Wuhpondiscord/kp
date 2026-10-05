"""Validate BTC execution inputs before forecasting; do not infer availability."""
import math
from .core import stamp
from .btc_research import execution_rows


def validated_execution_rows(rows):
    rows = list(rows)
    if len({r['ticker'] for r in rows}) != len(rows):
        raise ValueError('Duplicate BTC contract')
    for row in rows:
        execution = row.get('execution_at')
        if not execution:
            continue  # Existing replay reports missing executable quotes separately.
        at = stamp(execution)
        if not stamp(row['at']) <= at < stamp(row['close_time']) <= stamp(row['settled_at']):
            raise ValueError('Invalid BTC execution chronology')
        feature = row.get('execution_spot')
        if not isinstance(feature, dict): raise ValueError('Missing refreshed BTC inputs')
        if not stamp(feature['feature_time']) <= stamp(feature['feature_available_at']) <= at:
            raise ValueError('Future BTC execution features')
        if (at-stamp(feature['feature_time'])).total_seconds() > 65:
            raise ValueError('Stale BTC execution features')
        x = feature.get('x', [])
        if len(x) != 10 or not all(math.isfinite(float(v)) for v in x):
            raise ValueError('Invalid BTC feature vector')
        p = feature.get('diffusion')
        if p is None or not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
            raise ValueError('Invalid BTC diffusion probability')
    return execution_rows(rows)
