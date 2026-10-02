"""Explicit prediction scope, saved with each model rather than guessed in UI."""
from datetime import timedelta
from .core import stamp
from .discovery import WEATHER_SERIES


def model_coverage(active):
    return (active or {}).get('coverage') or dict(series=WEATHER_SERIES,min_hours=5,max_hours=25,legacy=True)


def coverage_issue(active,series,at,close_time):
    c=model_coverage(active);hours=(stamp(close_time)-stamp(at)).total_seconds()/3600
    if series not in c['series']:
        return 'Unsupported market type: this model covers daily high temperatures in NYC, Chicago, Miami and Denver. This market needs its own trained model.'
    if hours>c['max_hours']:
        opens=stamp(close_time)-timedelta(hours=c['max_hours'])
        return f"Too early for this model: {hours:.1f} hours until close; supported range {c['min_hours']:.1f}–{c['max_hours']:.1f} hours. Coverage starts {opens.isoformat()}."
    if hours<c['min_hours']:
        return f"Too late for this model: {max(0,hours):.1f} hours until close; supported range {c['min_hours']:.1f}–{c['max_hours']:.1f} hours. Choose a later-closing market."
    return None


def training_coverage(partitions):
    common=set.intersection(*[{r['series'] for r in rows} for rows in partitions])
    bounds=[(min(r['hours_left'] for r in rows if r['series']==s),max(r['hours_left'] for r in rows if r['series']==s)) for rows in partitions for s in common]
    if not bounds:raise ValueError('No category is represented across train, validation and test')
    low=max(a for a,b in bounds);high=min(b for a,b in bounds)
    if low>high:raise ValueError('No time window is represented across train, validation and test')
    return dict(series=sorted(common),min_hours=low,max_hours=high,legacy=False)
