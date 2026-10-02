"""Versioned research data, separate from the execution journal."""
import hashlib
import json

from .core import utcnow


class ResearchStore:
    def __init__(self, store):
        self.store = store
        with store.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS app_state (name TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS history_markets (
                    ticker TEXT PRIMARY KEY, series TEXT NOT NULL, close_time TEXT NOT NULL,
                    market TEXT NOT NULL, candles TEXT NOT NULL, downloaded_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS history_series_close ON history_markets(series, close_time);
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, report TEXT NOT NULL);
            ''')

    def set(self, name, payload):
        with self.store.connect() as db:
            db.execute('INSERT INTO app_state VALUES (?,?) ON CONFLICT(name) DO UPDATE SET payload=excluded.payload',
                       (name, json.dumps(payload, allow_nan=False)))

    def get(self, name, default=None):
        with self.store.connect() as db:
            row = db.execute('SELECT payload FROM app_state WHERE name=?', (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def add_history(self, series, market, candles):
        with self.store.connect() as db:
            db.execute('INSERT INTO history_markets VALUES (?,?,?,?,?,?) ON CONFLICT(ticker) DO NOTHING',
                       (market['ticker'], series, market['close_time'], json.dumps(market), json.dumps(candles), utcnow()))

    def history(self):
        with self.store.connect() as db:
            return [dict(series=r['series'], market=json.loads(r['market']), candles=json.loads(r['candles']),
                         downloaded_at=r['downloaded_at']) for r in db.execute(
                'SELECT * FROM history_markets ORDER BY close_time,ticker')]

    def history_keys(self):
        with self.store.connect() as db:
            return {r[0] for r in db.execute('SELECT ticker FROM history_markets')}

    def save_experiment(self, report):
        with self.store.connect() as db:
            db.execute('INSERT INTO experiments VALUES (?,?,?) ON CONFLICT(id) DO NOTHING',
                       (report['id'], utcnow(), json.dumps(report, allow_nan=False)))
        self.set('latest_experiment', report['id'])

    def experiment(self, experiment_id=None):
        experiment_id = experiment_id or self.get('latest_experiment')
        with self.store.connect() as db:
            row = db.execute('SELECT report FROM experiments WHERE id=?', (experiment_id,)).fetchone()
        return json.loads(row[0]) if row else None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
