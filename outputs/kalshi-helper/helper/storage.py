"""Append-only input journal and durable run reports, using the standard library."""
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
from contextlib import contextmanager

from .core import plain, utcnow, validate_record


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "research.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS inputs (
                    id TEXT PRIMARY KEY, dataset TEXT NOT NULL, available_at TEXT NOT NULL,
                    ticker TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    imported_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS input_time ON inputs(dataset, available_at);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL,
                    status TEXT NOT NULL, report TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS fetches (
                    id INTEGER PRIMARY KEY, fetched_at TEXT NOT NULL,
                    url TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def append(self, records, dataset):
        rows = []
        for r in records:
            validate_record(r)
            payload = json.dumps(plain(r), sort_keys=True, allow_nan=False)
            digest = hashlib.sha256((dataset + payload).encode()).hexdigest()
            rows.append((digest, dataset, r["available_at"], r["ticker"], r["type"], payload, utcnow()))
        with self.connect() as db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO inputs VALUES (?,?,?,?,?,?,?)", rows)
            return db.total_changes-before

    def records(self, dataset):
        # Sorting by parsed time in replay supports different timezone offsets.
        with self.connect() as db:
            return [json.loads(x[0]) for x in db.execute(
                "SELECT payload FROM inputs WHERE dataset=? ORDER BY rowid", (dataset,))]

    def save_run(self, run_id, mode, status, report):
        with self.connect() as db:
            db.execute("""INSERT INTO runs VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                       mode=excluded.mode, status=excluded.status, report=excluded.report""",
                       (run_id, utcnow(), mode, status, json.dumps(plain(report), allow_nan=False)))

    def runs(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 50").fetchall()
        return [dict(id=r["id"], created_at=r["created_at"], mode=r["mode"], status=r["status"],
                     report=json.loads(r["report"])) for r in rows]

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT report FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise ValueError("Unknown run")
        return json.loads(row[0])

    def datasets(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT dataset, COUNT(*) AS records,
                MIN(available_at) AS first_at, MAX(available_at) AS last_at
                FROM inputs GROUP BY dataset ORDER BY dataset""")]

    def raw(self, url, content):
        digest = hashlib.sha256(content).hexdigest()
        folder = self.root / "raw" / utcnow()[:10]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (uuid.uuid4().hex + ".json")
        path.write_bytes(content)
        with self.connect() as db:
            db.execute("INSERT INTO fetches(fetched_at,url,path,sha256) VALUES (?,?,?,?)",
                       (utcnow(), url, str(path.relative_to(self.root)), digest))

    def interrupt_previous_runs(self):
        with self.connect() as db:
            db.execute("UPDATE runs SET status='interrupted' WHERE status='running'")
