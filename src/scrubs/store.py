import json
import sqlite3
import typing
from dataclasses import dataclass
from hashlib import blake2b
from typing import Type, TypeVar


def hashobj(obj: str):
    return blake2b(obj.encode("utf8")).hexdigest()


@dataclass
class RawObject:
    id: str
    type: str
    object: str


COMPACT_SEPARATORS = (",", ":")


def compact_dumps(obj) -> str:
    return json.dumps(obj, separators=COMPACT_SEPARATORS)


ObjectID = str


class Store:
    def configure_db(self):
        # self.db.autocommit = True

        # https://kerkour.com/sqlite-for-servers
        self.db.executescript("""
        PRAGMA journal_mode = wal2;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = 134217728;
        PRAGMA foreign_keys = true;
        PRAGMA busy_timeout = 5000;
        """)

    def ensure_schema(self):
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS objects (
                id text PRIMARY KEY NOT NULL,
                type text NOT NULL,
                object text NOT NULL
        )
        """)

        self.db.execute("""
        CREATE TABLE IF NOT EXISTS caches (
                operation TEXT NOT NULL,
                result TEXT NOT NULL,
                PRIMARY KEY (operation)
        )
        """)
        self.db.commit()

    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path)
        self.configure_db()
        self.ensure_schema()

    # Objects
    def insert(self, type: str, obj: str) -> ObjectID:
        rt = compact_dumps(json.loads(obj))
        assert rt == obj, f"Object must round-trip: {rt!r} != {obj!r}"

        id = hashobj(obj)
        self.db.execute(
            "INSERT OR IGNORE INTO objects (id, type, object) VALUES (?, ?, ?)",
            (id, type, obj),
        )
        self.db.commit()
        return id

    def get(self, id: ObjectID, type: str | None = None) -> RawObject | None:
        cur = self.db.execute(
            "SELECT id, type, object FROM objects WHERE id = ?", (id,)
        )
        row = cur.fetchone()
        if row is None or (type is not None and row[1] != type):
            return None
        return RawObject(*row)

    def fetch(self, id: ObjectID, type: str | None = None) -> RawObject:
        obj = self.get(id, type)
        if obj is None:
            msg = f"Cannot find object: {id}"
            if type is not None:
                msg = msg + f" (of type {type})"
            raise KeyError(msg)
        return obj

    def object_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]

    # Cache: (operation, result)
    def put_cache(self, operation: ObjectID, result: ObjectID):
        self.db.execute(
            "INSERT OR ABORT INTO caches (operation, result) VALUES (?, ?)",
            (operation, result),
        )
        self.db.commit()

    def has_cache(self, operation: ObjectID) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM caches WHERE operation = ?", (operation,)
            ).fetchone()
            is not None
        )

    def get_cache(self, operation: ObjectID) -> ObjectID | None:
        row = self.db.execute(
            "SELECT result FROM caches WHERE operation = ?",
            (operation,),
        ).fetchone()
        if row is None:
            return None
        return row[0]

    def fetch_cache(self, operation: ObjectID) -> ObjectID:
        got = self.get_cache(operation)
        if got is None:
            raise KeyError(f"No cache result for object {operation}!")
        return got
