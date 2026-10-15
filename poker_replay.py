"""Validated, append-only local replay of completed poker hands.

Only the AI's encoded visible observations and actions are stored. Opaque
source groups balance sampling across people; they never become model inputs.
This module deliberately uses no torch/numpy imports so archives can be checked
independently of the training runtime. Change SCHEMA_VERSION if encoding changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sqlite3
from typing import Any


SCHEMA_VERSION = 1
FEATURE_DIM = 260
ACTION_DIM = 6
SOURCE_TYPES = ("human", "bot", "legacy_unknown")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512 or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string of at most 512 characters")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _decisions(values: Any, normalized_return: float) -> list[dict]:
    if not isinstance(values, list):
        raise ValueError("decisions must be a list")
    result = []
    for index, decision in enumerate(values):
        label = f"decision {index}"
        if not isinstance(decision, dict):
            raise ValueError(f"{label} must be a dictionary")
        allowed = {"state", "mask", "action", "return", "hand"}
        if set(decision) - allowed or not {"state", "mask", "action"} <= set(decision):
            raise ValueError(f"{label} has missing or unknown fields")
        state, mask = decision["state"], decision["mask"]
        if not isinstance(state, list) or len(state) != FEATURE_DIM:
            raise ValueError(f"{label} state must contain {FEATURE_DIM} features")
        state = [_finite(item, f"{label} feature") for item in state]
        if not isinstance(mask, list) or len(mask) != ACTION_DIM or any(type(item) is not bool for item in mask):
            raise ValueError(f"{label} mask must contain {ACTION_DIM} booleans")
        action = _integer(decision["action"], f"{label} action")
        if action >= ACTION_DIM or not mask[action]:
            raise ValueError(f"{label} records an illegal action")
        if "return" in decision:
            given_return = _finite(decision["return"], f"{label} return")
            if not math.isclose(given_return, normalized_return, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"{label} return disagrees with hand reward / stack_bb")
        if "hand" in decision:
            _integer(decision["hand"], f"{label} hand")
        result.append({"state": state, "mask": list(mask), "action": action, "return": normalized_return})
    return result


class ReplayStore:
    """One SQLite file, with immutable event IDs and per-person sampling.

    Sampling is with replacement: first choose an eligible source group
    uniformly, then a hand uniformly within that group. Only IDs are cached;
    observation JSON is read for the selected hands, not the entire archive.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._ids: dict[tuple[str, ...], dict[str, list[str]]] = {}
        self._data_version = -1
        try:
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._initialize()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> ReplayStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()
        self._ids.clear()

    def _initialize(self) -> None:
        expected = {"schema_version": str(SCHEMA_VERSION), "feature_dim": str(FEATURE_DIM),
                    "action_dim": str(ACTION_DIM)}
        try:
            # Serialize first-time schema creation with concurrent writers.
            self._connection.execute("BEGIN IMMEDIATE")
            tables = {row[0] for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")}
            if not tables:
                self._connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self._connection.executemany("INSERT INTO metadata (key, value) VALUES (?, ?)", expected.items())
                self._connection.execute("""
                    CREATE TABLE hands (
                        event_id TEXT PRIMARY KEY,
                        source_group TEXT NOT NULL,
                        source_type TEXT NOT NULL CHECK (source_type IN ('human', 'bot', 'legacy_unknown')),
                        hand_no INTEGER NOT NULL CHECK (hand_no >= 0),
                        reward_bb REAL NOT NULL,
                        stack_bb REAL NOT NULL CHECK (stack_bb > 0),
                        decision_count INTEGER NOT NULL CHECK (decision_count >= 0),
                        decisions_json TEXT NOT NULL
                    )
                """)
                self._connection.execute(
                    "CREATE INDEX hands_source ON hands (source_type, source_group, decision_count, event_id)")
            elif tables != {"metadata", "hands"}:
                raise ValueError("Replay database has an unrecognized schema")
            actual = dict(self._connection.execute("SELECT key, value FROM metadata"))
            if actual != expected:
                raise ValueError(f"Incompatible replay schema/encoding: {actual!r}; expected {expected!r}")
            expected_columns = {"event_id", "source_group", "source_type", "hand_no", "reward_bb",
                                "stack_bb", "decision_count", "decisions_json"}
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(hands)")}
            if columns != expected_columns:
                raise ValueError("Replay hands table has incompatible columns")
            self._connection.commit()
        except sqlite3.DatabaseError as exc:
            self._connection.rollback()
            raise ValueError(f"Invalid replay database: {exc}") from exc
        except Exception:
            self._connection.rollback()
            raise

    @staticmethod
    def _types(source_types) -> tuple[str, ...]:
        if isinstance(source_types, str):
            raise ValueError("source_types must be a sequence of source type strings")
        try:
            types = tuple(sorted(set(source_types)))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid source_types") from exc
        if any(item not in SOURCE_TYPES for item in types):
            raise ValueError(f"source_types must only contain {SOURCE_TYPES}")
        return types

    def add_hand(self, event_id: str, source_group: str, source_type: str, hand_no: int,
                 decisions: list[dict], reward_bb: float, stack_bb: float) -> bool:
        """Atomically append a completed hand. Identical retry returns False.

        Reusing an event ID for changed contents raises ValueError; it never
        overwrites previously recorded experience. Return is reward_bb/stack_bb.
        """
        event_id = _identifier(event_id, "event_id")
        source_group = _identifier(source_group, "source_group")
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SOURCE_TYPES}")
        hand_no = _integer(hand_no, "hand_no")
        reward_bb = _finite(reward_bb, "reward_bb")
        stack_bb = _finite(stack_bb, "stack_bb")
        if stack_bb <= 0:
            raise ValueError("stack_bb must be positive")
        normalized_return = reward_bb / stack_bb
        if not math.isfinite(normalized_return):
            raise ValueError("Normalized return must be finite")
        clean = _decisions(decisions, normalized_return)
        payload = json.dumps(clean, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        values = (event_id, source_group, source_type, hand_no, reward_bb, stack_bb, len(clean), payload)
        with self._connection:
            inserted = self._connection.execute("""
                INSERT INTO hands (event_id, source_group, source_type, hand_no, reward_bb, stack_bb,
                                   decision_count, decisions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(event_id) DO NOTHING
            """, values).rowcount
            if not inserted:
                previous = self._connection.execute("""
                    SELECT event_id, source_group, source_type, hand_no, reward_bb, stack_bb,
                           decision_count, decisions_json FROM hands WHERE event_id = ?
                """, (event_id,)).fetchone()
                if previous is None or tuple(previous) != values:
                    raise ValueError(f"Replay event {event_id!r} already exists with different contents")
                return False
        self._ids.clear()
        return True

    def stats(self, source_types=("human",)) -> dict:
        """Selected-source counts plus archive totals broken down by type."""
        types = self._types(source_types)
        by_type = {kind: {"hands": 0, "decisions": 0, "groups": 0, "trainable_hands": 0,
                          "trainable_groups": 0}
                   for kind in SOURCE_TYPES}
        for row in self._connection.execute("""
            SELECT source_type, COUNT(*) AS hands, SUM(decision_count) AS decisions,
                   COUNT(DISTINCT source_group) AS groups,
                   SUM(CASE WHEN decision_count > 0 THEN 1 ELSE 0 END) AS trainable_hands,
                   COUNT(DISTINCT CASE WHEN decision_count > 0 THEN source_group END) AS trainable_groups
            FROM hands GROUP BY source_type
        """):
            if row["source_type"] not in by_type:
                raise ValueError("Replay database contains an invalid source type")
            by_type[row["source_type"]] = {key: int(row[key]) for key in by_type[row["source_type"]]}
        if types:
            placeholders = ",".join("?" for _ in types)
            groups, trainable_groups = self._connection.execute(f"""
                SELECT COUNT(DISTINCT source_group),
                       COUNT(DISTINCT CASE WHEN decision_count > 0 THEN source_group END)
                FROM hands WHERE source_type IN ({placeholders})
            """, types).fetchone()
        else:
            groups = trainable_groups = 0
        return {"hands": sum(by_type[k]["hands"] for k in types),
                "decisions": sum(by_type[k]["decisions"] for k in types), "groups": groups,
                "trainable_hands": sum(by_type[k]["trainable_hands"] for k in types),
                "trainable_groups": int(trainable_groups),
                "by_type": by_type, "total_hands": sum(v["hands"] for v in by_type.values()),
                "total_decisions": sum(v["decisions"] for v in by_type.values())}

    def _eligible(self, types: tuple[str, ...]) -> dict[str, list[str]]:
        version = self._connection.execute("PRAGMA data_version").fetchone()[0]
        if version != self._data_version:
            self._ids.clear()
            self._data_version = version
        if types not in self._ids:
            groups: dict[str, list[str]] = {}
            if types:
                placeholders = ",".join("?" for _ in types)
                rows = self._connection.execute(f"""
                    SELECT source_group, event_id FROM hands
                    WHERE source_type IN ({placeholders}) AND decision_count > 0
                    ORDER BY source_group, event_id
                """, types)
                for group, event_id in rows:
                    groups.setdefault(group, []).append(event_id)
            self._ids[types] = groups
        return self._ids[types]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        try:
            reward_bb = _finite(row["reward_bb"], "stored reward_bb")
            stack_bb = _finite(row["stack_bb"], "stored stack_bb")
            if stack_bb <= 0:
                raise ValueError("Stored stack_bb must be positive")
            normalized_return = reward_bb / stack_bb
            if not math.isfinite(normalized_return):
                raise ValueError("Stored normalized return must be finite")
            decisions = _decisions(json.loads(row["decisions_json"]), normalized_return)
            if len(decisions) != row["decision_count"] or not decisions:
                raise ValueError("Stored decision count is inconsistent")
            return {"event_id": _identifier(row["event_id"], "stored event_id"),
                    "source_group": _identifier(row["source_group"], "stored source_group"),
                    "source_type": row["source_type"], "hand_no": _integer(row["hand_no"], "stored hand_no"),
                    "reward_bb": reward_bb, "stack_bb": stack_bb, "decisions": decisions}
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid replay event {row['event_id']!r}: {exc}") from exc

    def sample_hands(self, count: int, rng: random.Random, source_types=("human",)) -> list[dict]:
        """Sample eligible hands with replacement; return [] if none exist."""
        count = _integer(count, "count")
        types = self._types(source_types)
        if not count:
            return []
        eligible = self._eligible(types)
        if not eligible:
            return []
        groups = tuple(eligible)
        chosen = [rng.choice(eligible[rng.choice(groups)]) for _ in range(count)]
        unique_ids = list(dict.fromkeys(chosen))
        rows = {}
        # Stay below the variable limit even on older SQLite builds.
        for offset in range(0, len(unique_ids), 500):
            batch = unique_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            for row in self._connection.execute(
                    f"SELECT * FROM hands WHERE event_id IN ({placeholders})", batch):
                rows[row["event_id"]] = row
        if len(rows) != len(unique_ids):
            self._ids.clear()
            raise ValueError("Replay events disappeared while sampling; archive should be append-only")
        # Decode each occurrence separately so callers can mutate samples safely.
        return [self._decode(rows[event_id]) for event_id in chosen]
