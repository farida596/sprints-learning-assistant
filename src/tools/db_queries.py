"""Read-only learner database tools for the Advisor agent."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from src.config import LEARNER_CSV, LEARNER_DB
from src.logic import parse_employers, unable_to_verify

COLUMNS = ["learner_id", "name", "bootcamp_track", "completed_modules", "python_assignment_score",
           "target_salary_egp", "preferred_employers", "readiness_status"]
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.I)


def build_database(csv_path: Path | str = LEARNER_CSV, db_path: Path | str = LEARNER_DB) -> int:
    """Create or rebuild the SQLite database from the starter CSV."""
    csv_path, db_path = Path(csv_path), Path(db_path)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS learners")
        con.execute("""CREATE TABLE learners (
            learner_id TEXT PRIMARY KEY, name TEXT NOT NULL, bootcamp_track TEXT,
            completed_modules INTEGER, python_assignment_score INTEGER,
            target_salary_egp INTEGER, preferred_employers TEXT, readiness_status TEXT)""")
        con.executemany(
            "INSERT INTO learners VALUES (?,?,?,?,?,?,?,?)",
            [(r["learner_id"], r["name"], r["bootcamp_track"], int(r["completed_modules"]),
              int(r["python_assignment_score"]), int(r["target_salary_egp"]),
              r["preferred_employers"], r["readiness_status"]) for r in rows])
        con.commit()
    finally:
        con.close()
    return len(rows)


def _connect(db_path: Path | str) -> sqlite3.Connection:
    """Open the database read-only."""
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def validate_select(sql: str) -> str:
    """Validate one read-only SELECT statement."""
    cleaned = (sql or "").strip().rstrip(";").strip()
    if not cleaned.lower().startswith("select"):
        raise ValueError("only SELECT statements are allowed")
    if ";" in cleaned or _FORBIDDEN.search(cleaned):
        raise ValueError("multiple statements or write keywords are not allowed")
    return cleaned


def run_select(sql: str, db_path: Path | str = LEARNER_DB, max_rows: int = 50) -> list[dict[str, Any]]:
    """Run a validated read-only SELECT and return at most max_rows rows."""
    cleaned = validate_select(sql)
    con = _connect(db_path)
    try:
        return [dict(r) for r in con.execute(cleaned).fetchmany(max_rows)]
    finally:
        con.close()


def get_learner_profile(identifier: str, db_path: Path | str = LEARNER_DB) -> dict[str, Any]:
    """Fetch one learner by id or full/partial name."""
    ident = (identifier or "").strip()
    con = _connect(db_path)
    try:
        if re.fullmatch(r"SPR-\d{4}", ident, re.I):
            rows = con.execute("SELECT * FROM learners WHERE learner_id = ?", (ident.upper(),)).fetchall()
        else:
            rows = con.execute("SELECT * FROM learners WHERE name LIKE ?", (f"%{ident}%",)).fetchall()
    finally:
        con.close()
    if not rows:
        return {"status": "not_found", "query": ident}
    if len(rows) > 1:
        return {"status": "ambiguous", "matches": [{"learner_id": r["learner_id"], "name": r["name"]} for r in rows]}
    profile = dict(rows[0])
    profile["preferred_employers"] = parse_employers(profile["preferred_employers"])
    return {"status": "found", **profile}


def safe_get_learner_profile(identifier: str, db_path: Path | str = LEARNER_DB) -> dict[str, Any]:
    """Return a structured failure instead of raising on lookup errors."""
    try:
        return get_learner_profile(identifier, db_path)
    except Exception as exc:
        return unable_to_verify("sqlite_db", f"{type(exc).__name__}: {exc}")


def safe_run_select(sql: str, db_path: Path | str = LEARNER_DB) -> Any:
    """Run a safe SELECT and return a structured failure on errors."""
    try:
        return run_select(sql, db_path)
    except Exception as exc:
        return unable_to_verify("sqlite_db", f"{type(exc).__name__}: {exc}")


def get_db_tools(db_path: Path | str = LEARNER_DB):
    """Return the LangChain tools used by the Advisor's ReAct agent."""
    from langchain_core.tools import tool

    @tool
    def get_learner_record(identifier: str) -> str:
        """Look up one learner by id (e.g. SPR-1013) or name and return profile JSON."""
        return json.dumps(safe_get_learner_profile(identifier, db_path), ensure_ascii=False)

    @tool
    def query_learners_sql(sql: str) -> str:
        """Run a read-only SQLite SELECT on the learners table."""
        return json.dumps(safe_run_select(sql, db_path), ensure_ascii=False)

    return [get_learner_record, query_learners_sql]
