"""
db_connector.py — OrchestrAI NOC
Unified database connector: SQLite (default/test), PostgreSQL (prod), MySQL, REST API.
Uses only stdlib by default; psycopg2/mysql-connector are optional.
"""

import json
import sqlite3
import time
import os
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent / "data" / "databases"
DB_DIR.mkdir(parents=True, exist_ok=True)

HR_DB_PATH = DB_DIR / "hr_database.db"
SC_DB_PATH = DB_DIR / "supply_chain.db"
ST_DB_PATH = DB_DIR / "station_streams.db"


# ─────────────────────────────────────────────────────────────────────────────
#  GENERIC CONNECTOR
# ─────────────────────────────────────────────────────────────────────────────

def test_connection(db_type: str, params: dict) -> tuple[bool, str, list]:
    """
    Test a DB connection.
    Returns (success, message, sample_rows).
    Uses only stdlib where possible; graceful error if driver missing.
    """
    if db_type == "sqlite":
        try:
            path = params.get("path", str(HR_DB_PATH))
            conn = sqlite3.connect(path)
            cur  = conn.cursor()
            cur.execute("SELECT sqlite_version()")
            ver = cur.fetchone()[0]
            conn.close()
            return True, f"✓ SQLite {ver} — {path}", []
        except Exception as e:
            return False, str(e)[:200], []

    elif db_type == "postgresql":
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=params.get("host"), port=params.get("port", 5432),
                dbname=params.get("dbname"), user=params.get("user"),
                password=params.get("password"), connect_timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT version()")
            ver = cur.fetchone()[0]
            conn.close()
            return True, f"✓ PostgreSQL: {ver[:60]}", []
        except ImportError:
            return False, "psycopg2 not installed. Run: pip install psycopg2-binary", []
        except Exception as e:
            return False, str(e)[:200], []

    elif db_type == "mysql":
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=params.get("host"), port=params.get("port", 3306),
                database=params.get("dbname"), user=params.get("user"),
                password=params.get("password"), connection_timeout=5)
            conn.close()
            return True, "✓ MySQL connection successful", []
        except ImportError:
            return False, "mysql-connector-python not installed. Run: pip install mysql-connector-python", []
        except Exception as e:
            return False, str(e)[:200], []

    elif db_type == "rest":
        try:
            import urllib.request
            url   = params.get("url", "")
            token = params.get("token", "")
            headers = {"Accept": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            return True, f"✓ REST API responded 200 — {url[:50]}", []
        except Exception as e:
            return False, str(e)[:200], []

    return False, f"Unknown db_type: {db_type}", []


def query_sqlite(path: str, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT on a SQLite DB and return list of dicts."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def fetch_engineers(cfg: dict) -> list[dict]:
    """
    Fetch engineers from HR DB based on config.
    Falls back to empty list; caller handles static fallback.
    """
    db_type = cfg.get("db_type", "sqlite")
    try:
        if db_type == "sqlite":
            path = cfg.get("path", str(HR_DB_PATH))
            sql  = cfg.get("query", "SELECT * FROM engineers WHERE active=1 ORDER BY on_call DESC")
            return query_sqlite(path, sql)
        elif db_type in ("postgresql", "mysql"):
            import psycopg2
            conn = psycopg2.connect(host=cfg["host"], port=cfg.get("port",5432),
                dbname=cfg["dbname"], user=cfg["user"], password=cfg["password"])
            cur = conn.cursor()
            cur.execute(cfg.get("query","SELECT * FROM engineers WHERE active=true"))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
            return rows
        elif db_type == "rest":
            import urllib.request
            req = urllib.request.Request(cfg.get("url",""),
                headers={"Authorization":f"Bearer {cfg.get('token','')}","Accept":"application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            return data if isinstance(data, list) else data.get("engineers", data.get("data", []))
    except Exception as e:
        return []
    return []


def fetch_parts(cfg: dict, subsystem: str = "") -> list[dict]:
    """Fetch spare parts from Supply Chain DB, optionally filtered by subsystem keywords."""
    db_type = cfg.get("db_type", "sqlite")
    kw_map  = {
        "power_subsystem":       ["rectifier","battery","BBU","MCB","fuse","vrla"],
        "thermal_management":    ["fan","filter","bearing","heatsink","hvac"],
        "rf_antenna":            ["connector","feeder","DIN","coax","PA","antenna"],
        "backhaul_connectivity": ["fibre","SFP","splice","microwave","ODU","patch"],
        "baseband_processing":   ["BBU","DDR","card","FPGA","module","eRAN"],
    }
    keywords = kw_map.get(subsystem, [])

    try:
        if db_type == "sqlite":
            path = cfg.get("path", str(SC_DB_PATH))
            rows = query_sqlite(path, "SELECT * FROM spare_parts WHERE quantity_available > 0 ORDER BY part_category, part_name")
        elif db_type in ("postgresql","mysql"):
            import psycopg2
            conn = psycopg2.connect(host=cfg["host"], port=cfg.get("port",5432),
                dbname=cfg["dbname"], user=cfg["user"], password=cfg["password"])
            cur = conn.cursor()
            cur.execute(cfg.get("query","SELECT * FROM spare_parts WHERE quantity_available > 0"))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
        else:
            return []

        if keywords and rows:
            filtered = [r for r in rows if any(k.lower() in str(r.get("part_name","")).lower()
                        or k.lower() in str(r.get("part_category","")).lower() for k in keywords)]
            return filtered if filtered else rows[:5]
        return rows
    except Exception:
        return []


def fetch_station_stream(cfg: dict, station_id: str, limit: int = 50) -> list[dict]:
    """Fetch recent telemetry rows for a station from the streams DB."""
    db_type = cfg.get("db_type", "sqlite")
    try:
        if db_type == "sqlite":
            path = cfg.get("path", str(ST_DB_PATH))
            return query_sqlite(path,
                "SELECT * FROM station_telemetry WHERE station_id=? ORDER BY ts DESC LIMIT ?",
                (station_id, limit))
    except Exception:
        return []
    return []
