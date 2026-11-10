import json
import logging
import sqlite3
import time
from contextlib import closing, contextmanager

from fastapi import HTTPException
from content_policy import SensitiveContentError


class DocumentParseError(HTTPException):
    pass


@contextmanager
def track(database, account, action, namespace, document_id=None):
    started = time.perf_counter()
    entry = {"document_id": document_id, "version": None, "verification": None, "sources": [],
             "namespace": namespace}
    status = 204 if action == "offline" else 200
    failure_kind = None
    try:
        yield entry
    except HTTPException as error:
        status = error.status_code
        if isinstance(error, DocumentParseError):
            failure_kind = "document_parse"
        elif isinstance(error, SensitiveContentError):
            failure_kind = "sensitive_content"
        raise
    except GeneratorExit:
        status = 499
        raise
    except sqlite3.OperationalError as error:
        status = 500
        code = getattr(error, "sqlite_errorcode", 0)
        if code & 0xff in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            failure_kind = "database_busy"
        raise
    except Exception:
        status = 500
        raise
    finally:
        try:
            with closing(database()) as connection, connection:
                cursor = connection.execute("""
                    INSERT INTO audit_events(tenant, actor, action, namespace, document_id, version,
                                             status, duration_ms, verification, sources, created_at, failure_kind)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (account.tenant if account else "default", account.name if account else "local",
                      action, entry["namespace"], entry["document_id"], entry["version"], status,
                      round((time.perf_counter() - started) * 1000), entry["verification"],
                      json.dumps(entry["sources"]), time.time_ns() // 1_000_000, failure_kind))
            entry["event_id"] = cursor.lastrowid
        except sqlite3.Error:
            logging.getLogger(__name__).exception("Could not persist an audit event")


def record_answer(entry, result):
    entry["verification"] = result["verification"]
    entry["sources"] = [
        {key: item[key] for key in ("document_id", "version", "chunk")}
        for item in result["sources"]
    ]


ALERT_WINDOW_MINUTES = 15


def operational_alerts(connection, tenant, now_ms):
    counts = connection.execute("""
        SELECT count(CASE WHEN failure_kind = 'document_parse' THEN 1 END),
               count(CASE WHEN failure_kind = 'database_busy' THEN 1 END),
               count(CASE WHEN action = 'question' AND status IN (502, 504) THEN 1 END),
               count(CASE WHEN action = 'question' AND status = 429 THEN 1 END),
               count(CASE WHEN action = 'question' AND status = 200
                          AND verification = 'unsupported' THEN 1 END),
               count(CASE WHEN action = 'question' AND status = 200
                          AND verification IN ('supported', 'unsupported') THEN 1 END)
        FROM audit_events WHERE tenant = ? AND created_at >= ? AND created_at <= ?
    """, (tenant, now_ms - ALERT_WINDOW_MINUTES * 60_000, now_ms)).fetchone()
    parse_failures, busy, model_failures, limited, unsupported, checked = counts
    alerts = [
        {"code": code, "count": count, "threshold": threshold, "level": level}
        for code, count, threshold, level in (
            ("document_parse", parse_failures, 1, "warning"),
            ("database_busy", busy, 1, "error"),
            ("model_error", model_failures, 3, "error"),
            ("rate_limit", limited, 3, "warning"),
        ) if count >= threshold
    ]
    if checked >= 10 and unsupported >= 3 and unsupported * 4 >= checked:
        alerts.append({"code": "verification_spike", "count": unsupported, "sample_count": checked,
                       "rate": unsupported / checked, "threshold": 0.25, "level": "warning"})
    return {"window_minutes": ALERT_WINDOW_MINUTES, "checked_at": now_ms, "alerts": alerts}
