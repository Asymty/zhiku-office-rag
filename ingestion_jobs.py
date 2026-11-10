import json
import logging
import sqlite3
import time
from contextlib import closing
from threading import Event, Thread

from fastapi import HTTPException

logger = logging.getLogger(__name__)
JOB_COLUMNS = "id, name, namespace, status, stage, result, error, created_at, updated_at"


def now_ms():
    return time.time_ns() // 1_000_000


def public_job(row):
    job = dict(row)
    if job["result"]:
        job["result"] = json.loads(job["result"])
    return job


def claim_job(database):
    now = now_ms()
    with closing(database()) as connection, connection:
        connection.row_factory = sqlite3.Row
        cutoff = now - 120_000
        if connection.execute("""
            SELECT 1 FROM ingestion_jobs WHERE status = 'running' AND heartbeat_at < ? LIMIT 1
        """, (cutoff,)).fetchone():
            connection.execute("""
                UPDATE ingestion_jobs SET status = 'failed', error = ?, data = NULL, updated_at = ?
                WHERE status = 'running' AND heartbeat_at < ?
            """, ("入库任务中断，请检查文档版本后重新提交。", now, cutoff))
        row = connection.execute("""
            SELECT id FROM ingestion_jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1
        """).fetchone()
        if not row:
            return None
        return connection.execute("""
            UPDATE ingestion_jobs SET status = 'running', stage = 'parsing', heartbeat_at = ?, updated_at = ?
            WHERE id = ? AND status = 'queued' RETURNING *
        """, (now, now, row["id"])).fetchone()


def run_job(database, process, job):
    identity = job["id"]
    finished = Event()

    def heartbeat():
        while not finished.wait(10):
            try:
                with closing(database()) as connection, connection:
                    connection.execute("""
                        UPDATE ingestion_jobs SET heartbeat_at = ? WHERE id = ? AND status = 'running'
                    """, (now_ms(), identity))
            except sqlite3.Error:
                logger.exception("Could not renew ingestion job %s", identity)

    def progress(stage):
        with closing(database()) as connection, connection:
            changed = connection.execute("""
                UPDATE ingestion_jobs SET stage = ?, updated_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'running'
            """, (stage, now_ms(), now_ms(), identity)).rowcount
        if not changed:
            raise HTTPException(409, "入库任务已中断。")

    def complete(connection, result):
        changed = connection.execute("""
            UPDATE ingestion_jobs SET status = 'completed', stage = 'completed', result = ?,
                                      data = NULL, updated_at = ?
            WHERE id = ? AND status = 'running'
        """, (json.dumps(result), now_ms(), identity)).rowcount
        if not changed:
            raise HTTPException(409, "入库任务已中断。")

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    error = None
    try:
        process(job, progress, complete)
    except HTTPException as exc:
        error = str(exc.detail)[:500]
    except Exception:
        logger.exception("Ingestion job %s failed", identity)
        error = "入库处理失败，请查看服务日志。"
    finally:
        finished.set()
        worker.join()
    if error is not None:
        with closing(database()) as connection, connection:
            connection.execute("""
                UPDATE ingestion_jobs SET status = 'failed', error = ?, data = NULL, updated_at = ?
                WHERE id = ? AND status = 'running'
            """, (error, now_ms(), identity))


def run_worker(database, process, stop):
    while not stop.is_set():
        try:
            job = claim_job(database)
            if job:
                run_job(database, process, job)
                continue
        except sqlite3.Error:
            logger.exception("Ingestion worker could not access the queue")
        stop.wait(1)
