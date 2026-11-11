import secrets
import time
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, StringConstraints

from ...access import require_scope
from ...content_policy import load_rules, require_allowed


TicketCategory = Literal["hardware", "network", "software", "other"]


class TicketFields(BaseModel):
    category: TicketCategory
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    location: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


def new_draft(connection, tenant, owner, namespace, fields):
    require_allowed(load_rules(), fields.description, fields.location)
    now = int(time.time())
    identity = secrets.token_hex(12)
    connection.execute("DELETE FROM it_tickets WHERE status = 'draft' AND expires_at <= ?", (now,))
    connection.execute("""
        INSERT INTO it_tickets (id, tenant, owner, namespace, category, description, location, status, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'draft', ?)
    """, (identity, tenant, owner, namespace, fields.category, fields.description, fields.location, now + 900))
    return {"draft_id": identity, **fields.model_dump()}


def create_ticket(connection, identity, tenant, owner, account):
    row = connection.execute("""
        SELECT namespace, category, description, location, status, expires_at
        FROM it_tickets WHERE id = ? AND tenant = ? AND owner = ?
    """, (identity, tenant, owner)).fetchone()
    if not row:
        raise HTTPException(404, "找不到该工单草稿。")
    namespace, category, description, location, status, expires = row
    require_scope(account, namespace)
    if status == "draft":
        now = int(time.time())
        if expires <= now:
            raise HTTPException(410, "工单草稿已过期，请重新填写。")
        require_allowed(load_rules(), description, location)
        connection.execute("UPDATE it_tickets SET status = 'open', submitted_at = ? WHERE id = ?", (now, identity))
    return {"ticket_id": "IT-" + identity.upper(), "status": "open", "namespace": namespace,
            "category": category, "already_submitted": status == "open"}
