import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import OrderedDict
from contextlib import closing
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, StringConstraints

ScopeName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
COOKIE = "inkatlas_access"
MEMBER_COOKIE = "inkatlas_session"
SESSION_AGE = 8 * 3600
CAPTCHA_AGE = 180
CAPTCHA_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
MIN_PASSWORD = 6
LOGIN_ATTEMPTS = OrderedDict()
LOGIN_LOCK = Lock()


class Member(BaseModel):
    name: ScopeName
    tenant: ScopeName = "default"
    role: Literal["reader", "editor", "admin"] = "reader"
    namespaces: list[ScopeName] = Field(default_factory=lambda: ["default"], min_length=1, max_length=100)
    access_tags: list[str] = Field(default_factory=list, max_length=100)


class Account(Member):
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Credentials(BaseModel):
    name: ScopeName
    password: str = Field(min_length=1, max_length=128)


class LoginCredentials(Credentials):
    captcha_id: str = Field(pattern=r"^[A-Za-z0-9_-]{32}$")
    captcha: str = Field(min_length=5, max_length=5)


class NewMember(Credentials):
    password: str = Field(min_length=MIN_PASSWORD, max_length=128)
    role: Literal["reader", "editor"] = "reader"
    namespace: ScopeName = "default"


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=1, maxmem=64 * 1024 * 1024)


def store_member(connection, member, password):
    salt = secrets.token_bytes(16)
    connection.execute(
        "INSERT INTO members(name, profile, salt, password_hash) VALUES (?, ?, ?, ?)",
        (member.name, member.model_dump_json(), salt, password_hash(password, salt)),
    )


def same_origin(request):
    origin = request.headers.get("Origin")
    if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "请求来源不匹配。")


def new_captcha_code():
    return "".join(secrets.choice(CAPTCHA_ALPHABET) for _ in range(5))


def captcha_digest(identity, code):
    return hashlib.sha256(f"{identity}:{code.upper()}".encode()).hexdigest()


def issue_captcha(connection):
    identity = secrets.token_urlsafe(24)
    code = new_captcha_code()
    now = int(time.time())
    with connection:
        connection.execute("DELETE FROM login_challenges WHERE expires_at <= ?", (now,))
        connection.execute("INSERT INTO login_challenges VALUES (?, ?, ?)",
                           (identity, captcha_digest(identity, code), now + CAPTCHA_AGE))
    image = Image.new("RGB", (154, 48), "#eaf4ee")
    draw = ImageDraw.Draw(image)
    for _ in range(45):
        draw.point((secrets.randbelow(154), secrets.randbelow(48)), fill="#739387")
    font = ImageFont.load_default(size=27)
    for index, character in enumerate(code):
        draw.text((13 + index * 27, 6 + secrets.randbelow(8)), character, fill="#183a30", font=font)
    output = BytesIO()
    image.save(output, format="PNG")
    return {"id": identity, "image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}


def check_captcha(connection, credentials, now):
    with connection:
        row = connection.execute(
            "DELETE FROM login_challenges WHERE id = ? RETURNING answer_hash, expires_at",
            (credentials.captcha_id,),
        ).fetchone()
    if not row or row[1] <= now or not hmac.compare_digest(
        row[0], captcha_digest(credentials.captcha_id, credentials.captcha)
    ):
        raise HTTPException(400, "验证码错误或已过期，请重新输入。")


def sign_in_member(request, credentials, connection):
    same_origin(request)
    ip = request.client.host if request.client else ""
    now = int(time.time())
    with LOGIN_LOCK:
        window, count = LOGIN_ATTEMPTS.get(ip, (now, 0))
        if now - window >= 60:
            window, count = now, 0
        if count >= 10:
            raise HTTPException(429, "登录尝试过多，请稍后再试。", headers={"Retry-After": str(60 - now + window)})
        LOGIN_ATTEMPTS[ip] = (window, count + 1)
        LOGIN_ATTEMPTS.move_to_end(ip)
        if len(LOGIN_ATTEMPTS) > 1024:
            LOGIN_ATTEMPTS.popitem(last=False)
    check_captcha(connection, credentials, now)
    row = connection.execute(
        "SELECT profile, salt, password_hash FROM members WHERE name = ?", (credentials.name,)
    ).fetchone()
    candidate = password_hash(credentials.password, row[1] if row else bytes(16))
    if row is None or not hmac.compare_digest(candidate, row[2]):
        raise HTTPException(401, "用户名或密码错误。")
    token = secrets.token_urlsafe(32)
    connection.execute("DELETE FROM member_sessions WHERE expires_at <= ?", (now,))
    connection.execute("INSERT INTO member_sessions VALUES (?, ?, ?)",
                       (hashlib.sha256(token.encode()).hexdigest(), credentials.name, now + SESSION_AGE))
    return Member.model_validate_json(row[0]), token


def accounts():
    filename = os.getenv("INKATLAS_ACCOUNTS_FILE")
    if not filename:
        return None
    try:
        values = [Account.model_validate(item) for item in json.loads(Path(filename).read_text(encoding="utf-8-sig"))]
        if not values or len({item.token_sha256 for item in values}) != len(values):
            raise ValueError
        return values
    except (OSError, ValueError, TypeError) as error:
        raise HTTPException(503, "账号配置不可用。") from error


def authenticate(request: Request):
    from app import database

    configured = accounts()
    authorization = request.headers.get("Authorization")
    if authorization is None and request.method not in {"GET", "HEAD", "OPTIONS"}:
        same_origin(request)
    session = request.cookies.get(MEMBER_COOKIE, "") if authorization is None else ""
    with closing(database()) as connection:
        if 0 < len(session) <= 512:
            row = connection.execute("""
                SELECT m.profile FROM member_sessions s JOIN members m ON m.name = s.member
                WHERE s.token_hash = ? AND s.expires_at > ?
            """, (hashlib.sha256(session.encode()).hexdigest(), int(time.time()))).fetchone()
            if row:
                return Member.model_validate_json(row[0])
        has_members = connection.execute("SELECT 1 FROM members LIMIT 1").fetchone()
    if configured is None and not has_members:
        return None
    if authorization is not None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            token = ""
    else:
        token = request.cookies.get(COOKIE, "")
    if 0 < len(token) <= 512:
        digest = hashlib.sha256(token.encode()).hexdigest()
        for account in configured or ():
            if hmac.compare_digest(account.token_sha256, digest):
                return account
    raise HTTPException(401, "请先登录。", headers={"WWW-Authenticate": "Bearer"})


Actor = Annotated[Member | None, Depends(authenticate)]


def can_write(account: Actor):
    if account and account.role == "reader":
        raise HTTPException(403, "当前账号没有文档管理权限。")
    return account


Writer = Annotated[Member | None, Depends(can_write)]


def require_scope(account, namespace, tags=()):
    if account and (namespace not in account.namespaces or not set(tags).issubset(account.access_tags)):
        raise HTTPException(403, "当前账号无权访问该知识库或权限标签。")


def document_scope(account):
    if account is None:
        return "d.tenant = ?", ["default"]
    return """d.tenant = ? AND d.namespace IN (SELECT value FROM json_each(?))
        AND NOT EXISTS (
            SELECT 1 FROM json_each(d.access_tags) AS required
            WHERE required.value NOT IN (SELECT value FROM json_each(?))
        )""", [account.tenant, json.dumps(account.namespaces), json.dumps(account.access_tags)]
