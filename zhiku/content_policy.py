import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException


class SensitiveContentError(HTTPException):
    def __init__(self):
        super().__init__(403, "内容触发了知识库的敏感词规则。")


def normalize(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(char for char in text if not char.isspace() and unicodedata.category(char) != "Cf")


@lru_cache(maxsize=4)
def compile_rules(contents):
    terms = {normalize(line) for line in contents.splitlines() if not line.lstrip().startswith("#")}
    terms.discard("")
    if len(terms) > 500 or any(len(term) > 100 for term in terms):
        raise HTTPException(503, "敏感词配置超出限制。")
    return re.compile("|".join(re.escape(term) for term in sorted(terms))) if terms else None


def load_rules():
    path = os.getenv("INKATLAS_SENSITIVE_WORDS_FILE")
    if not path:
        return None
    try:
        with Path(path).open("r", encoding="utf-8-sig") as file:
            contents = file.read(65_537)
        if len(contents) > 65_536:
            raise HTTPException(503, "敏感词配置超出限制。")
        return compile_rules(contents)
    except (OSError, ValueError) as error:
        raise HTTPException(503, "敏感词配置无法读取。") from error


def blocked(rules, text):
    return bool(rules and rules.search(normalize(text)))


def require_allowed(rules, *values):
    if any(blocked(rules, text) for text in values):
        raise SensitiveContentError()
