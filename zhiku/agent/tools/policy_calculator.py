import re
import unicodedata
from decimal import Decimal

from pydantic import BaseModel, Field


NIGHTLY_LIMIT = re.compile(
    r"住宿[^。；！？\n]{0,28}?(?:上限|限额)\s*(?:为|是|:)?\s*(?:每晚|每夜)\s*"
    r"([0-9]{1,7}(?:\.[0-9]{1,2})?)\s*元"
    r"|(?:每晚|每夜)\s*住宿[^。；！？\n]{0,24}?(?:上限|限额|最高)\s*(?:为|是|:)?\s*"
    r"([0-9]{1,7}(?:\.[0-9]{1,2})?)\s*元"
)
SENTENCE_END = "。；;！？!?"


class PolicyQuote(BaseModel):
    source_id: str = Field(max_length=16)
    quote: str = Field(max_length=300)


def nights_from_question(question):
    values = set(re.findall(r"(?<![\d.\-])([0-9]{1,3})\s*晚", unicodedata.normalize("NFKC", question)))
    if len(values) == 1 and 1 <= (nights := int(values.pop())) <= 365:
        return nights
    return None


def reimbursement_calculator(nights, quote):
    quote = unicodedata.normalize("NFKC", quote.strip().rstrip(SENTENCE_END))
    if re.search(r"[。；;！？!?\n]", quote):
        return None
    match = NIGHTLY_LIMIT.search(quote)
    if not match:
        return None
    cap = Decimal(match[1] or match[2])
    return {"nightly_cap": format(cap, "f"), "nights": nights,
            "maximum": format(cap * nights, "f"), "currency": "CNY"}


def is_policy_sentence(quote, evidence):
    target = quote.strip().rstrip(SENTENCE_END).strip()
    return bool(target) and any(
        line.strip().rstrip(SENTENCE_END).strip() == target
        for line in re.split(r"(?<=[。；;！？!?])|\n+", evidence)
    )
