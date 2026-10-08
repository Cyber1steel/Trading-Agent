"""Stable fingerprints for prompts and reasoning artifacts."""

from hashlib import sha256
import json

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from math import isfinite
from pydantic import BaseModel
from uuid import UUID

from app.market_data.contracts import MarketDataModel, require_aware_utc

REASONING_CONTRACT_VERSION = "3a.1.0"
PROMPT_CONTRACT_VERSION = "3a-prompt.1.0"


def digest(payload: object, *, domain: str) -> str:
    encoded = json.dumps(
        {"schema": REASONING_CONTRACT_VERSION, "domain": domain, "payload": _canonical(payload)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _canonical(value):
    if isinstance(value, MarketDataModel):
        return _canonical(value.model_dump(mode="python"))
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="python"))
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite Decimal cannot be fingerprinted")
        return str(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("non-finite retrieval score cannot be fingerprinted")
        return {"$float_hex": value.hex()}
    if isinstance(value, datetime):
        return require_aware_utc(value, "reasoning fingerprint time").isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError(f"Unsupported reasoning fingerprint value: {type(value).__name__}")


def context_fingerprint(context) -> str:
    return digest(context.model_dump(mode="python", exclude={"fingerprint"}), domain="reasoning-context")


def result_fingerprint(result_fields: dict) -> str:
    return digest(result_fields, domain="reasoning-result")
