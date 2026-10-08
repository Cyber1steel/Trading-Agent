"""Canonical SHA-256 fingerprints for deterministic risk artifacts."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from uuid import UUID

from pydantic import BaseModel

from app.market_data.contracts import MarketDataModel, require_aware_utc


def canonical_value(value):
    if isinstance(value, MarketDataModel):
        return canonical_value(value.model_dump(mode="python"))
    if isinstance(value, BaseModel):
        return canonical_value(value.model_dump(mode="python"))
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Fingerprint values must contain finite Decimals")
        return str(value)
    if isinstance(value, datetime):
        return require_aware_utc(value, "fingerprint datetime").isoformat().replace("+00:00", "Z")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}")


def fingerprint(payload: object, *, domain: str) -> str:
    envelope = {"schema": "risk-fingerprint.v1", "domain": domain, "payload": canonical_value(payload)}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()
