from hashlib import sha256
import json
from decimal import Decimal
from datetime import datetime
from uuid import UUID

from app.market_data.contracts import MarketDataModel, require_aware_utc


def _normalize(obj):
    if isinstance(obj, MarketDataModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return require_aware_utc(obj, "fingerprint datetime").isoformat().replace("+00:00", "Z")
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def fingerprint(value, domain: str = "backtest") -> str:
    payload = _normalize(value)
    encoded = json.dumps({"domain": domain, "payload": payload}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()
