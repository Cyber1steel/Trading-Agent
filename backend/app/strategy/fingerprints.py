"""Canonical, versioned SHA-256 fingerprints for strategy evaluations."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel

from app.market_data.contracts import MarketDataModel, require_aware_utc

if TYPE_CHECKING:
    from app.market_analysis.contracts import AnalysisResult


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
    envelope = {"schema": "strategy-fingerprint.v1", "domain": domain, "payload": canonical_value(payload)}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def analysis_full_fingerprint(analysis: "AnalysisResult") -> str:
    """Reproduce Phase 2E's canonical full-result fingerprint without changing Phase 2E."""
    from app.market_analysis.contracts import normalized_hash
    fields = {
        "analysis_version": analysis.analysis_version,
        "parameters": analysis.parameters.model_dump(mode="json"),
        "input_start": analysis.input_start,
        "analysis_start": analysis.analysis_start,
        "analysis_end": analysis.analysis_end,
        "cutoff_at": analysis.cutoff_at,
        "inputs": tuple(item.model_dump(mode="json") for item in analysis.inputs),
        "observations": tuple(item.model_dump(mode="json") for item in analysis.observations),
    }
    return normalized_hash(fields)


def analysis_prefix_fingerprint(analysis: "AnalysisResult", evaluation_at: datetime) -> str:
    """Hash only analysis inputs and observations knowable at the requested cutoff.

    Full dataset hashes remain attached to the full AnalysisResult identity. The visible
    prefix binds exact source identities and every Phase 2E observation available by T,
    while intentionally excluding future observations and full-slice content hashes.
    """
    cutoff = require_aware_utc(evaluation_at, "evaluation_at")
    visible = tuple(item.model_dump(mode="json") for item in analysis.observations
                    if require_aware_utc(item.known_at, "observation known_at") <= cutoff)
    sources = tuple({
        "dataset_id": item.provenance.dataset_id,
        "dataset_version": item.provenance.dataset_version,
        "instrument": item.provenance.instrument,
        "timeframe": item.provenance.timeframe,
        "provider_id": item.provenance.provider_id,
        "provider_version": item.provenance.provider_version,
        "provider_symbol": item.provenance.provider_symbol,
        "price_basis": item.provenance.price_basis,
        "timestamp_convention": item.provenance.timestamp_convention,
    } for item in sorted(analysis.inputs, key=lambda selected: selected.provenance.timeframe.value))
    return fingerprint({
        "analysis_version": analysis.analysis_version,
        "parameters": analysis.parameters,
        "input_start": analysis.input_start,
        "analysis_start": analysis.analysis_start,
        "evaluation_at": cutoff,
        "sources": sources,
        "observations": visible,
    }, domain="analysis-visible-prefix")
