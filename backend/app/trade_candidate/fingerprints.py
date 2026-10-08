"""Versioned deterministic fingerprints for candidate and evidence artifacts."""

from app.risk.fingerprints import fingerprint

CANDIDATE_SCHEMA_VERSION = "2i.1.0"
EVIDENCE_SCHEMA_VERSION = "2i-evidence.1.0"
EXECUTION_ASSUMPTIONS_SCHEMA_VERSION = "execution-assumptions.1"


def evidence_fingerprint(evidence_package) -> str:
    return fingerprint(evidence_package.model_dump(mode="python"), domain="trade-evidence-package")


def candidate_fingerprint(candidate_fields: dict) -> str:
    """Hash stable candidate fields; generated IDs and wall-clock metadata are excluded."""
    return fingerprint(candidate_fields, domain="trade-candidate")


def candidate_id_for(candidate_hash: str) -> str:
    return f"tc-{candidate_hash}"
