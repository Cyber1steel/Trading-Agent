"""Offline, deterministic adversarial evaluation for the reasoning boundary.

Scenarios use the production request and service contracts. They describe inputs and
expected outcomes; the simulated provider is only an adapter for tests/evaluation and
is never selected by production orchestration.
"""

from enum import Enum

from pydantic import Field, model_validator

from app.market_data.contracts import MarketDataModel
from app.reasoning.contracts import ReasoningRequest, ReasoningStatus
from app.reasoning.errors import ProviderUnavailable, ProviderTimeout, ReasoningContextError
from app.reasoning.fingerprints import PROMPT_CONTRACT_VERSION
from app.reasoning.gateway import ReasoningProviderGateway
from app.reasoning.provider import (
    ProviderAvailability,
    ProviderCapabilities,
    ProviderReply,
)
from app.reasoning.service import ReasoningService


class ScenarioCategory(str, Enum):
    GOLDEN = "GOLDEN"
    STATUS_OVERRIDE = "STATUS_OVERRIDE"
    RISK_OVERRIDE = "RISK_OVERRIDE"
    HALLUCINATED_EVIDENCE = "HALLUCINATED_EVIDENCE"
    HALLUCINATED_METRIC = "HALLUCINATED_METRIC"
    FUTURE_LEAKAGE = "FUTURE_LEAKAGE"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    CONTRADICTION = "CONTRADICTION"
    CONFIDENCE = "CONFIDENCE"
    PROVENANCE = "PROVENANCE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    DETERMINISM = "DETERMINISM"


class ScenarioSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ProviderFailureMode(str, Enum):
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    EMPTY = "EMPTY"


class EvaluationScenario(MarketDataModel):
    """Immutable evaluation input and its explicit expected boundary behavior."""

    scenario_id: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    category: ScenarioCategory
    severity: ScenarioSeverity
    request: ReasoningRequest
    provider_response: str = "{}"
    provider_failure: ProviderFailureMode | None = None
    expected_status: ReasoningStatus | None = None
    expected_failure_code: str | None = None
    expected_context_rejection: bool = False
    expected_provider_called: bool = True
    expected_supporting_evidence_ids: tuple[str, ...] | None = None
    expected_contradicting_evidence_ids: tuple[str, ...] | None = None
    expected_knowledge_evidence_ids: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def explicit_expected_outcome(self):
        if self.expected_context_rejection:
            if self.expected_status is not None or self.expected_failure_code is not None:
                raise ValueError("context-rejection scenarios cannot also expect a provider result")
        elif self.expected_status is None:
            raise ValueError("scenario must declare an expected status or context rejection")
        return self

    @classmethod
    def create(cls, **fields):
        """Normalize caller sequences into immutable tuples before validation."""
        for name in (
            "expected_supporting_evidence_ids",
            "expected_contradicting_evidence_ids",
            "expected_knowledge_evidence_ids",
        ):
            if fields.get(name) is not None:
                fields[name] = tuple(fields[name])
        return cls.model_validate(fields)


class ScenarioResult(MarketDataModel):
    scenario_id: str
    category: ScenarioCategory
    severity: ScenarioSeverity
    passed: bool
    actual_status: ReasoningStatus | None = None
    actual_failure_code: str | None = None
    provider_called: bool
    error: str | None = None


class EvaluationReport(MarketDataModel):
    scenario_results: tuple[ScenarioResult, ...]
    total_scenarios: int
    passed: int
    failed: int
    correctly_rejected: int
    unsupported_claims_caught: int
    hallucinated_evidence_caught: int
    future_leakage_attempts_blocked: int
    risk_override_attempts_blocked: int
    prompt_injection_attempts_blocked: int
    provenance_violations_caught: int


class SimulatedProvider:
    """Offline fake provider that emits exact raw payloads or configured failures."""

    provider_id = "offline-simulated-provider"
    model_id = "scenario-fixture-v1"

    def __init__(self, response: str, failure: ProviderFailureMode | None = None):
        self.response = response
        self.failure = failure
        self.calls = 0
        self.last_prompt = None

    @property
    def capabilities(self):
        return ProviderCapabilities(
            provider_id=self.provider_id,
            model_id=self.model_id,
            structured_output=True,
            max_input_bytes=1048576,
            max_output_tokens=8192,
            streaming=False,
            supported_contract_versions=(PROMPT_CONTRACT_VERSION,),
            availability=ProviderAvailability.AVAILABLE,
        )

    def generate(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        if self.failure is ProviderFailureMode.TIMEOUT:
            raise ProviderTimeout("simulated provider timeout")
        if self.failure is ProviderFailureMode.UNAVAILABLE:
            raise ProviderUnavailable("simulated provider unavailable")
        if self.failure is ProviderFailureMode.EMPTY:
            return ProviderReply(self.provider_id, self.model_id, "", ())
        return ProviderReply(
            provider_id=self.provider_id,
            model_id=self.model_id,
            raw_response=self.response,
            request_metadata=(("scenario", "offline"),),
        )


def run_scenario(scenario: EvaluationScenario) -> ScenarioResult:
    """Execute one scenario through the production reasoning service."""
    provider = SimulatedProvider(scenario.provider_response, scenario.provider_failure)
    gateway = ReasoningProviderGateway(
        {provider.provider_id: provider}, (provider.provider_id,),
    )
    try:
        result = ReasoningService(gateway).reason(scenario.request)
    except ReasoningContextError as exc:
        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            severity=scenario.severity,
            passed=scenario.expected_context_rejection and not scenario.expected_provider_called and provider.calls == 0,
            actual_failure_code="context_rejected",
            provider_called=provider.calls > 0,
            error=type(exc).__name__,
        )
    except Exception as exc:
        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            severity=scenario.severity,
            passed=False,
            actual_failure_code="unexpected_evaluation_error",
            provider_called=provider.calls > 0,
            error=type(exc).__name__,
        )
    checks = [
        not scenario.expected_context_rejection,
        scenario.expected_status is None or result.status is scenario.expected_status,
        scenario.expected_failure_code is None or result.failure_code == scenario.expected_failure_code,
        provider.calls == int(scenario.expected_provider_called),
    ]
    if scenario.expected_supporting_evidence_ids is not None:
        checks.append(tuple(item.evidence_id for item in result.supporting_evidence) == scenario.expected_supporting_evidence_ids)
    if scenario.expected_contradicting_evidence_ids is not None:
        checks.append(tuple(item.evidence_id for item in result.contradicting_evidence) == scenario.expected_contradicting_evidence_ids)
    if scenario.expected_knowledge_evidence_ids is not None:
        checks.append(tuple(item.evidence_id for item in result.knowledge_references) == scenario.expected_knowledge_evidence_ids)
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        severity=scenario.severity,
        passed=all(checks),
        actual_status=result.status,
        actual_failure_code=result.failure_code,
        provider_called=provider.calls > 0,
        error=None if all(checks) else "actual outcome did not match scenario expectations",
    )


def run_evaluation(scenarios: tuple[EvaluationScenario, ...]) -> EvaluationReport:
    """Run an ordered immutable scenario suite and compute safety-focused metrics."""
    if not isinstance(scenarios, tuple) or any(not isinstance(item, EvaluationScenario) for item in scenarios):
        raise TypeError("scenarios must be a tuple of validated EvaluationScenario values")
    ids = tuple(item.scenario_id for item in scenarios)
    if len(set(ids)) != len(ids):
        raise ValueError("scenario IDs must be unique")
    results = tuple(run_scenario(item) for item in scenarios)
    passed_results = tuple(item for item in results if item.passed)
    counts = {category: sum(item.passed and item.category is category for item in results) for category in ScenarioCategory}
    return EvaluationReport(
        scenario_results=results,
        total_scenarios=len(results),
        passed=len(passed_results),
        failed=len(results) - len(passed_results),
        correctly_rejected=sum(
            result.passed and (
                item.expected_context_rejection or item.expected_failure_code is not None
                or item.expected_status is not ReasoningStatus.ACTIONABLE
            )
            for item, result in zip(scenarios, results)
        ),
        unsupported_claims_caught=(
            counts[ScenarioCategory.HALLUCINATED_METRIC]
            + counts[ScenarioCategory.CONFIDENCE]
            + counts[ScenarioCategory.RISK_OVERRIDE]
        ),
        hallucinated_evidence_caught=counts[ScenarioCategory.HALLUCINATED_EVIDENCE],
        future_leakage_attempts_blocked=counts[ScenarioCategory.FUTURE_LEAKAGE],
        risk_override_attempts_blocked=counts[ScenarioCategory.RISK_OVERRIDE],
        prompt_injection_attempts_blocked=counts[ScenarioCategory.PROMPT_INJECTION],
        provenance_violations_caught=counts[ScenarioCategory.PROVENANCE],
    )
