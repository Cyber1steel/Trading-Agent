"""Deterministic risk evaluation service."""

from decimal import Context, Decimal, localcontext

from app.risk.calculations import calculate_trade_risk
from app.risk.contracts import (
    RiskConfiguration,
    RiskFinding,
    RiskResult,
    RiskSeverity,
    RiskStatus,
    TradeRiskRequest,
)
from app.risk.errors import InsufficientEvidenceError, RiskRejectedError
from app.risk.fingerprints import fingerprint
from app.risk.validation import validate_risk_calculation


class RiskEngineService:
    """Evaluate trade risk in a deterministic, fail-closed way."""

    def evaluate(self, request: TradeRiskRequest) -> RiskResult:
        with localcontext(Context(prec=34)):
            return self._evaluate(request)

    def _evaluate(self, request: TradeRiskRequest) -> RiskResult:
        config = request.risk_configuration
        if config is None:
            return self._insufficient_result(
                request,
                "risk configuration is required",
                details={"rule": "risk_configuration"},
            )

        if request.risk_configuration_fingerprint is not None and request.risk_configuration_fingerprint != config.fingerprint:
            return self._reject_result(
                request,
                [
                    RiskFinding(
                        rule="risk_configuration_fingerprint",
                        observed_value=request.risk_configuration_fingerprint,
                        expected_constraint=config.fingerprint,
                        severity=RiskSeverity.REJECT,
                        message="risk_configuration_fingerprint does not match the supplied configuration identity.",
                    )
                ],
                message="Risk configuration fingerprint mismatch.",
            )

        if request.instrument is None:
            return self._insufficient_result(request, "instrument is required", details={"rule": "instrument"})
        if request.side is None:
            return self._insufficient_result(request, "trade direction is required", details={"rule": "side"})
        if request.entry_price is None:
            return self._insufficient_result(request, "entry_price is required", details={"rule": "entry_price"})
        if request.stop_price is None:
            return self._insufficient_result(request, "stop_price is required", details={"rule": "stop_price"})
        if request.account_equity is None:
            return self._insufficient_result(request, "account_equity is required", details={"rule": "account_equity"})
        if request.execution_assumptions is None and config.execution_assumptions is None:
            return self._insufficient_result(request, "execution assumptions are required", details={"rule": "execution_assumptions"})

        try:
            calculation = calculate_trade_risk(request)
            findings = validate_risk_calculation(request, config, calculation)
        except InsufficientEvidenceError as exc:
            return self._insufficient_result(request, str(exc), details={"rule": exc.rule or "insufficient_evidence"})
        except RiskRejectedError as exc:
            findings = (
                RiskFinding(
                    rule=exc.rule or "trade_validation",
                    observed_value=str(exc),
                    expected_constraint="configured risk limits and stop/target relationships must pass",
                    severity=RiskSeverity.REJECT,
                    message=str(exc),
                ),
            )
            return self._reject_result(request, findings, message=str(exc))

        status = RiskStatus.RISK_VALID
        if findings:
            reject = any(item.severity == RiskSeverity.REJECT for item in findings)
            insufficient = any(item.severity == RiskSeverity.INSUFFICIENT for item in findings)
            if insufficient:
                status = RiskStatus.INSUFFICIENT_EVIDENCE
            elif reject:
                status = RiskStatus.RISK_REJECTED

        result = RiskResult(
            status=status,
            request_fingerprint=request.fingerprint,
            requested_risk=request.account_equity * config.risk_per_trade,
            maximum_monetary_risk=calculation.maximum_monetary_risk,
            stop_distance=calculation.stop_distance,
            position_size=calculation.position_size,
            notional_exposure=calculation.notional_exposure,
            estimated_fees=calculation.estimated_fees,
            estimated_spread_cost=calculation.estimated_spread_cost,
            estimated_slippage_impact=calculation.estimated_slippage_impact,
            total_estimated_downside=calculation.total_estimated_downside,
            reward_risk=calculation.reward_risk,
            validation_findings=findings,
            warnings=(),
            provenance={
                "configuration_id": config.configuration_id,
                "configuration_version": str(config.configuration_version),
                "instrument": request.instrument.symbol,
                "side": request.side.value,
                "strategy_id": request.strategy_id or "",
                "setup_id": request.setup_id or "",
                "analysis_fingerprint": request.analysis_fingerprint or "",
            },
            fingerprint="",
        )
        result = result.model_copy(update={"fingerprint": self._fingerprint_result(result)})
        return result

    def _insufficient_result(self, request: TradeRiskRequest, message: str, *, details: dict[str, str] | None = None) -> RiskResult:
        result = RiskResult(
            status=RiskStatus.INSUFFICIENT_EVIDENCE,
            request_fingerprint=request.fingerprint,
            requested_risk=(request.account_equity * request.risk_configuration.risk_per_trade) if request.account_equity is not None and request.risk_configuration is not None else None,
            maximum_monetary_risk=None,
            stop_distance=(abs(request.entry_price - request.stop_price) if request.entry_price is not None and request.stop_price is not None else None),
            position_size=None,
            notional_exposure=None,
            estimated_fees=None,
            estimated_spread_cost=None,
            estimated_slippage_impact=None,
            total_estimated_downside=None,
            reward_risk=None,
            validation_findings=(
                RiskFinding(
                    rule=details.get("rule", "insufficient_evidence") if details else "insufficient_evidence",
                    observed_value=message,
                    expected_constraint="required risk inputs and execution assumptions must be explicit",
                    severity=RiskSeverity.INSUFFICIENT,
                    message=message,
                ),
            ),
            warnings=(),
            provenance={
                "configuration_id": request.risk_configuration.configuration_id if request.risk_configuration else "",
                "strategy_id": request.strategy_id or "",
                "setup_id": request.setup_id or "",
                "instrument": request.instrument.symbol if request.instrument is not None else "",
            },
            fingerprint="",
        )
        result = result.model_copy(update={"fingerprint": self._fingerprint_result(result)})
        return result

    def _reject_result(self, request: TradeRiskRequest, findings: tuple[RiskFinding, ...] | list[RiskFinding], *, message: str) -> RiskResult:
        requested_risk = None
        if request.account_equity is not None and request.risk_configuration is not None:
            requested_risk = request.account_equity * request.risk_configuration.risk_per_trade

        stop_distance = None
        if request.entry_price is not None and request.stop_price is not None:
            stop_distance = abs(request.entry_price - request.stop_price)

        result = RiskResult(
            status=RiskStatus.RISK_REJECTED,
            request_fingerprint=request.fingerprint,
            requested_risk=requested_risk,
            maximum_monetary_risk=requested_risk,
            stop_distance=stop_distance,
            position_size=None,
            notional_exposure=None,
            estimated_fees=None,
            estimated_spread_cost=None,
            estimated_slippage_impact=None,
            total_estimated_downside=None,
            reward_risk=None,
            validation_findings=tuple(findings),
            warnings=(message,),
            provenance={
                "configuration_id": request.risk_configuration.configuration_id if request.risk_configuration else "",
                "strategy_id": request.strategy_id or "",
                "setup_id": request.setup_id or "",
                "instrument": request.instrument.symbol if request.instrument is not None else "",
            },
            fingerprint="",
        )
        result = result.model_copy(update={"fingerprint": self._fingerprint_result(result)})
        return result

    @staticmethod
    def _fingerprint_result(result: RiskResult) -> str:
        payload = {
            "status": result.status.value,
            "request_fingerprint": result.request_fingerprint,
            "requested_risk": str(result.requested_risk) if result.requested_risk is not None else None,
            "maximum_monetary_risk": str(result.maximum_monetary_risk) if result.maximum_monetary_risk is not None else None,
            "stop_distance": str(result.stop_distance) if result.stop_distance is not None else None,
            "position_size": str(result.position_size) if result.position_size is not None else None,
            "notional_exposure": str(result.notional_exposure) if result.notional_exposure is not None else None,
            "estimated_fees": str(result.estimated_fees) if result.estimated_fees is not None else None,
            "estimated_spread_cost": str(result.estimated_spread_cost) if result.estimated_spread_cost is not None else None,
            "estimated_slippage_impact": str(result.estimated_slippage_impact) if result.estimated_slippage_impact is not None else None,
            "total_estimated_downside": str(result.total_estimated_downside) if result.total_estimated_downside is not None else None,
            "reward_risk": str(result.reward_risk) if result.reward_risk is not None else None,
            "validation_findings": [item.model_dump(mode="python") for item in result.validation_findings],
            "warnings": list(result.warnings),
            "provenance": result.provenance,
        }
        return fingerprint(payload, domain="risk-result")
