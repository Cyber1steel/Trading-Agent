"""Deterministically assemble a research candidate from existing domain outputs."""

from decimal import Context, Decimal, localcontext

from app.execution.assumptions import ExecutionAssumptions
from app.market_analysis.contracts import AnalysisResult
from app.risk.contracts import RiskResult, RiskSeverity, RiskStatus, TradeRiskRequest
from app.risk.service import RiskEngineService
from app.strategy.contracts import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    LifecycleState,
    SetupDefinition,
    StrategyDefinition,
)
from app.strategy.fingerprints import analysis_full_fingerprint, analysis_prefix_fingerprint, fingerprint
from app.trade_candidate.contracts import (
    AnalysisEvidenceRef,
    CandidateStatus,
    DatasetEvidenceRef,
    DefinitionEvidenceRef,
    EvaluationEvidenceRef,
    EvidencePackage,
    ExecutionEvidenceRef,
    RiskEvidenceRef,
    TradeCandidate,
)
from app.trade_candidate.errors import CandidateInputError
from app.trade_candidate.fingerprints import candidate_fingerprint, candidate_id_for
from pydantic import ValidationError

_CALCULATION_CONTEXT = Context(prec=34)


def _evaluation_fingerprint(result: EvaluationResult) -> str:
    """Reproduce the strategy result digest to detect copied or stale artifacts."""
    def visible_evidence(item):
        return item.model_dump(mode="python", exclude={"analysis_fingerprint"})

    trace = tuple({
        "condition_id": item.condition_id,
        "known_at": item.known_at,
        "outcome": item.outcome,
        "evidence": tuple(visible_evidence(ref) for ref in item.evidence),
    } for item in result.condition_trace)
    payload = {
        "request_fingerprint": result.request_fingerprint,
        "status": result.status,
        "condition_trace": trace,
        "evidence": tuple(visible_evidence(item) for item in result.evidence),
        "lifecycle_transitions": tuple({
            "state": item.state,
            "known_at": item.known_at,
            "reason": item.reason,
            "evidence": tuple(visible_evidence(ref) for ref in item.evidence),
        } for item in result.lifecycle_transitions),
    }
    return fingerprint(payload, domain="evaluation-result")


class TradeCandidateService:
    """Compose Phase 2E/2F/risk outputs without recalculating their decisions."""

    def assemble(
        self,
        *,
        analysis: AnalysisResult,
        strategy: StrategyDefinition,
        setup: SetupDefinition,
        evaluation: EvaluationResult,
        risk_request: TradeRiskRequest,
        risk_result: RiskResult,
        execution_assumptions: ExecutionAssumptions | None,
    ) -> TradeCandidate:
        typed_inputs = (
            (analysis, AnalysisResult), (strategy, StrategyDefinition), (setup, SetupDefinition),
            (evaluation, EvaluationResult), (risk_request, TradeRiskRequest), (risk_result, RiskResult),
        )
        validated = []
        for value, expected_type in typed_inputs:
            if not isinstance(value, expected_type):
                raise CandidateInputError(f"{expected_type.__name__} is required for candidate assembly")
            try:
                validated.append(expected_type.model_validate(value.model_dump(mode="python")))
            except ValidationError as exc:
                raise CandidateInputError(f"{expected_type.__name__} failed contract revalidation") from exc
        analysis, strategy, setup, evaluation, risk_request, risk_result = validated
        findings: list[str] = []
        missing: list[str] = []

        # Recheck producer identities and cross-artifact bindings. This validates
        # artifact integrity; it does not rerun analysis, setup evaluation, or sizing.
        actual_analysis_fp = analysis_full_fingerprint(analysis)
        if analysis.fingerprint != actual_analysis_fp:
            findings.append("analysis fingerprint does not match analysis content")
        if evaluation.fingerprint != _evaluation_fingerprint(evaluation):
            findings.append("strategy evaluation fingerprint does not match evaluation content")
        if risk_result.fingerprint != RiskEngineService._fingerprint_result(risk_result):
            findings.append("risk result fingerprint does not match risk result content")
        if risk_result.request_fingerprint != risk_request.fingerprint:
            findings.append("risk result does not reference the supplied risk request")

        primary = next((item.provenance for item in analysis.inputs
                        if item.provenance.timeframe == setup.primary_timeframe), None)
        if primary is None:
            missing.append("analysis does not contain the setup primary timeframe")
        elif primary.instrument != setup.instrument:
            findings.append("analysis instrument does not match the setup instrument")
        if analysis.analysis_version != strategy.analysis_version:
            findings.append("strategy and analysis versions do not match")
        try:
            exact_setup = strategy.setup(setup.setup_id, setup.setup_version)
        except ValueError:
            exact_setup = None
        if exact_setup is None or exact_setup.fingerprint != setup.fingerprint:
            findings.append("setup definition is not the exact version contained by the strategy")

        prefix = analysis_prefix_fingerprint(analysis, evaluation.evaluation_at)
        expected_evaluation_request = EvaluationRequest(
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.strategy_version,
            strategy_fingerprint=strategy.fingerprint,
            setup_id=setup.setup_id,
            setup_version=setup.setup_version,
            setup_fingerprint=setup.fingerprint,
            analysis_fingerprint=analysis.fingerprint,
            evaluation_prefix_fingerprint=prefix,
            evaluation_at=evaluation.evaluation_at,
        )
        if evaluation.request_fingerprint != expected_evaluation_request.fingerprint:
            findings.append("strategy evaluation request fingerprint is stale or incompatible")
        expected_eval_identity = (
            strategy.strategy_id, strategy.strategy_version, strategy.fingerprint,
            setup.setup_id, setup.setup_version, setup.fingerprint,
            analysis.fingerprint, prefix,
        )
        actual_eval_identity = (
            evaluation.strategy_id, evaluation.strategy_version, evaluation.strategy_fingerprint,
            evaluation.setup_id, evaluation.setup_version, evaluation.setup_fingerprint,
            evaluation.analysis_fingerprint, evaluation.evaluation_prefix_fingerprint,
        )
        if expected_eval_identity != actual_eval_identity:
            findings.append("strategy evaluation identity does not match the supplied definitions and analysis")
        known_observations = [item for item in analysis.observations
                              if item.known_at == evaluation.evaluation_at]
        if len(known_observations) != 1:
            findings.append("evaluation timestamp does not identify exactly one analysis observation")
        if evaluation.evaluation_at > analysis.cutoff_at:
            findings.append("evaluation timestamp is after the analysis cutoff")
        if any(item.known_at > analysis.cutoff_at for item in analysis.observations):
            findings.append("analysis contains observations beyond its declared cutoff")
        if any(item.known_at != item.bar_open + setup.primary_timeframe.nominal_duration
               for item in analysis.observations):
            findings.append("analysis observation close time is inconsistent with the setup timeframe")
        observation_times = {item.known_at for item in analysis.observations}
        if risk_request.evaluation_at != evaluation.evaluation_at:
            findings.append("risk request timestamp differs from strategy evaluation timestamp")
        if risk_request.analysis_fingerprint != analysis.fingerprint:
            findings.append("risk request analysis fingerprint differs from supplied analysis")
        if risk_request.evaluation_prefix_fingerprint != evaluation.evaluation_prefix_fingerprint:
            findings.append("risk request visible-prefix fingerprint differs from strategy evaluation")
        expected_risk_identity = (
            str(strategy.strategy_id), setup.setup_id, strategy.fingerprint,
            setup.fingerprint, analysis.fingerprint, evaluation.evaluation_prefix_fingerprint,
        )
        actual_risk_identity = (
            risk_request.strategy_id, risk_request.setup_id, risk_request.strategy_fingerprint,
            risk_request.setup_fingerprint, risk_request.analysis_fingerprint,
            risk_request.evaluation_prefix_fingerprint,
        )
        if actual_risk_identity != expected_risk_identity:
            findings.append("risk request strategy/setup/analysis identity is incomplete or incompatible")
        if risk_request.instrument != setup.instrument:
            findings.append("risk request instrument differs from the setup instrument")
        if risk_request.side is None:
            missing.append("risk request direction is missing")
        if risk_request.execution_assumptions is None and (
            risk_request.risk_configuration is None
            or risk_request.risk_configuration.execution_assumptions is None
        ):
            missing.append("risk execution assumptions are unavailable")
        effective_assumptions = risk_request.execution_assumptions or (
            risk_request.risk_configuration.execution_assumptions
            if risk_request.risk_configuration is not None else None
        )
        if execution_assumptions is not None and not isinstance(execution_assumptions, ExecutionAssumptions):
            raise CandidateInputError("execution_assumptions must be an ExecutionAssumptions instance")
        if execution_assumptions is None:
            missing.append("candidate execution assumptions are missing")
        elif effective_assumptions is None:
            missing.append("risk request execution assumptions are unavailable")
        elif execution_assumptions != effective_assumptions:
            findings.append("candidate execution assumptions differ from those used by the risk request")
        if execution_assumptions is None:
            execution_ref = None
        else:
            execution_ref = ExecutionEvidenceRef(
                assumptions=execution_assumptions,
                fingerprint=fingerprint(execution_assumptions, domain="execution-assumptions"),
            )

        # Evidence must be drawn only from the selected datasets and the visible
        # analysis prefix. The full analysis digest is retained as provenance, but
        # no suffix evidence can enter this package.
        dataset_by_key = {
            (item.provenance.dataset_id, item.provenance.dataset_version,
             item.provenance.instrument, item.provenance.timeframe): item
            for item in analysis.inputs
        }
        all_evidence = list(evaluation.evidence)
        for item in evaluation.condition_trace:
            if item.known_at > evaluation.evaluation_at:
                findings.append("condition trace is known after the candidate evaluation timestamp")
            all_evidence.extend(item.evidence)
        for item in evaluation.lifecycle_transitions:
            if item.known_at > evaluation.evaluation_at:
                findings.append("lifecycle transition is known after the candidate evaluation timestamp")
            all_evidence.extend(item.evidence)
        evidence_times = {item.known_at for item in all_evidence}
        prefix_by_time = {
            known_at: analysis_prefix_fingerprint(analysis, known_at)
            for known_at in evidence_times if known_at <= evaluation.evaluation_at
        }
        for evidence in all_evidence:
            key = (evidence.dataset_id, evidence.dataset_version,
                   evidence.instrument, evidence.timeframe)
            selected = dataset_by_key.get(key)
            if selected is None:
                findings.append("evaluation evidence references a dataset outside the analysis inputs")
            if evidence.analysis_fingerprint != analysis.fingerprint:
                findings.append("evaluation evidence has a stale analysis fingerprint")
            if evidence.evaluation_prefix_fingerprint != prefix_by_time.get(evidence.known_at):
                findings.append("evaluation evidence has a stale visible-prefix fingerprint")
            if evidence.known_at > evaluation.evaluation_at:
                findings.append("evaluation evidence is known after the candidate evaluation timestamp")
            if evidence.known_at not in observation_times:
                findings.append("evaluation evidence timestamp is not an analysis observation boundary")
            if evidence.bar_open > evidence.known_at:
                findings.append("evaluation evidence refers to a bar that opens after it is known")
            if (evidence.parent_bar_open is not None
                    and evidence.parent_bar_open + evidence.timeframe.nominal_duration > evidence.known_at):
                findings.append("higher-timeframe evidence refers to a parent bar that has not closed")
            if any(timestamp is not None and timestamp > evidence.known_at for timestamp in (
                evidence.supporting_candidate_at, evidence.supporting_confirmed_at,
            )):
                findings.append("evaluation evidence refers to a future supporting swing event")
        if evaluation.status is EvaluationStatus.SETUP_CONFIRMED:
            confirming = [item for item in evaluation.lifecycle_transitions
                          if item.state is LifecycleState.CONFIRMED]
            if not confirming or not any(item.evidence for item in confirming):
                missing.append("confirmed setup has no provenance-linked confirmation evidence")
        if not evaluation.evidence:
            missing.append("strategy evaluation contains no evidence references")

        datasets = tuple(DatasetEvidenceRef(
            dataset_id=item.provenance.dataset_id,
            dataset_version=item.provenance.dataset_version,
            instrument=item.provenance.instrument,
            timeframe=item.provenance.timeframe,
            content_hash=item.provenance.content_hash,
            selected_slice_hash=item.slice_hash,
            provider_id=item.provenance.provider_id,
            provider_version=item.provenance.provider_version,
        ) for item in analysis.inputs)

        config = risk_request.risk_configuration
        risk_ref = RiskEvidenceRef(
            configuration_id=config.configuration_id if config else None,
            configuration_version=config.configuration_version if config else None,
            configuration_fingerprint=config.fingerprint if config else None,
            request_fingerprint=risk_request.fingerprint,
            result_fingerprint=risk_result.fingerprint,
            status=risk_result.status,
            request=risk_request,
            result=risk_result,
        )
        evidence_package = EvidencePackage(
            datasets=datasets,
            analysis=AnalysisEvidenceRef(
                analysis_version=analysis.analysis_version,
                analysis_fingerprint=analysis.fingerprint,
                evaluation_prefix_fingerprint=evaluation.evaluation_prefix_fingerprint,
                cutoff_at=analysis.cutoff_at, input_start=analysis.input_start,
                analysis_start=analysis.analysis_start, analysis_end=analysis.analysis_end,
            ),
            definitions=DefinitionEvidenceRef(
                strategy_id=strategy.strategy_id, strategy_version=strategy.strategy_version,
                strategy_fingerprint=strategy.fingerprint, setup_id=setup.setup_id,
                setup_version=setup.setup_version, setup_fingerprint=setup.fingerprint,
            ),
            evaluation=EvaluationEvidenceRef(
                status=evaluation.status, fingerprint=evaluation.fingerprint,
                request_fingerprint=evaluation.request_fingerprint,
                evaluation_at=evaluation.evaluation_at, source_evidence=evaluation.evidence,
                result=evaluation,
            ),
            risk=risk_ref,
            execution=execution_ref,
            evidence=evaluation.evidence,
        )

        entry, stop, target = risk_request.entry_price, risk_request.stop_price, risk_request.target_price
        side = risk_request.side
        if entry is None or stop is None:
            missing.append("risk request entry or stop price is missing")
        else:
            if entry <= 0 or stop <= 0:
                findings.append("entry and stop prices must be positive")
            if side is not None and ((side.value == "long" and stop >= entry)
                                     or (side.value == "short" and stop <= entry)):
                findings.append("stop price is on the invalid side of entry for the trade direction")
        if target is not None and entry is not None and side is not None:
            if target <= 0:
                findings.append("take-profit price must be positive")
            if (side.value == "long" and target <= entry) or (side.value == "short" and target >= entry):
                findings.append("take-profit price is on the invalid side of entry for the trade direction")

        if risk_result.status is RiskStatus.RISK_VALID:
            if config is None:
                missing.append("risk configuration is absent")
            if risk_request.account_equity is None or risk_request.account_equity <= 0:
                missing.append("positive account equity is absent")
            if risk_result.position_size is None or risk_result.total_estimated_downside is None:
                missing.append("risk result is missing position size or downside")
            if risk_result.position_size is not None and risk_result.position_size <= 0:
                findings.append("risk result quantity must be positive")
            if risk_result.total_estimated_downside is not None and risk_result.total_estimated_downside <= 0:
                findings.append("risk result downside must be positive")
            if entry is not None and stop is not None:
                expected_distance = abs(entry - stop)
                if risk_result.stop_distance != expected_distance:
                    findings.append("risk result stop distance differs from the supplied entry/stop")
                if config is not None and config.minimum_stop_distance is not None and expected_distance < config.minimum_stop_distance:
                    findings.append("stop distance is below the configured minimum")
                if config is not None and config.maximum_stop_distance is not None and expected_distance > config.maximum_stop_distance:
                    findings.append("stop distance exceeds the configured maximum")
            if config is not None and risk_request.account_equity is not None:
                with localcontext(_CALCULATION_CONTEXT):
                    requested_risk = risk_request.account_equity * config.risk_per_trade
                if risk_result.requested_risk != requested_risk:
                    findings.append("risk result budget differs from supplied account equity/configuration")
                if risk_result.maximum_monetary_risk != requested_risk:
                    findings.append("risk result maximum risk differs from supplied account equity/configuration")
                if (risk_result.total_estimated_downside is not None
                        and risk_result.total_estimated_downside > requested_risk):
                    findings.append("risk result downside exceeds its deterministic risk budget")
            if config is not None and risk_result.position_size is not None:
                if config.max_position_size is not None and risk_result.position_size > config.max_position_size:
                    findings.append("risk result quantity exceeds configured maximum position size")
                if (config.max_notional_exposure is not None and risk_result.notional_exposure is not None
                        and risk_result.notional_exposure > config.max_notional_exposure):
                    findings.append("risk result notional exposure exceeds configured maximum")
                if config.max_notional_exposure is not None and risk_result.notional_exposure is None:
                    missing.append("risk result is missing configured notional exposure")
            if any(item.severity in (RiskSeverity.REJECT, RiskSeverity.INSUFFICIENT)
                   for item in risk_result.validation_findings):
                findings.append("risk result is marked valid but contains a rejecting or incomplete finding")
            if target is not None and entry is not None and stop is not None and side is not None:
                with localcontext(_CALCULATION_CONTEXT):
                    risk_distance = abs(entry - stop)
                    reward_distance = target - entry if side.value == "long" else entry - target
                    expected_rr = reward_distance / risk_distance if risk_distance > 0 else None
                if risk_result.reward_risk != expected_rr:
                    findings.append("risk result reward/risk differs from supplied prices")
                if (config is not None and config.minimum_reward_risk is not None
                        and expected_rr is not None and expected_rr < config.minimum_reward_risk):
                    findings.append("reward/risk is below the configured minimum")
            elif risk_result.reward_risk is not None:
                findings.append("risk result contains reward/risk without a take-profit")
            expected_provenance = {
                "configuration_id": config.configuration_id if config else "",
                "configuration_version": str(config.configuration_version) if config else "",
                "instrument": risk_request.instrument.symbol if risk_request.instrument else "",
                "side": side.value if side else "",
                "strategy_id": str(strategy.strategy_id), "setup_id": setup.setup_id,
                "analysis_fingerprint": analysis.fingerprint,
            }
            if any(risk_result.provenance.get(key) != value for key, value in expected_provenance.items()):
                findings.append("risk result provenance does not match the supplied risk request")

        # Non-actionable producer states are preserved instead of being promoted.
        if findings:
            status = CandidateStatus.REJECTED
        elif risk_result.status is RiskStatus.RISK_REJECTED:
            status = CandidateStatus.REJECTED
        elif evaluation.status in (EvaluationStatus.INVALIDATED, EvaluationStatus.EXPIRED,
                                   EvaluationStatus.SETUP_NOT_CONFIRMED):
            status = CandidateStatus.WAIT
        elif evaluation.status is EvaluationStatus.INSUFFICIENT_EVIDENCE:
            status = CandidateStatus.INSUFFICIENT_EVIDENCE
        elif risk_result.status is RiskStatus.INSUFFICIENT_EVIDENCE or missing:
            status = CandidateStatus.INSUFFICIENT_EVIDENCE
        elif evaluation.status is EvaluationStatus.SETUP_CONFIRMED and risk_result.status is RiskStatus.RISK_VALID:
            status = CandidateStatus.ACTIONABLE
        else:
            status = CandidateStatus.WAIT

        if evaluation.status in (EvaluationStatus.INVALIDATED, EvaluationStatus.EXPIRED):
            transition = next((item for item in reversed(evaluation.lifecycle_transitions)
                               if item.state in (LifecycleState.INVALIDATED, LifecycleState.EXPIRED)), None)
            expiry_at = transition.known_at if transition and transition.state is LifecycleState.EXPIRED else None
        else:
            expiry_at = None
        account_equity = risk_request.account_equity
        risk_amount = risk_result.total_estimated_downside
        with localcontext(_CALCULATION_CONTEXT):
            risk_percentage = (risk_amount / account_equity
                               if risk_amount is not None and account_equity is not None and account_equity > 0
                               else None)
        target_values = (target,) if target is not None else ()
        fields = {
            "schema_version": "2i.1.0", "candidate_version": 1,
            "instrument": setup.instrument, "direction": side,
            "entry_price": entry, "stop_loss": stop, "take_profits": target_values,
            "quantity": risk_result.position_size,
            "account_currency": config.account_currency if config else None,
            "risk_amount": risk_amount, "risk_percentage": risk_percentage,
            "expected_reward_risk": risk_result.reward_risk,
            "strategy_id": strategy.strategy_id, "strategy_version": strategy.strategy_version,
            "strategy_fingerprint": strategy.fingerprint, "setup_id": setup.setup_id,
            "setup_version": setup.setup_version, "setup_fingerprint": setup.fingerprint,
            "analysis_version": analysis.analysis_version, "analysis_fingerprint": analysis.fingerprint,
            "dataset_id": primary.dataset_id if primary else None,
            "dataset_version": primary.dataset_version if primary else None,
            "primary_timeframe": setup.primary_timeframe,
            "risk_configuration_id": config.configuration_id if config else None,
            "risk_configuration_version": config.configuration_version if config else None,
            "risk_configuration_fingerprint": config.fingerprint if config else None,
            "execution_assumptions_fingerprint": execution_ref.fingerprint if execution_ref else None,
            "evaluation_at": evaluation.evaluation_at, "expiry_at": expiry_at,
            "status": status, "findings": tuple(findings + missing), "evidence": evidence_package,
        }
        candidate_hash = candidate_fingerprint(fields)
        return TradeCandidate(
            candidate_id=candidate_id_for(candidate_hash), fingerprint=candidate_hash, **fields
        )
