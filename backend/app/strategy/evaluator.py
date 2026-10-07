"""Pure replay evaluator for a versioned setup over Phase 2E AnalysisResult."""

from datetime import datetime

from app.market_analysis.contracts import AnalysisResult, ConfirmedSwing, SwingKind
from app.market_context.sessions import CalendarStatus
from app.market_data.contracts import TimestampConvention
from app.strategy.conditions import ResolvedField, evaluate_condition
from app.strategy.contracts import (
    Compare, ConditionTrace, EvaluationRequest, EvaluationResult,
    EvaluationStatus, EvidenceReference, FieldRef, LifecycleState,
    LifecycleTransition, StrategyDefinition, SetupDefinition, TruthValue,
)
from app.strategy.errors import StrategyEvaluationError
from app.strategy.fingerprints import (
    analysis_full_fingerprint, analysis_prefix_fingerprint, fingerprint,
)


def _last_swing(swings: tuple[ConfirmedSwing, ...], kind: SwingKind):
    matching = [item for item in swings if item.kind is kind]
    return max(matching, key=lambda item: (item.known_at, item.candidate_at)) if matching else None


def _resolve_field(observation, node: Compare, primary_timeframe):
    field = node.field
    if field.value.startswith("higher."):
        parent = next((item for item in observation.higher_timeframes if item.timeframe == node.timeframe), None)
        if parent is None:
            return ResolvedField(None, "required higher-timeframe context is absent", node.timeframe,
                                 known_at=observation.known_at)
        if parent.unavailable_reason is not None or parent.bar_open is None or parent.close is None:
            return ResolvedField(None, parent.unavailable_reason.value if parent.unavailable_reason else "no closed parent bar",
                                 node.timeframe, known_at=observation.known_at)
        parent_close = parent.bar_open + node.timeframe.nominal_duration
        if parent_close > observation.known_at or parent.known_at > observation.known_at:
            return ResolvedField(None, "parent candle is not yet knowable", node.timeframe,
                                 parent.bar_open, observation.known_at)
        eligible_swings = tuple(
            swing for swing in parent.confirmed_swings
            if swing.known_at <= observation.known_at and swing.known_at <= parent_close
        )
        high_swing = _last_swing(eligible_swings, SwingKind.HIGH)
        low_swing = _last_swing(eligible_swings, SwingKind.LOW)
        if field is FieldRef.HIGHER_CLOSE:
            value = parent.close
        elif field is FieldRef.HIGHER_STRUCTURAL_STATE:
            value = parent.structural_state
        elif field is FieldRef.HIGHER_LAST_SWING_HIGH_PRICE:
            value = high_swing.price if high_swing else None
        elif field is FieldRef.HIGHER_LAST_SWING_HIGH_KNOWN_AT:
            value = high_swing.known_at if high_swing else None
        elif field is FieldRef.HIGHER_LAST_SWING_LOW_PRICE:
            value = low_swing.price if low_swing else None
        elif field is FieldRef.HIGHER_LAST_SWING_LOW_KNOWN_AT:
            value = low_swing.known_at if low_swing else None
        else:
            value = None
        if isinstance(value, datetime) and value > observation.known_at:
            value = None
        return ResolvedField(value, "no confirmed parent swing" if value is None else None,
                             node.timeframe, parent.bar_open, observation.known_at,
                             high_swing.candidate_at if field is FieldRef.HIGHER_LAST_SWING_HIGH_PRICE and high_swing else
                             low_swing.candidate_at if field is FieldRef.HIGHER_LAST_SWING_LOW_PRICE and low_swing else None,
                             high_swing.confirmed_at if field in (FieldRef.HIGHER_LAST_SWING_HIGH_PRICE,
                                                                  FieldRef.HIGHER_LAST_SWING_HIGH_KNOWN_AT) and high_swing else
                             low_swing.confirmed_at if field in (FieldRef.HIGHER_LAST_SWING_LOW_PRICE,
                                                                 FieldRef.HIGHER_LAST_SWING_LOW_KNOWN_AT) and low_swing else None)

    if (field is FieldRef.STRUCTURAL_STATE and observation.structural_transition is not None
            and observation.structural_transition.known_at > observation.known_at):
        return ResolvedField(None, "structural transition is not yet knowable", primary_timeframe,
                             observation.bar_open, observation.known_at)
    swings = tuple(item for item in observation.confirmed_swings if item.known_at <= observation.known_at)
    high_swing, low_swing = _last_swing(swings, SwingKind.HIGH), _last_swing(swings, SwingKind.LOW)
    values = {
        FieldRef.OPEN: observation.open, FieldRef.HIGH: observation.high, FieldRef.LOW: observation.low,
        FieldRef.CLOSE: observation.close, FieldRef.VOLUME: observation.volume, FieldRef.BID: observation.bid,
        FieldRef.ASK: observation.ask, FieldRef.SPREAD: observation.spread,
        FieldRef.CANDLE_RANGE: observation.features.range, FieldRef.CANDLE_BODY: observation.features.body,
        FieldRef.UPPER_WICK: observation.features.upper_wick, FieldRef.LOWER_WICK: observation.features.lower_wick,
        FieldRef.BODY_RATIO: observation.features.body_ratio.value,
        FieldRef.CLOSE_LOCATION: observation.features.close_location.value,
        FieldRef.SIMPLE_RETURN: observation.features.simple_return.value,
        FieldRef.TRUE_RANGE: observation.true_range, FieldRef.ATR: observation.atr.value,
        FieldRef.STRUCTURAL_STATE: observation.structural_state,
        FieldRef.LAST_SWING_HIGH_PRICE: high_swing.price if high_swing else None,
        FieldRef.LAST_SWING_HIGH_CANDIDATE_AT: high_swing.candidate_at if high_swing else None,
        FieldRef.LAST_SWING_HIGH_KNOWN_AT: high_swing.known_at if high_swing else None,
        FieldRef.LAST_SWING_LOW_PRICE: low_swing.price if low_swing else None,
        FieldRef.LAST_SWING_LOW_CANDIDATE_AT: low_swing.candidate_at if low_swing else None,
        FieldRef.LAST_SWING_LOW_KNOWN_AT: low_swing.known_at if low_swing else None,
        FieldRef.SESSION_WEEKEND: observation.session.weekend,
        FieldRef.SESSION_CALENDAR_STATUS: observation.session.calendar_status,
        FieldRef.SESSION_LABEL_COUNT: len(observation.session.labels),
        FieldRef.SESSION_ENTERED_COUNT: len(observation.session_transition.entered) if observation.session_transition else None,
        FieldRef.SESSION_EXITED_COUNT: len(observation.session_transition.exited) if observation.session_transition else None,
    }
    value = values[field]
    reason = None
    if field is FieldRef.BODY_RATIO and value is None:
        reason = observation.features.body_ratio.unavailable_reason.value
    elif field is FieldRef.CLOSE_LOCATION and value is None:
        reason = observation.features.close_location.unavailable_reason.value
    elif field is FieldRef.SIMPLE_RETURN and value is None:
        reason = observation.features.simple_return.unavailable_reason.value
    elif field is FieldRef.ATR and value is None:
        reason = observation.atr.unavailable_reason.value
    elif value is None:
        reason = "optional value or confirmed swing is unavailable"
    swing = high_swing if field.value.startswith("swing.last_high") else low_swing if field.value.startswith("swing.last_low") else None
    if field is FieldRef.STRUCTURAL_STATE and observation.structural_transition is not None:
        transition = observation.structural_transition
        swing = next((item for item in swings if item.kind is transition.responsible_swing
                      and item.known_at == transition.known_at), None)
    return ResolvedField(value, reason, primary_timeframe, observation.bar_open, observation.known_at,
                         swing.candidate_at if swing else None, swing.confirmed_at if swing else None)


def _make_evidence(node, resolved, comparison, outcome, analysis, provenance, observation):
    tf = resolved.timeframe or provenance.timeframe
    source = next((item.provenance for item in analysis.inputs if item.provenance.timeframe == tf), None)
    if source is None:
        # Preserve an auditable missing-source reference using the primary dataset identity.
        source = provenance
    unavailable = resolved.unavailable_reason if outcome is TruthValue.INSUFFICIENT else None
    value = resolved.value if unavailable is None else None
    return EvidenceReference(
        dataset_id=source.dataset_id, dataset_version=source.dataset_version,
        instrument=source.instrument, timeframe=tf, bar_open=resolved.bar_open or observation.bar_open,
        known_at=resolved.known_at or observation.known_at, analysis_version=analysis.analysis_version,
        analysis_fingerprint=analysis.fingerprint,
        evaluation_prefix_fingerprint=analysis_prefix_fingerprint(analysis, observation.known_at),
        field=node.field, value=value,
        unavailable_reason=unavailable, condition_id=fingerprint(node, domain="condition")[:16],
        operator=node.operator, comparison_value=comparison,
        parent_bar_open=resolved.bar_open if tf != provenance.timeframe else None,
        supporting_candidate_at=resolved.candidate_at, supporting_confirmed_at=resolved.confirmed_at,
    )


def evaluate_setup(analysis: AnalysisResult, strategy: StrategyDefinition, setup: SetupDefinition,
                   request: EvaluationRequest) -> EvaluationResult:
    """Replay the exact setup through the observation named by evaluation_at."""
    primary = next((item.provenance for item in analysis.inputs
                    if item.provenance.timeframe == setup.primary_timeframe), None)
    if primary is None or primary.instrument != setup.instrument:
        raise StrategyEvaluationError("Analysis provenance does not contain the setup instrument/timeframe")
    input_timeframes = [item.provenance.timeframe for item in analysis.inputs]
    if len(input_timeframes) != len(set(input_timeframes)):
        raise StrategyEvaluationError("Analysis inputs contain duplicate timeframes")
    if any(item.provenance.instrument != setup.instrument for item in analysis.inputs):
        raise StrategyEvaluationError("Analysis provenance instrument differs from the setup")
    if any(item.provenance.timestamp_convention is not TimestampConvention.BAR_OPEN for item in analysis.inputs):
        raise StrategyEvaluationError("Strategy evaluation requires Phase 2E BAR-OPEN provenance")
    if analysis.analysis_version != strategy.analysis_version:
        raise StrategyEvaluationError("Strategy and AnalysisResult versions do not match")
    actual_analysis_fingerprint = analysis_full_fingerprint(analysis)
    if analysis.fingerprint != actual_analysis_fingerprint:
        raise StrategyEvaluationError("AnalysisResult fingerprint does not match its supplied content")
    visible_prefix = analysis_prefix_fingerprint(analysis, request.evaluation_at)
    expected = (strategy.strategy_id, strategy.strategy_version, strategy.fingerprint,
                setup.setup_id, setup.setup_version, setup.fingerprint, analysis.fingerprint,
                visible_prefix)
    actual = (request.strategy_id, request.strategy_version, request.strategy_fingerprint,
              request.setup_id, request.setup_version, request.setup_fingerprint, request.analysis_fingerprint,
              request.evaluation_prefix_fingerprint)
    if expected != actual:
        raise StrategyEvaluationError("EvaluationRequest does not identify the exact supplied inputs")
    if strategy.setup(setup.setup_id, setup.setup_version).fingerprint != setup.fingerprint:
        raise StrategyEvaluationError("Setup is not the exact setup contained by StrategyDefinition")
    if any(tf not in {item.provenance.timeframe for item in analysis.inputs}
           for tf in setup.required_higher_timeframes):
        raise StrategyEvaluationError("Analysis provenance is missing a required higher timeframe")
    if not (analysis.input_start <= analysis.analysis_start < analysis.analysis_end):
        raise StrategyEvaluationError("AnalysisResult has an invalid declared observation range")
    for item in analysis.observations:
        if (item.known_at.tzinfo is None or item.known_at.utcoffset() is None
                or item.bar_open.tzinfo is None or item.bar_open.utcoffset() is None):
            raise StrategyEvaluationError("Primary timestamps must be timezone-aware")
        if not analysis.analysis_start <= item.bar_open < analysis.analysis_end:
            raise StrategyEvaluationError("Analysis observations must remain inside the declared analysis range")
        if item.known_at != item.bar_open + setup.primary_timeframe.nominal_duration:
            raise StrategyEvaluationError("Primary observation known_at does not match its fixed-duration close")
        if item.known_at > analysis.cutoff_at:
            raise StrategyEvaluationError("AnalysisResult contains an observation beyond its declared cutoff")
    exact = [item for item in analysis.observations if item.known_at == request.evaluation_at]
    if len(exact) != 1:
        raise StrategyEvaluationError("evaluation_at must exactly identify one primary observation known_at")
    eligible = tuple(item for item in analysis.observations if item.known_at <= request.evaluation_at)
    if tuple(sorted(eligible, key=lambda item: (item.known_at, item.bar_open))) != eligible:
        raise StrategyEvaluationError("Analysis observations must be deterministically ordered")
    if len({item.known_at for item in eligible}) != len(eligible):
        raise StrategyEvaluationError("Primary observation known_at timestamps must be unique")

    trace, all_evidence, transitions = [], [], []
    state = None
    candidate_index = None
    candidate_evidence = ()
    previous_activation = None
    final_outcome = TruthValue.INSUFFICIENT
    final_status = EvaluationStatus.INSUFFICIENT_EVIDENCE

    def check(condition, obs):
        if condition is None:
            return TruthValue.TRUE, ()
        outcome, leaves = evaluate_condition(
            condition, strategy,
            lambda node: _resolve_field(obs, node, setup.primary_timeframe),
        )
        evidence = tuple(_make_evidence(node, resolved, compare, leaf_outcome, analysis, primary, obs)
                         for node, resolved, compare, leaf_outcome in leaves)
        trace.append(ConditionTrace(
            condition_id=fingerprint(condition, domain="condition")[:16], known_at=obs.known_at,
            outcome=outcome, evidence=evidence,
        ))
        all_evidence.extend(evidence)
        return outcome, evidence

    def context_check(obs):
        references = []
        outcomes = []
        for label in setup.required_context.session_labels:
            matches = label in obs.session.labels
            outcomes.append(TruthValue.TRUE if matches else TruthValue.FALSE)
            references.append(EvidenceReference(
                dataset_id=primary.dataset_id, dataset_version=primary.dataset_version,
                instrument=primary.instrument, timeframe=primary.timeframe,
                bar_open=obs.bar_open, known_at=obs.known_at,
                analysis_version=analysis.analysis_version, analysis_fingerprint=analysis.fingerprint,
                evaluation_prefix_fingerprint=analysis_prefix_fingerprint(analysis, obs.known_at),
                field=FieldRef.SESSION_LABELS, value=obs.session.labels,
                condition_id=fingerprint({"session_label": label}, domain="context-condition")[:16],
                context_requirement=f"session label contains {label}", comparison_value=label,
            ))
        required_status = setup.required_context.calendar_status
        if required_status is not None:
            actual_status = obs.session.calendar_status
            outcome = (TruthValue.INSUFFICIENT if actual_status is CalendarStatus.UNKNOWN
                       and required_status is not CalendarStatus.UNKNOWN
                       else TruthValue.TRUE if actual_status is required_status else TruthValue.FALSE)
            outcomes.append(outcome)
            references.append(EvidenceReference(
                dataset_id=primary.dataset_id, dataset_version=primary.dataset_version,
                instrument=primary.instrument, timeframe=primary.timeframe,
                bar_open=obs.bar_open, known_at=obs.known_at,
                analysis_version=analysis.analysis_version, analysis_fingerprint=analysis.fingerprint,
                evaluation_prefix_fingerprint=analysis_prefix_fingerprint(analysis, obs.known_at),
                field=FieldRef.SESSION_CALENDAR_STATUS,
                value=None if outcome is TruthValue.INSUFFICIENT else actual_status,
                unavailable_reason="calendar status is unknown" if outcome is TruthValue.INSUFFICIENT else None,
                condition_id=fingerprint({"calendar_status": required_status}, domain="context-condition")[:16],
                context_requirement=f"calendar status equals {required_status.value}",
                comparison_value=required_status,
            ))
        references = tuple(references)
        if references:
            aggregate = TruthValue.FALSE if TruthValue.FALSE in outcomes else (
                TruthValue.INSUFFICIENT if TruthValue.INSUFFICIENT in outcomes else TruthValue.TRUE
            )
            trace.append(ConditionTrace(
                condition_id=fingerprint(setup.required_context, domain="required-context")[:16],
                known_at=obs.known_at, outcome=aggregate, evidence=references,
            ))
            all_evidence.extend(references)
        if TruthValue.FALSE in outcomes:
            return TruthValue.FALSE
        if TruthValue.INSUFFICIENT in outcomes:
            return TruthValue.INSUFFICIENT
        return TruthValue.TRUE

    for index, obs in enumerate(eligible):
        if state in (LifecycleState.INVALIDATED, LifecycleState.EXPIRED):
            break
        context_outcome = context_check(obs)
        prereq, _ = check(setup.prerequisites, obs)
        if context_outcome is not TruthValue.TRUE:
            prereq = context_outcome

        if state in (LifecycleState.CANDIDATE, LifecycleState.CONFIRMED):
            invalid, invalid_evidence = (check(setup.invalidation_conditions, obs)
                                         if setup.invalidation_conditions is not None
                                         else (TruthValue.FALSE, ()))
            if invalid is TruthValue.TRUE:
                state = LifecycleState.INVALIDATED
                transitions.append(LifecycleTransition(state=state, known_at=obs.known_at,
                                                       reason="invalidation condition became true", evidence=invalid_evidence))
                continue
            if state is LifecycleState.CANDIDATE:
                later_count = index - candidate_index
                if later_count >= setup.expiry.max_later_observations:
                    state = LifecycleState.EXPIRED
                    transitions.append(LifecycleTransition(state=state, known_at=obs.known_at,
                                                           reason="maximum later primary observations reached",
                                                           evidence=candidate_evidence))
                    continue
                if prereq is not TruthValue.TRUE:
                    final_outcome = prereq
                    continue
                if invalid is TruthValue.INSUFFICIENT:
                    final_outcome = TruthValue.INSUFFICIENT
                    continue
                if setup.confirmation_conditions is not None:
                    confirmation, confirmation_evidence = check(setup.confirmation_conditions, obs)
                    if confirmation is TruthValue.TRUE:
                        state = LifecycleState.CONFIRMED
                        final_outcome = TruthValue.TRUE
                        transitions.append(LifecycleTransition(state=state, known_at=obs.known_at,
                                                               reason="confirmation conditions became true",
                                                               evidence=confirmation_evidence))
                    else:
                        final_outcome = confirmation
            else:
                final_outcome = TruthValue.INSUFFICIENT if invalid is TruthValue.INSUFFICIENT else TruthValue.TRUE
            continue

        entry, entry_evidence = check(setup.entry_conditions, obs)
        activation = (TruthValue.FALSE if TruthValue.FALSE in (prereq, entry) else
                      TruthValue.TRUE if prereq is TruthValue.TRUE and entry is TruthValue.TRUE else
                      TruthValue.INSUFFICIENT)
        if activation is TruthValue.TRUE and previous_activation is TruthValue.FALSE:
            state = LifecycleState.CANDIDATE
            candidate_index = index
            candidate_evidence = entry_evidence
            transitions.append(LifecycleTransition(state=state, known_at=obs.known_at,
                                                   reason="entry conditions became true", evidence=entry_evidence))
            if setup.confirmation_conditions is None:
                state = LifecycleState.CONFIRMED
                final_outcome = TruthValue.TRUE
                transitions.append(LifecycleTransition(state=state, known_at=obs.known_at,
                                                       reason="confirmation is not required", evidence=entry_evidence))
        elif activation is TruthValue.TRUE:
            # The qualifying condition is already true at the left edge of the supplied range.
            final_outcome = TruthValue.INSUFFICIENT
        else:
            final_outcome = activation
        previous_activation = activation

    if state is LifecycleState.INVALIDATED:
        final_status = EvaluationStatus.INVALIDATED
    elif state is LifecycleState.EXPIRED:
        final_status = EvaluationStatus.EXPIRED
    elif state is LifecycleState.CONFIRMED and final_outcome is TruthValue.INSUFFICIENT:
        final_status = EvaluationStatus.INSUFFICIENT_EVIDENCE
    elif state is LifecycleState.CONFIRMED:
        final_status = EvaluationStatus.SETUP_CONFIRMED
    elif final_outcome is TruthValue.INSUFFICIENT:
        final_status = EvaluationStatus.INSUFFICIENT_EVIDENCE
    else:
        final_status = EvaluationStatus.SETUP_NOT_CONFIRMED

    def decision_evidence(item):
        # Full-analysis identity is retained in records but intentionally excluded from
        # the suffix-invariant decision digest; the visible-prefix identity remains bound.
        return item.model_dump(mode="python", exclude={"analysis_fingerprint"})

    decision_trace = tuple({
        "condition_id": item.condition_id, "known_at": item.known_at, "outcome": item.outcome,
        "evidence": tuple(decision_evidence(evidence) for evidence in item.evidence),
    } for item in trace)
    payload = {
        "request_fingerprint": request.fingerprint, "status": final_status,
        "condition_trace": decision_trace,
        "evidence": tuple(decision_evidence(item) for item in all_evidence),
        "lifecycle_transitions": tuple({
            "state": item.state, "known_at": item.known_at, "reason": item.reason,
            "evidence": tuple(decision_evidence(evidence) for evidence in item.evidence),
        } for item in transitions),
    }
    result_fingerprint = fingerprint(payload, domain="evaluation-result")
    return EvaluationResult(
        status=final_status, strategy_id=strategy.strategy_id, strategy_version=strategy.strategy_version,
        strategy_fingerprint=strategy.fingerprint, setup_id=setup.setup_id, setup_version=setup.setup_version,
        setup_fingerprint=setup.fingerprint, analysis_fingerprint=analysis.fingerprint,
        evaluation_prefix_fingerprint=visible_prefix,
        evaluation_at=request.evaluation_at, request_fingerprint=request.fingerprint,
        condition_trace=tuple(trace), evidence=tuple(all_evidence), lifecycle_transitions=tuple(transitions),
        fingerprint=result_fingerprint,
    )
