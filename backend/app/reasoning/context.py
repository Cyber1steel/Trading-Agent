"""Build a cutoff-safe reasoning context from existing deterministic artifacts."""

from hashlib import sha256

from app.backtesting.contracts import BacktestResult
from app.backtesting.fingerprints import fingerprint as backtest_fingerprint
from app.knowledge.embedding.contracts import RetrievalResult
from app.market_analysis.contracts import AnalysisResult
from app.market_data.contracts import require_aware_utc
from app.reasoning.contracts import (
    CandidateReference,
    CandidateStatus,
    EvidenceFact,
    EvidenceTier,
    HistoricalValidationEvidence,
    KnowledgeEvidence,
    MarketObservationEvidence,
    ReasoningContext,
)
from app.reasoning.errors import ReasoningContextError
from app.reasoning.fingerprints import digest
from app.strategy.fingerprints import analysis_full_fingerprint
from app.strategy.fingerprints import analysis_prefix_fingerprint
from app.trade_candidate.contracts import TradeCandidate


def _fact_id(kind: str, value: object) -> str:
    return f"det-{kind}-{digest(value, domain='reasoning-fact')[:24]}"


def _candidate_facts(candidate: TradeCandidate) -> tuple[EvidenceFact, ...]:
    facts: list[EvidenceFact] = []

    def add(kind: str, label: str, value=None, known_at=None, artifact_reference=None):
        facts.append(EvidenceFact(
            evidence_id=_fact_id(kind, {"label": label, "value": value, "known_at": known_at}),
            tier=EvidenceTier.DETERMINISTIC_STRATEGY_RISK,
            kind=kind,
            known_at=known_at,
            artifact_reference=artifact_reference or candidate.fingerprint,
            label=label,
            value=value,
        ))

    add("candidate_status", "Deterministic candidate status", candidate.status.value, candidate.evaluation_at)
    add("candidate_identity", "Candidate identity", candidate.candidate_id, candidate.evaluation_at)
    add("instrument", "Instrument", candidate.instrument.symbol, candidate.evaluation_at)
    add("direction", "Candidate direction", candidate.direction.value if candidate.direction else None, candidate.evaluation_at)
    add("entry", "Deterministic entry price", candidate.entry_price, candidate.evaluation_at)
    add("stop", "Deterministic stop price", candidate.stop_loss, candidate.evaluation_at)
    add("quantity", "Deterministic quantity", candidate.quantity, candidate.evaluation_at)
    add("risk_amount", "Deterministic risk amount", candidate.risk_amount, candidate.evaluation_at)
    add("risk_percentage", "Deterministic risk percentage", candidate.risk_percentage, candidate.evaluation_at)
    add("reward_risk", "Deterministic reward/risk", candidate.expected_reward_risk, candidate.evaluation_at)
    for index, target in enumerate(candidate.take_profits):
        add("take_profit", f"Deterministic take-profit {index + 1}", target, candidate.evaluation_at)
    add("dataset", "Dataset identity", f"{candidate.dataset_id}:{candidate.dataset_version}", candidate.evaluation_at)
    add("analysis", "Analysis fingerprint", candidate.analysis_fingerprint, candidate.evaluation_at)
    add("strategy", "Strategy identity/version", f"{candidate.strategy_id}:{candidate.strategy_version}", candidate.evaluation_at)
    add("setup", "Setup identity/version", f"{candidate.setup_id}:{candidate.setup_version}", candidate.evaluation_at)
    add("risk_artifact", "Risk result fingerprint", candidate.evidence.risk.result_fingerprint, candidate.evaluation_at)
    add("execution_assumptions", "Execution assumptions fingerprint", candidate.execution_assumptions_fingerprint, candidate.evaluation_at)
    for index, ref in enumerate(candidate.evidence.evidence):
        known_at = require_aware_utc(ref.known_at, "evidence known_at")
        value = ref.value if ref.value is not None else ref.unavailable_reason
        add("strategy_evidence", f"Strategy evidence {index + 1}: {ref.condition_id}/{ref.field.value}", value, known_at)
    return tuple(sorted(facts, key=lambda item: (item.known_at or candidate.evaluation_at, item.evidence_id)))


def _market_facts(observations, analysis_fingerprint: str):
    facts = []
    for item in observations:
        observation = item.observation
        known = observation.known_at
        base = f"{item.timeframe} {observation.bar_open.isoformat()}"
        values = [
            ("market_open", "Open", observation.open),
            ("market_high", "High", observation.high),
            ("market_low", "Low", observation.low),
            ("market_close", "Close", observation.close),
            ("true_range", "True range", observation.true_range),
            ("atr", "ATR", observation.atr.value),
            ("structure", "Structural state", observation.structural_state.value),
        ]
        for swing in observation.confirmed_swings:
            if swing.known_at > known:
                raise ReasoningContextError("swing confirmation is known after its primary observation")
            values.append((
                "confirmed_swing",
                f"{swing.kind.value} swing candidate {swing.candidate_at.isoformat()} confirmed {swing.confirmed_at.isoformat()}",
                swing.price,
            ))
        if observation.structural_transition is not None:
            if observation.structural_transition.known_at > known:
                raise ReasoningContextError("structure transition is known after its primary observation")
            values.append((
                "structure_transition",
                f"Structure transition {observation.structural_transition.previous_state.value} to {observation.structural_transition.new_state.value}",
                observation.structural_transition.known_at,
            ))
        for kind, label, value in values:
            if value is None:
                continue
            facts.append(EvidenceFact(
                evidence_id=f"det-{kind}-{digest({'analysis': analysis_fingerprint, 'bar': base, 'label': label, 'value': value}, domain='market-fact')[:24]}",
                tier=EvidenceTier.DETERMINISTIC_MARKET,
                kind=kind,
                known_at=known,
                artifact_reference=analysis_fingerprint,
                label=f"{base}: {label}",
                value=value,
            ))
        for parent in observation.higher_timeframes:
            if parent.known_at > known:
                raise ReasoningContextError("higher-timeframe evidence is known after its primary observation")
            if parent.close is not None:
                facts.append(EvidenceFact(
                    evidence_id=f"det-higher-close-{digest({'analysis': analysis_fingerprint, 'bar': base, 'timeframe': parent.timeframe.value, 'close': parent.close}, domain='market-fact')[:24]}",
                    tier=EvidenceTier.DETERMINISTIC_MARKET,
                    kind="higher_timeframe_close",
                    known_at=parent.known_at,
                    artifact_reference=analysis_fingerprint,
                    label=f"{base}: {parent.timeframe.value} parent close",
                    value=parent.close,
                ))
            facts.append(EvidenceFact(
                evidence_id=f"det-higher-state-{digest({'analysis': analysis_fingerprint, 'bar': base, 'timeframe': parent.timeframe.value, 'state': parent.structural_state.value}, domain='market-fact')[:24]}",
                tier=EvidenceTier.DETERMINISTIC_MARKET,
                kind="higher_timeframe_structure",
                known_at=parent.known_at,
                artifact_reference=analysis_fingerprint,
                label=f"{base}: {parent.timeframe.value} parent structure",
                value=parent.structural_state.value,
            ))
            for swing in parent.confirmed_swings:
                if swing.known_at > parent.known_at:
                    raise ReasoningContextError("higher-timeframe swing is known after its parent observation")
                facts.append(EvidenceFact(
                    evidence_id=f"det-higher-swing-{digest({'analysis': analysis_fingerprint, 'bar': base, 'timeframe': parent.timeframe.value, 'kind': swing.kind.value, 'candidate': swing.candidate_at, 'price': swing.price}, domain='market-fact')[:24]}",
                    tier=EvidenceTier.DETERMINISTIC_MARKET,
                    kind="higher_timeframe_confirmed_swing",
                    known_at=swing.known_at,
                    artifact_reference=analysis_fingerprint,
                    label=f"{base}: {parent.timeframe.value} confirmed {swing.kind.value} swing",
                    value=swing.price,
                ))
    return tuple(facts)


def _knowledge_evidence(results: tuple[RetrievalResult, ...], evaluation_time) -> tuple[KnowledgeEvidence, ...]:
    unique: dict[str, tuple[int, RetrievalResult, str]] = {}
    for rank, result in enumerate(results, start=1):
        if result.published_at is None:
            continue
        if require_aware_utc(result.published_at, "published_at") > evaluation_time:
            continue
        digest = sha256(result.text.encode("utf-8")).hexdigest()
        if result.content_sha256 != digest:
            raise ReasoningContextError("retrieval result did not preserve a validated canonical content hash")
        existing = unique.get(result.chunk_id)
        if existing is not None:
            old = existing[1]
            if (old.source_id, old.text, old.file_path, old.page_number) != (
                result.source_id, result.text, result.file_path, result.page_number
            ):
                raise ReasoningContextError("duplicate retrieved chunk has conflicting provenance/content")
            continue
        unique[result.chunk_id] = (rank, result, digest)

    converted = []
    for rank, result, digest in unique.values():
        converted.append(KnowledgeEvidence(
            evidence_id=f"knowledge:{result.chunk_id}",
            source_id=result.source_id,
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            source_type=result.source_type,
            title=result.title,
            file_path=result.file_path,
            source_url=result.source_url,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            char_start=result.char_start,
            char_end=result.char_end,
            published_at=result.published_at,
            retrieved_content_sha256=digest,
            retrieval_mode=result.retrieval_mode,
            retrieval_rank=rank,
            semantic_rank=result.semantic_rank,
            lexical_rank=result.lexical_rank,
            distance=result.distance,
            lexical_score=result.lexical_score,
            hybrid_score=result.hybrid_score,
            embedding_provider=result.embedding_provider,
            embedding_model=result.embedding_model,
            embedding_dimension=result.embedding_dimension,
            text=result.text,
        ))
    return tuple(sorted(converted, key=lambda item: (item.retrieval_rank, item.chunk_id)))


class ReasoningContextBuilder:
    """Validate identities and construct only cutoff-visible, typed evidence."""

    def build(
        self,
        candidate: TradeCandidate,
        *,
        evaluation_time,
        analysis: AnalysisResult | None = None,
        knowledge: tuple[RetrievalResult, ...] = (),
        backtest: BacktestResult | None = None,
        backtest_available_at=None,
    ) -> ReasoningContext:
        if not isinstance(candidate, TradeCandidate):
            raise ReasoningContextError("a validated TradeCandidate is required")
        try:
            candidate = TradeCandidate.model_validate(candidate.model_dump(mode="python"))
        except Exception as exc:
            raise ReasoningContextError("candidate failed integrity revalidation") from exc
        try:
            cutoff = require_aware_utc(evaluation_time, "evaluation_time")
        except ValueError as exc:
            raise ReasoningContextError(str(exc)) from exc
        if candidate.evaluation_at > cutoff:
            raise ReasoningContextError("candidate was evaluated after the reasoning cutoff")
        observations = ()
        if analysis is not None:
            if not isinstance(analysis, AnalysisResult):
                raise ReasoningContextError("analysis must use the Phase 2E AnalysisResult contract")
            try:
                analysis = AnalysisResult.model_validate(analysis.model_dump(mode="python"))
            except Exception as exc:
                raise ReasoningContextError("analysis failed contract revalidation") from exc
            if analysis.fingerprint != candidate.analysis_fingerprint:
                raise ReasoningContextError("analysis fingerprint does not match candidate provenance")
            if analysis_full_fingerprint(analysis) != analysis.fingerprint:
                raise ReasoningContextError("analysis content does not match its fingerprint")
            primary = next((item.provenance for item in analysis.inputs
                            if item.provenance.timeframe == candidate.primary_timeframe), None)
            if primary is None or primary.dataset_id != candidate.dataset_id or primary.dataset_version != candidate.dataset_version:
                raise ReasoningContextError("primary analysis dataset identity does not match candidate")
            if analysis.cutoff_at != candidate.evidence.analysis.cutoff_at:
                raise ReasoningContextError("analysis cutoff does not match candidate provenance")
            if analysis_prefix_fingerprint(analysis, candidate.evaluation_at) != candidate.evidence.analysis.evaluation_prefix_fingerprint:
                raise ReasoningContextError("analysis visible prefix does not match candidate provenance")
            visible = [item for item in analysis.observations if item.known_at <= cutoff]
            observations = tuple(MarketObservationEvidence.from_observation(item, candidate.primary_timeframe.value)
                                for item in visible)

        if not isinstance(knowledge, tuple):
            raise ReasoningContextError("knowledge results must be an immutable tuple")
        for item in knowledge:
            if not isinstance(item, RetrievalResult):
                raise ReasoningContextError("knowledge must come from the Phase 2C RetrievalResult contract")
        try:
            knowledge = tuple(
                RetrievalResult.model_validate(item.model_dump(mode="python")) for item in knowledge
            )
        except Exception as exc:
            raise ReasoningContextError("retrieval result failed contract revalidation") from exc
        knowledge_refs = _knowledge_evidence(knowledge, cutoff)

        historical = None
        if backtest is not None:
            if not isinstance(backtest, BacktestResult):
                raise ReasoningContextError("backtest must use the Phase 2G BacktestResult contract")
            try:
                backtest = BacktestResult.model_validate(backtest.model_dump(mode="python"))
            except Exception as exc:
                raise ReasoningContextError("backtest failed contract revalidation") from exc
            if backtest_available_at is None:
                raise ReasoningContextError("backtest availability time is required; the producer artifact has no run timestamp")
            backtest_available_at = require_aware_utc(backtest_available_at, "backtest_available_at")
            if backtest_available_at > cutoff:
                raise ReasoningContextError("backtest was not available at the reasoning cutoff")
            request = backtest.request
            expected_backtest_fingerprint = backtest_fingerprint({
                "request_fingerprint": request.replay_fingerprint,
                "manifest": {
                    "dataset_id": backtest.manifest.dataset_id,
                    "dataset_version": backtest.manifest.dataset_version,
                    "content_hash": backtest.manifest.content_hash,
                },
                "trades": [trade.model_dump(mode="python") for trade in backtest.trades],
                "events": backtest.events,
                "metrics": backtest.metrics,
                "starting_capital": backtest.starting_capital,
                "ending_equity": backtest.ending_equity,
                "unrealized_pnl": backtest.unrealized_pnl,
            }, domain="backtest-result")
            if expected_backtest_fingerprint != backtest.result_fingerprint:
                raise ReasoningContextError("backtest content does not match its fingerprint")
            if backtest_available_at < request.end:
                raise ReasoningContextError("backtest cannot be known before its historical range ends")
            if (
                backtest.manifest.dataset_id != request.dataset_id
                or backtest.manifest.dataset_version != request.dataset_version
                or backtest.manifest.instrument.instrument_id != request.instrument_id
                or backtest.manifest.timeframe.value != request.timeframe
                or backtest.execution != request.execution
            ):
                raise ReasoningContextError("backtest result manifest/execution does not match its request")
            if request.end > cutoff:
                raise ReasoningContextError("backtest range extends beyond the reasoning cutoff")
            if (request.strategy_id, request.strategy_version, request.setup_id, request.setup_version) != (
                candidate.strategy_id, candidate.strategy_version, candidate.setup_id, candidate.setup_version
            ):
                raise ReasoningContextError("backtest strategy/setup identity does not match candidate")
            if request.instrument_id != candidate.instrument.instrument_id or request.timeframe != candidate.primary_timeframe.value:
                raise ReasoningContextError("backtest instrument/timeframe identity does not match candidate")
            if request.analysis_fingerprint != candidate.analysis_fingerprint:
                raise ReasoningContextError("backtest analysis identity does not match candidate")
            if candidate.dataset_id is not None and (
                request.dataset_id, request.dataset_version
            ) != (candidate.dataset_id, candidate.dataset_version):
                raise ReasoningContextError("backtest dataset identity does not match candidate")
            historical = HistoricalValidationEvidence.from_result(backtest, known_at=backtest_available_at)

        effective_status = candidate.status
        if candidate.expiry_at is not None and candidate.expiry_at <= cutoff and candidate.status is CandidateStatus.ACTIONABLE:
            effective_status = CandidateStatus.WAIT
        facts = list(_candidate_facts(candidate))
        facts.extend(_market_facts(observations, candidate.analysis_fingerprint))
        if historical is not None:
            for kind, label, value in (
                ("backtest_total_trades", "Historical backtest total trades", historical.total_trades),
                ("backtest_closed_trades", "Historical backtest closed trades", historical.closed_trades),
                ("backtest_net_profit", "Historical backtest net profit", historical.net_profit),
                ("backtest_win_rate", "Historical backtest win rate", historical.win_rate),
                ("backtest_max_drawdown", "Historical backtest maximum drawdown", historical.maximum_drawdown),
            ):
                facts.append(EvidenceFact(
                    evidence_id=_fact_id(kind, {"value": value, "fingerprint": historical.result_fingerprint}),
                    tier=EvidenceTier.HISTORICAL_VALIDATION,
                    kind=kind,
                    known_at=historical.known_at,
                    artifact_reference=historical.result_fingerprint,
                    label=label,
                    value=value,
                ))
        facts = tuple(sorted(facts, key=lambda item: (item.known_at or candidate.evaluation_at, item.evidence_id)))
        payload = {
            "schema_version": "3a.1.0",
            "prompt_contract_version": "3a-prompt.1.0",
            "evaluation_time": cutoff,
            "candidate": CandidateReference.from_candidate(candidate),
            "effective_candidate_status": effective_status,
            "market_observations": observations,
            "historical_validation": historical,
            "knowledge": knowledge_refs,
            "deterministic_facts": facts,
            "knowledge_unavailable_reason": (
                "No retrieved knowledge evidence with known publication time at or before the reasoning cutoff was supplied."
                if not knowledge_refs else None
            ),
        }
        # Fingerprint material inputs only; the stored fingerprint is not itself an input.
        context_digest = context_fingerprint_from_fields(payload)
        return ReasoningContext(**payload, fingerprint=context_digest)

    def build_with_retrieval(
        self,
        candidate: TradeCandidate,
        *,
        evaluation_time,
        question: str,
        retrieval_service,
        analysis: AnalysisResult | None = None,
        backtest: BacktestResult | None = None,
        backtest_available_at=None,
        top_k: int = 8,
        filters=None,
    ) -> ReasoningContext:
        """Retrieve through the injected existing service, then validate/build context."""
        results = resolve_retrieval(retrieval_service, question, top_k=top_k, filters=filters)
        return self.build(
            candidate,
            evaluation_time=evaluation_time,
            analysis=analysis,
            knowledge=results,
            backtest=backtest,
            backtest_available_at=backtest_available_at,
        )


def context_fingerprint_from_fields(fields: dict) -> str:
    from app.reasoning.fingerprints import digest
    return digest(fields, domain="reasoning-context")


def resolve_retrieval(retrieval_service, query: str, top_k: int = 8, filters=None):
    """Call the existing semantic/lexical/hybrid retrieval service; no search is reimplemented."""
    if retrieval_service is None:
        return ()
    results = retrieval_service.search(query, top_k=top_k, filters=filters)
    return tuple(results)
