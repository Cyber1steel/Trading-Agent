"""Deterministic chronological replay over supplied Phase 2E analysis."""

from decimal import Context, Decimal, localcontext

from app.backtesting.contracts import BacktestRequest, BacktestResult, TradeRecord, TradeSide
from app.backtesting.execution import OrderSide, simulate_fill
from app.backtesting.fingerprints import fingerprint
from app.backtesting.ledger import EventLedger
from app.backtesting.metrics import compute_metrics
from app.market_analysis.contracts import AnalysisResult, DatasetProvenance
from app.market_data.timeframes import Timeframe
from app.strategy.contracts import EvaluationRequest, LifecycleState, StrategyDefinition
from app.strategy.evaluator import evaluate_setup
from app.strategy.fingerprints import analysis_full_fingerprint, analysis_prefix_fingerprint

_BACKTEST_CONTEXT = Context(prec=34)


class BacktestEngine:
    """Replay one explicit setup. All run state is local to this call."""

    def run(self, request: BacktestRequest, manifest, analysis: AnalysisResult,
            strategy: StrategyDefinition, setup) -> BacktestResult:
        with localcontext(_BACKTEST_CONTEXT):
            return self._run(request, manifest, analysis, strategy, setup)

    def _run(self, request, manifest, analysis, strategy, setup):
        self._validate_inputs(request, manifest, analysis, strategy, setup)
        ledger = EventLedger()
        ordered = tuple(analysis.observations)
        observations = tuple(
            item for item in ordered
            if request.start <= item.bar_open < request.end and item.known_at <= request.end
        )
        if not observations:
            raise ValueError("backtest range contains no closed primary observations")
        ledger.append("BACKTEST_STARTED", known_at=request.start, reason="backtest started")

        evaluation_request = EvaluationRequest(
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.strategy_version,
            strategy_fingerprint=strategy.fingerprint,
            setup_id=setup.setup_id,
            setup_version=setup.setup_version,
            setup_fingerprint=setup.fingerprint,
            analysis_fingerprint=analysis.fingerprint,
            evaluation_prefix_fingerprint=analysis_prefix_fingerprint(analysis, request.end),
            evaluation_at=request.end,
        )
        evaluation = evaluate_setup(analysis, strategy, setup, evaluation_request)
        transitions_by_time = {}
        for transition in evaluation.lifecycle_transitions:
            if request.start <= transition.known_at < request.end:
                transitions_by_time.setdefault(transition.known_at, []).append(transition)
        terminal_transitions = tuple(
            item for item in evaluation.lifecycle_transitions
            if item.state in (LifecycleState.INVALIDATED, LifecycleState.EXPIRED)
        )

        trades = []
        position = None
        for index, observation in enumerate(observations):
            for transition in transitions_by_time.get(observation.known_at, ()):
                ledger.append(
                    "SETUP_STATE_TRANSITION", transition.known_at, transition.reason,
                    {"state": transition.state.value,
                     "evidence": [item.model_dump(mode="json") for item in transition.evidence]},
                )
                if transition.state is LifecycleState.CONFIRMED and position is None:
                    next_observation = next((item for item in observations[index + 1:]
                                             if item.bar_open >= transition.known_at), None)
                    if next_observation is None:
                        ledger.append("ENTRY_REQUESTED", transition.known_at,
                                      "no later bar to execute entry")
                        continue
                    cancelled = any(
                        terminal.known_at <= next_observation.bar_open
                        and terminal.known_at >= transition.known_at
                        for terminal in terminal_transitions
                    )
                    if cancelled:
                        ledger.append("ENTRY_CANCELLED", transition.known_at,
                                      "setup became terminal before the next executable bar-open")
                        continue
                    entry_side = OrderSide.BUY if request.side is TradeSide.LONG else OrderSide.SELL
                    fill = simulate_fill(next_observation.open, request.execution.fixed_quantity,
                                         request.execution, entry_side)
                    conversion = request.pnl_to_account_rate
                    position = TradeRecord(
                        trade_id=f"T{len(trades) + 1}", strategy_id=strategy.strategy_id,
                        setup_id=setup.setup_id, side=request.side,
                        entry_reference_price=fill.requested_price,
                        entry_bar_open=next_observation.bar_open, entry_price=fill.executed_price,
                        entry_quantity=request.execution.fixed_quantity,
                        fees=fill.fees * conversion,
                        slippage=fill.slippage_cost * conversion,
                        spread_cost=fill.spread_cost * conversion,
                        request_fingerprint=request.replay_fingerprint,
                    )
                    trades.append(position)
                    ledger.append("ENTRY_FILLED", next_observation.bar_open, "entry filled",
                                  {"trade_id": position.trade_id})
                    continue

                if (transition.state not in (LifecycleState.INVALIDATED, LifecycleState.EXPIRED)
                        or position is None or transition.known_at <= position.entry_bar_open):
                    continue
                next_observation = next((item for item in observations[index + 1:]
                                         if item.bar_open >= transition.known_at), None)
                if next_observation is None:
                    ledger.append("EXIT_REQUESTED", transition.known_at,
                                  "no later bar to execute exit", {"reason": transition.state.value})
                    continue

                exit_side = OrderSide.SELL if position.side is TradeSide.LONG else OrderSide.BUY
                fill = simulate_fill(next_observation.open, position.entry_quantity, request.execution, exit_side)
                sign = Decimal(1) if position.side is TradeSide.LONG else Decimal(-1)
                conversion = request.pnl_to_account_rate
                gross = ((fill.requested_price - position.entry_reference_price)
                         * Decimal(position.entry_quantity) * sign * conversion)
                exit_fees = fill.fees * conversion
                total_fees = position.fees + exit_fees
                total_slippage = position.slippage + fill.slippage_cost * conversion
                total_spread_cost = position.spread_cost + fill.spread_cost * conversion
                net = gross - total_fees - total_slippage - total_spread_cost
                closed_values = position.model_dump(mode="python")
                closed_values.update({
                    "exit_bar_open": next_observation.bar_open,
                    "exit_reference_price": fill.requested_price,
                    "exit_price": fill.executed_price,
                    "gross_pnl": gross,
                    "fees": total_fees,
                    "slippage": total_slippage,
                    "spread_cost": total_spread_cost,
                    "net_pnl": net,
                    "return_pct": net / (position.entry_reference_price * Decimal(position.entry_quantity) * conversion),
                    "exit_reason": transition.state.value.lower(),
                })
                closed = TradeRecord.model_validate(closed_values)
                trades[-1] = closed
                position = None
                ledger.append("POSITION_CLOSED", next_observation.bar_open,
                              transition.state.value.lower(), {"trade_id": closed.trade_id})

        realized_pnl = sum(
            (trade.net_pnl for trade in trades if trade.net_pnl is not None), Decimal(0)
        )
        unrealized_pnl = Decimal(0)
        warnings = ()
        if position is not None:
            last_observation = observations[-1]
            exit_side = OrderSide.SELL if position.side is TradeSide.LONG else OrderSide.BUY
            mark_fill = simulate_fill(last_observation.close, position.entry_quantity,
                                      request.execution, exit_side)
            sign = Decimal(1) if position.side is TradeSide.LONG else Decimal(-1)
            gross_mark = ((mark_fill.requested_price - position.entry_reference_price)
                          * Decimal(position.entry_quantity) * sign * request.pnl_to_account_rate)
            unrealized_pnl = (
                gross_mark - position.fees - mark_fill.fees * request.pnl_to_account_rate
                - position.slippage - mark_fill.slippage_cost * request.pnl_to_account_rate
                - position.spread_cost - mark_fill.spread_cost * request.pnl_to_account_rate
            )
            warnings = (
                "An open position remains at the cutoff; ending equity includes a conservative mark-to-market at the last closed candle, with assumed exit spread, slippage, and fees.",
            )
        ending_equity = request.initial_capital + realized_pnl + unrealized_pnl
        metrics = compute_metrics(trades, starting_capital=request.initial_capital)
        metrics["unrealized_pnl"] = unrealized_pnl
        events = tuple({
            "seq": event.seq, "type": event.type, "known_at": event.known_at,
            "reason": event.reason, "payload": event.payload,
        } for event in ledger.events)
        result_fingerprint = fingerprint({
            "request_fingerprint": request.replay_fingerprint,
            "manifest": {"dataset_id": manifest.dataset_id,
                         "dataset_version": manifest.dataset_version,
                         "content_hash": manifest.content_hash},
            "trades": [trade.model_dump(mode="python") for trade in trades],
            "events": events,
            "metrics": metrics,
            "starting_capital": request.initial_capital,
            "ending_equity": ending_equity,
            "unrealized_pnl": unrealized_pnl,
        }, domain="backtest-result")
        return BacktestResult(
            request=request, manifest=manifest, engine_version=request.engine_version,
            execution=request.execution, starting_capital=request.initial_capital,
            ending_equity=ending_equity, unrealized_pnl=unrealized_pnl,
            trades=tuple(trades), events=events,
            metrics=metrics, warnings=warnings, result_fingerprint=result_fingerprint,
        )

    @staticmethod
    def _validate_inputs(request, manifest, analysis, strategy, setup):
        if analysis.fingerprint != request.analysis_fingerprint:
            raise ValueError("analysis fingerprint mismatch")
        if analysis.fingerprint != analysis_full_fingerprint(analysis):
            raise ValueError("analysis fingerprint does not match supplied analysis content")
        if analysis_prefix_fingerprint(analysis, request.end) != request.analysis_prefix_fingerprint:
            raise ValueError("analysis prefix fingerprint mismatch")
        if (request.dataset_id, request.dataset_version) != (manifest.dataset_id, manifest.dataset_version):
            raise ValueError("request dataset identity does not match the supplied manifest")
        if request.instrument_id != manifest.instrument.instrument_id:
            raise ValueError("request instrument identity does not match the supplied manifest")
        timeframe = Timeframe.parse(request.timeframe)
        if timeframe != manifest.timeframe:
            raise ValueError("request timeframe does not match the supplied manifest")
        if manifest.instrument != setup.instrument:
            raise ValueError("setup instrument does not match the supplied manifest")
        if setup.primary_timeframe != timeframe:
            raise ValueError("setup timeframe does not match the backtest request")
        if (request.strategy_id, request.strategy_version, request.strategy_fingerprint) != (
            strategy.strategy_id, strategy.strategy_version, strategy.fingerprint
        ):
            raise ValueError("strategy identity does not match the backtest request")
        if (request.setup_id, request.setup_version, request.setup_fingerprint) != (
            setup.setup_id, setup.setup_version, setup.fingerprint
        ):
            raise ValueError("setup identity does not match the backtest request")
        try:
            contained = strategy.setup(setup.setup_id, setup.setup_version)
        except ValueError as exc:
            raise ValueError("setup is not contained in the supplied strategy definition") from exc
        if contained.fingerprint != setup.fingerprint:
            raise ValueError("setup content does not match the supplied strategy definition")
        if analysis.analysis_version != strategy.analysis_version:
            raise ValueError("analysis version does not match the strategy definition")
        primary = next((item.provenance for item in analysis.inputs
                        if item.provenance.timeframe == timeframe), None)
        if primary is None:
            raise ValueError("analysis does not contain the requested primary timeframe")
        if primary != DatasetProvenance.from_manifest(manifest):
            raise ValueError("analysis primary-source provenance does not match the supplied manifest")
        if request.start < manifest.requested_start or request.end > manifest.requested_end:
            raise ValueError("backtest range must be inside the supplied dataset range")
        if request.start < analysis.analysis_start or request.end > analysis.analysis_end:
            raise ValueError("backtest range must be inside the supplied analysis range")
        if request.end > analysis.cutoff_at:
            raise ValueError("backtest end cannot exceed the analysis knowledge cutoff")
        analysis_input = next((item.provenance for item in analysis.inputs
                               if item.provenance.timeframe == timeframe), None)
        if analysis_input.price_basis.value != request.execution.price_basis.value:
            raise ValueError("execution price basis must match the analyzed market-data price basis")
        ordered = tuple(analysis.observations)
        if any(ordered[index].known_at >= ordered[index + 1].known_at
               or ordered[index].bar_open >= ordered[index + 1].bar_open
               for index in range(len(ordered) - 1)):
            raise ValueError("analysis observations must be strictly chronological")
        if any(item.known_at != item.bar_open + timeframe.nominal_duration for item in ordered):
            raise ValueError("analysis observation availability must match the primary timeframe close")
        if any(item.known_at > analysis.cutoff_at for item in ordered):
            raise ValueError("analysis contains observations beyond its declared cutoff")
