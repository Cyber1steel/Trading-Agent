from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Optional

from app.backtesting.contracts import BacktestRequest, BacktestResult, TradeRecord
from app.backtesting.ledger import EventLedger
from app.backtesting.execution import simulate_fill
from app.backtesting.metrics import compute_metrics
from app.strategy.evaluator import evaluate_setup
from app.strategy.contracts import StrategyDefinition
from app.market_analysis.contracts import AnalysisResult
from app.backtesting.fingerprints import fingerprint


class BacktestEngine:
    def __init__(self):
        self.ledger = EventLedger()

    def run(self, request: BacktestRequest, manifest, analysis: AnalysisResult, strategy: StrategyDefinition, setup) -> BacktestResult:
        self.ledger = EventLedger()
        # validations
        if analysis.fingerprint != request.analysis_fingerprint:
            raise ValueError("analysis fingerprint mismatch")

        from app.strategy.fingerprints import analysis_prefix_fingerprint
        expected_prefix = analysis_prefix_fingerprint(analysis, request.end)
        if expected_prefix != request.analysis_prefix_fingerprint:
            raise ValueError("analysis prefix fingerprint mismatch")

        # load eligible observations
        ordered = tuple(analysis.observations)
        if any(ordered[i].known_at > ordered[i + 1].known_at for i in range(len(ordered) - 1)):
            raise ValueError("analysis observations must be in chronological order")
        observations = [o for o in ordered if request.start <= o.bar_open < request.end]
        if any(observations[i].known_at > observations[i + 1].known_at for i in range(len(observations) - 1)):
            raise ValueError("analysis observations must be in chronological order")
        self.ledger.append("BACKTEST_STARTED", known_at=request.start, reason="backtest started", payload={})

        trades = []
        position = None

        from app.strategy.contracts import EvaluationRequest

        for idx, obs in enumerate(observations):
            # Evaluate setup at this observation known_at
            eval_req = EvaluationRequest(
                strategy_id=strategy.strategy_id, strategy_version=strategy.strategy_version,
                strategy_fingerprint=strategy.fingerprint, setup_id=setup.setup_id, setup_version=setup.setup_version,
                setup_fingerprint=setup.fingerprint, analysis_fingerprint=analysis.fingerprint,
                evaluation_prefix_fingerprint=analysis_prefix_fingerprint(analysis, obs.known_at),
                evaluation_at=obs.known_at,
            )
            result = evaluate_setup(analysis, strategy, setup, eval_req)
            # If confirmed and no open position -> request entry
            if result and result.status.name == "SETUP_CONFIRMED" and position is None:
                self.ledger.append("SETUP_CONFIRMED", known_at=obs.known_at, reason="setup confirmed", payload={"setup_id": setup.setup_id})
                # Request entry filled at next observation open
                if idx + 1 < len(observations):
                    next_obs = observations[idx + 1]
                    requested_price = next_obs.open
                    exec_res = simulate_fill(requested_price, request.execution.fixed_quantity, request.execution)
                    entry_trade = TradeRecord(
                        trade_id=f"T{len(trades)+1}", strategy_id=strategy.strategy_id, setup_id=setup.setup_id,
                        entry_bar_open=next_obs.bar_open, entry_price=exec_res.executed_price, entry_quantity=request.execution.fixed_quantity,
                        fees=exec_res.fees, slippage=exec_res.slippage, request_fingerprint=request.analysis_prefix_fingerprint
                    )
                    trades.append(entry_trade)
                    position = entry_trade
                    self.ledger.append("ENTRY_FILLED", known_at=next_obs.bar_open, reason="entry filled", payload={"trade_id": entry_trade.trade_id})
                else:
                    self.ledger.append("ENTRY_REQUESTED", known_at=obs.known_at, reason="no later bar to execute entry", payload={})

            # Manage open position: check invalidation or expiry
            if position is not None:
                # check if setup invalidated at this observation
                if result and any(t.state.name == "INVALIDATED" for t in result.lifecycle_transitions):
                    # close at next bar open if available
                    if idx + 1 < len(observations):
                        next_obs = observations[idx + 1]
                        exec_res = simulate_fill(next_obs.open, position.entry_quantity, request.execution)
                        # finalize trade
                        gross = (exec_res.executed_price - position.entry_price) * Decimal(position.entry_quantity)
                        net = gross - position.fees - exec_res.fees - position.slippage - exec_res.slippage
                        closed = position.__class__(**{**position.model_dump(), "exit_bar_open": next_obs.bar_open,
                                                       "exit_price": exec_res.executed_price, "gross_pnl": gross,
                                                       "fees": position.fees + exec_res.fees, "slippage": position.slippage + exec_res.slippage,
                                                       "net_pnl": net, "return_pct": (net / (position.entry_price * Decimal(position.entry_quantity)))})
                        trades[-1] = closed
                        position = None
                        self.ledger.append("POSITION_CLOSED", known_at=next_obs.bar_open, reason="invalidated", payload={"trade_id": closed.trade_id})
                    else:
                        self.ledger.append("EXIT_REQUESTED", known_at=obs.known_at, reason="no later bar to execute exit", payload={})

        # finalize results
        ending_equity = request.initial_capital
        for t in trades:
            if t.net_pnl is not None:
                ending_equity += t.net_pnl
        metrics = compute_metrics(trades)
        # convert dataclass events to serializable dicts
        events = [
            {"seq": e.seq, "type": e.type, "known_at": e.known_at, "reason": e.reason, "payload": e.payload}
            for e in self.ledger.events
        ]
        result_fp = fingerprint({"request": request.model_dump(), "trades": [t.model_dump() for t in trades]}, domain="backtest-result")
        return BacktestResult(
            request=request, manifest=manifest, engine_version=request.engine_version,
            execution=request.execution, starting_capital=request.initial_capital, ending_equity=ending_equity,
            trades=tuple(trades), events=tuple(events), metrics=metrics, warnings=(), result_fingerprint=result_fp,
        )
