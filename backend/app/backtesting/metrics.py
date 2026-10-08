from decimal import Decimal
from typing import Iterable


def compute_metrics(trades: Iterable, *, starting_capital: Decimal = Decimal(0)) -> dict:
    trades = list(trades)
    closed = [trade for trade in trades if trade.net_pnl is not None]
    wins = [trade for trade in closed if trade.net_pnl > 0]
    losses = [trade for trade in closed if trade.net_pnl < 0]
    net_profit = sum((trade.net_pnl for trade in closed), Decimal(0))
    net_wins = sum((trade.net_pnl for trade in wins), Decimal(0))
    net_losses = -sum((trade.net_pnl for trade in losses), Decimal(0))
    average_trade = net_profit / Decimal(len(closed)) if closed else None
    win_rate = Decimal(len(wins)) / Decimal(len(closed)) if closed else None
    peak_equity = starting_capital
    equity = starting_capital
    max_drawdown = Decimal(0)
    for trade in closed:
        equity += trade.net_pnl
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)
    max_drawdown_pct = max_drawdown / peak_equity if peak_equity > 0 else None
    return {
        "total_trades": len(trades),
        "closed_trades": len(closed),
        "open_trades": len(trades) - len(closed),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": win_rate,
        "net_profit_from_winners": net_wins,
        "net_loss_from_losers": net_losses,
        "net_profit": net_profit,
        "average_closed_trade": average_trade,
        "largest_winner": max((trade.net_pnl for trade in wins), default=None),
        "largest_loser": min((trade.net_pnl for trade in losses), default=None),
        "profit_factor": net_wins / net_losses if net_losses else None,
        "maximum_drawdown": max_drawdown,
        "maximum_drawdown_pct": max_drawdown_pct,
    }
