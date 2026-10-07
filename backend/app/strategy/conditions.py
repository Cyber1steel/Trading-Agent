"""Typed condition evaluation with explicit three-valued logic."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from app.strategy.contracts import (
    AllOf, AnyOf, Compare, ComparisonOperator, Condition, ConditionOperand,
    FieldRef, Not, StrategyDefinition, TruthValue,
)


@dataclass(frozen=True)
class ResolvedField:
    value: object | None
    unavailable_reason: str | None = None
    timeframe: object | None = None
    bar_open: object | None = None
    known_at: object | None = None
    candidate_at: object | None = None
    confirmed_at: object | None = None


def resolve_operand(operand: ConditionOperand, definition: StrategyDefinition):
    if operand.parameter is None:
        return operand.value
    for parameter in definition.parameters:
        if parameter.name == operand.parameter.name:
            return parameter.value
    raise ValueError(f"Unknown parameter {operand.parameter.name!r}")


def compare_values(left, right, operator: ComparisonOperator) -> bool:
    if isinstance(left, Enum):
        left = left.value
    if isinstance(right, Enum):
        right = right.value
    if isinstance(left, Decimal) != isinstance(right, Decimal):
        raise TypeError("Decimal comparisons require Decimal operands")
    if isinstance(left, datetime) != isinstance(right, datetime):
        raise TypeError("Datetime comparisons require aware datetime operands")
    if isinstance(left, float) or isinstance(right, float):
        raise TypeError("Float comparison values are forbidden")
    operations = {
        ComparisonOperator.EQ: lambda: left == right,
        ComparisonOperator.NE: lambda: left != right,
        ComparisonOperator.GT: lambda: left > right,
        ComparisonOperator.GTE: lambda: left >= right,
        ComparisonOperator.LT: lambda: left < right,
        ComparisonOperator.LTE: lambda: left <= right,
    }
    return operations[operator]()


def evaluate_condition(condition: Condition, definition: StrategyDefinition, field_resolver):
    """Return outcome plus ordered leaf details (condition, resolved field, compared value)."""
    leaves = []

    def visit(node):
        if isinstance(node, Compare):
            resolved = field_resolver(node)
            compared = resolve_operand(node.operand, definition)
            if resolved.value is None:
                leaves.append((node, resolved, compared, TruthValue.INSUFFICIENT))
                return TruthValue.INSUFFICIENT
            result = compare_values(resolved.value, compared, node.operator)
            truth = TruthValue.TRUE if result else TruthValue.FALSE
            leaves.append((node, resolved, compared, truth))
            return truth
        if isinstance(node, Not):
            value = visit(node.condition)
            return {TruthValue.TRUE: TruthValue.FALSE, TruthValue.FALSE: TruthValue.TRUE,
                    TruthValue.INSUFFICIENT: TruthValue.INSUFFICIENT}[value]
        values = tuple(visit(child) for child in node.conditions)
        if isinstance(node, AllOf):
            if TruthValue.FALSE in values: return TruthValue.FALSE
            if all(value is TruthValue.TRUE for value in values): return TruthValue.TRUE
            return TruthValue.INSUFFICIENT
        if isinstance(node, AnyOf):
            if TruthValue.TRUE in values: return TruthValue.TRUE
            if all(value is TruthValue.FALSE for value in values): return TruthValue.FALSE
            return TruthValue.INSUFFICIENT
        raise TypeError(f"Unsupported condition node: {type(node).__name__}")

    return visit(condition), tuple(leaves)
