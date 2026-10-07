"""Deterministic, in-memory strategy and setup definition foundation."""

from app.strategy.contracts import (
    AllOf, AnyOf, Compare, ComparisonOperator, ConditionOperand, EvaluationRequest,
    EvaluationResult, EvaluationStatus, ExpiryRule, FieldRef, LifecycleState,
    LifecycleTransition, Not, ParameterRef, ParameterType, RequiredMarketContext,
    SetupDefinition, StrategyDefinition, StrategyParameter, TruthValue,
)
from app.strategy.evaluator import evaluate_setup

__all__ = [
    "AllOf", "AnyOf", "Compare", "ComparisonOperator", "ConditionOperand",
    "EvaluationRequest", "EvaluationResult", "EvaluationStatus", "ExpiryRule",
    "FieldRef", "LifecycleState", "LifecycleTransition", "Not", "ParameterRef",
    "ParameterType", "RequiredMarketContext", "SetupDefinition", "StrategyDefinition",
    "StrategyParameter", "TruthValue", "evaluate_setup",
]
