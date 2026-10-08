"""Risk validation findings layered over deterministic calculations."""

from decimal import Decimal

from app.risk.contracts import RiskCalculation, RiskConfiguration, RiskFinding, RiskSeverity, TradeRiskRequest


def validate_risk_calculation(request: TradeRiskRequest, config: RiskConfiguration, calculation: RiskCalculation) -> tuple[RiskFinding, ...]:
    findings: list[RiskFinding] = []

    if calculation.maximum_monetary_risk is not None and calculation.total_estimated_downside is not None:
        if calculation.total_estimated_downside > calculation.maximum_monetary_risk:
            findings.append(
                RiskFinding(
                    rule="downside_cap",
                    observed_value=str(calculation.total_estimated_downside),
                    expected_constraint=f"<= {calculation.maximum_monetary_risk}",
                    severity=RiskSeverity.REJECT,
                    message="Total estimated downside exceeds the configured permitted risk.",
                )
            )

    if config.minimum_reward_risk is not None and calculation.reward_risk is not None:
        if calculation.reward_risk < config.minimum_reward_risk:
            findings.append(
                RiskFinding(
                    rule="minimum_reward_risk",
                    observed_value=str(calculation.reward_risk),
                    expected_constraint=f">= {config.minimum_reward_risk}",
                    severity=RiskSeverity.REJECT,
                    message="Reward/risk is below the configured minimum.",
                )
            )

    if config.max_position_size is not None and calculation.position_size is not None:
        if calculation.position_size > config.max_position_size:
            findings.append(
                RiskFinding(
                    rule="max_position_size",
                    observed_value=calculation.position_size,
                    expected_constraint=f"<= {config.max_position_size}",
                    severity=RiskSeverity.REJECT,
                    message="Position size exceeds configured position limits.",
                )
            )

    if config.max_notional_exposure is not None and calculation.notional_exposure is not None:
        if calculation.notional_exposure > config.max_notional_exposure:
            findings.append(
                RiskFinding(
                    rule="max_notional_exposure",
                    observed_value=str(calculation.notional_exposure),
                    expected_constraint=f"<= {config.max_notional_exposure}",
                    severity=RiskSeverity.REJECT,
                    message="Notional exposure exceeds the configured maximum.",
                )
            )

    if request.side is not None and request.entry_price is not None and request.stop_price is not None:
        if request.side.value == "long" and request.stop_price >= request.entry_price:
            findings.append(
                RiskFinding(
                    rule="long_stop_order",
                    observed_value=str(request.stop_price),
                    expected_constraint=f"< {request.entry_price}",
                    severity=RiskSeverity.REJECT,
                    message="LONG stop must be below entry.",
                )
            )
        if request.side.value == "short" and request.stop_price <= request.entry_price:
            findings.append(
                RiskFinding(
                    rule="short_stop_order",
                    observed_value=str(request.stop_price),
                    expected_constraint=f"> {request.entry_price}",
                    severity=RiskSeverity.REJECT,
                    message="SHORT stop must be above entry.",
                )
            )

    minimum_stop_distance = config.minimum_stop_distance
    maximum_stop_distance = config.maximum_stop_distance

    if calculation.stop_distance is not None and minimum_stop_distance is not None:
        if calculation.stop_distance < minimum_stop_distance or (
            maximum_stop_distance is not None and calculation.stop_distance > maximum_stop_distance
        ):
            findings.append(
                RiskFinding(
                    rule="minimum_stop_distance",
                    observed_value=str(calculation.stop_distance),
                    expected_constraint=f"within [{minimum_stop_distance}, {maximum_stop_distance}]",
                    severity=RiskSeverity.REJECT,
                    message="Stop distance falls outside the configured stop-distance band.",
                )
            )

    if calculation.stop_distance is not None and maximum_stop_distance is not None:
        if calculation.stop_distance > maximum_stop_distance:
            findings.append(
                RiskFinding(
                    rule="maximum_stop_distance",
                    observed_value=str(calculation.stop_distance),
                    expected_constraint=f"<= {maximum_stop_distance}",
                    severity=RiskSeverity.REJECT,
                    message="Stop distance exceeds the configured maximum.",
                )
            )

    if request.target_price is not None and calculation.reward_risk is not None and config.minimum_reward_risk is not None:
        if calculation.reward_risk < config.minimum_reward_risk:
            findings.append(
                RiskFinding(
                    rule="target_reward_risk",
                    observed_value=str(calculation.reward_risk),
                    expected_constraint=f">= {config.minimum_reward_risk}",
                    severity=RiskSeverity.REJECT,
                    message="Target reward/risk is below the minimum requirement.",
                )
            )

    return tuple(findings)
