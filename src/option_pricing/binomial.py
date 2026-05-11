from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

from option_pricing.models import ExerciseStyle, MarketInputs, OptionContract, OptionType


@dataclass(frozen=True)
class BinomialResult:
    price: float
    steps: int


def _present_value_dividends_after(market: MarketInputs, time: float, maturity: float) -> float:
    return sum(
        amount * exp(-market.rate * (dividend_time - time))
        for dividend_time, amount in market.discrete_dividends
        if time < dividend_time < maturity
    )


def _payoff(contract: OptionContract, stock: float) -> float:
    if contract.option_type == OptionType.CALL:
        return max(stock - contract.strike, 0.0)
    return max(contract.strike - stock, 0.0)


def _stock_at_node(
    adjusted_spot: float,
    up: float,
    down: float,
    market: MarketInputs,
    maturity: float,
    step: int,
    up_moves: int,
    dt: float,
) -> float:
    time = step * dt
    risky_stock = adjusted_spot * (up**up_moves) * (down ** (step - up_moves))
    future_dividends = _present_value_dividends_after(market, time, maturity)
    return max(risky_stock + future_dividends, 0.0)


def binomial_price(contract: OptionContract, market: MarketInputs, steps: int = 500) -> BinomialResult:
    """Price a vanilla option with a Cox-Ross-Rubinstein tree."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    dt = contract.maturity / steps
    discount = exp(-market.rate * dt)
    up = exp(market.volatility * sqrt(dt))
    down = 1.0 / up
    probability = (exp((market.rate - market.dividend_yield) * dt) - down) / (up - down)

    if not 0.0 <= probability <= 1.0:
        raise ValueError("risk-neutral probability is outside [0, 1]; increase steps or check inputs")

    prepaid_dividends = _present_value_dividends_after(market, 0.0, contract.maturity)
    adjusted_spot = contract.spot - prepaid_dividends
    if adjusted_spot <= 0:
        raise ValueError("present value of discrete dividends must be lower than spot")

    values = [
        _payoff(
            contract,
            _stock_at_node(adjusted_spot, up, down, market, contract.maturity, steps, j, dt),
        )
        for j in range(steps + 1)
    ]

    for step in range(steps - 1, -1, -1):
        next_values = []
        for j in range(step + 1):
            continuation = discount * (probability * values[j + 1] + (1.0 - probability) * values[j])
            if contract.exercise_style == ExerciseStyle.AMERICAN:
                stock = _stock_at_node(
                    adjusted_spot,
                    up,
                    down,
                    market,
                    contract.maturity,
                    step,
                    j,
                    dt,
                )
                next_values.append(max(continuation, _payoff(contract, stock)))
            else:
                next_values.append(continuation)
        values = next_values

    return BinomialResult(price=values[0], steps=steps)
