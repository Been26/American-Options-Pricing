from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from option_pricing.models import ExerciseStyle, MarketInputs, OptionContract, OptionType


@dataclass(frozen=True)
class LSMCResult:
    price: float
    standard_error: float
    paths: int
    steps: int


def _payoff(contract: OptionContract, stock: np.ndarray) -> np.ndarray:
    if contract.option_type == OptionType.CALL:
        return np.maximum(stock - contract.strike, 0.0)
    return np.maximum(contract.strike - stock, 0.0)


def _dividend_by_step(market: MarketInputs, maturity: float, steps: int) -> dict[int, float]:
    dividends: dict[int, float] = {}
    for time, amount in market.discrete_dividends:
        if time < maturity:
            step = max(1, min(steps, round(time / maturity * steps)))
            dividends[step] = dividends.get(step, 0.0) + amount
    return dividends


def _simulate_paths(
    contract: OptionContract,
    market: MarketInputs,
    paths: int,
    steps: int,
    seed: int | None,
    antithetic: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = contract.maturity / steps
    drift = (market.rate - market.dividend_yield - 0.5 * market.volatility**2) * dt
    diffusion = market.volatility * sqrt(dt)
    dividends = _dividend_by_step(market, contract.maturity, steps)

    if antithetic:
        half_paths = (paths + 1) // 2
        normals = rng.standard_normal((half_paths, steps))
        normals = np.vstack([normals, -normals])[:paths]
    else:
        normals = rng.standard_normal((paths, steps))

    stocks = np.empty((paths, steps + 1), dtype=float)
    stocks[:, 0] = contract.spot

    for step in range(1, steps + 1):
        stocks[:, step] = stocks[:, step - 1] * np.exp(drift + diffusion * normals[:, step - 1])
        if step in dividends:
            stocks[:, step] = np.maximum(stocks[:, step] - dividends[step], 0.0)

    return stocks


def lsmc_price(
    contract: OptionContract,
    market: MarketInputs,
    paths: int = 50_000,
    steps: int = 100,
    seed: int | None = None,
    polynomial_degree: int = 2,
    antithetic: bool = True,
) -> LSMCResult:
    """Price an option with the Longstaff-Schwartz Monte Carlo method."""
    if contract.exercise_style != ExerciseStyle.AMERICAN:
        raise ValueError("LSMC is intended for American exercise options")
    if paths <= 0:
        raise ValueError("paths must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if polynomial_degree < 1:
        raise ValueError("polynomial_degree must be at least 1")

    stocks = _simulate_paths(contract, market, paths, steps, seed, antithetic)
    dt = contract.maturity / steps
    discount = exp(-market.rate * dt)

    cashflows = _payoff(contract, stocks[:, -1])
    exercise_time = np.full(paths, steps, dtype=int)

    for step in range(steps - 1, 0, -1):
        intrinsic = _payoff(contract, stocks[:, step])
        in_the_money = intrinsic > 0.0

        if np.count_nonzero(in_the_money) > polynomial_degree:
            x = stocks[in_the_money, step]
            y = cashflows[in_the_money] * discount ** (exercise_time[in_the_money] - step)
            coefficients = np.polyfit(x, y, polynomial_degree)
            continuation = np.polyval(coefficients, x)
            exercise = intrinsic[in_the_money] > continuation

            selected = np.flatnonzero(in_the_money)[exercise]
            cashflows[selected] = intrinsic[selected]
            exercise_time[selected] = step

    present_values = cashflows * discount**exercise_time
    price = float(np.mean(present_values))
    standard_error = float(np.std(present_values, ddof=1) / sqrt(paths)) if paths > 1 else 0.0

    return LSMCResult(price=price, standard_error=standard_error, paths=paths, steps=steps)
