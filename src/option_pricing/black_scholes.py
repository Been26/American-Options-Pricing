from __future__ import annotations

from math import erf, exp, log, sqrt

from option_pricing.models import ExerciseStyle, MarketInputs, OptionContract, OptionType


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_price(contract: OptionContract, market: MarketInputs) -> float:
    """Return the Black-Scholes-Merton European option price.

    The formula supports a continuous dividend yield. It does not support
    American early exercise or discrete cash dividends.
    """
    if contract.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError("Black-Scholes-Merton is only implemented for European options")
    if market.discrete_dividends:
        raise ValueError("Black-Scholes-Merton only supports continuous dividend_yield")

    s = contract.spot
    k = contract.strike
    t = contract.maturity
    r = market.rate
    q = market.dividend_yield
    sigma = market.volatility

    d1 = (log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)

    if contract.option_type == OptionType.CALL:
        return s * exp(-q * t) * _norm_cdf(d1) - k * exp(-r * t) * _norm_cdf(d2)
    return k * exp(-r * t) * _norm_cdf(-d2) - s * exp(-q * t) * _norm_cdf(-d1)
