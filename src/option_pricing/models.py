from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class OptionContract:
    spot: float
    strike: float
    maturity: float
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")

    def payoff(self, stock_price: float) -> float:
        if self.option_type == OptionType.CALL:
            return max(stock_price - self.strike, 0.0)
        return max(self.strike - stock_price, 0.0)


@dataclass(frozen=True)
class MarketInputs:
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    discrete_dividends: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.volatility <= 0:
            raise ValueError("volatility must be positive")
        if self.dividend_yield < 0:
            raise ValueError("dividend_yield must be non-negative")
        for time, amount in self.discrete_dividends:
            if time <= 0:
                raise ValueError("dividend times must be positive")
            if amount < 0:
                raise ValueError("dividend amounts must be non-negative")
