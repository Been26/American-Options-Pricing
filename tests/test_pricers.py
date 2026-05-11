import pytest

from option_pricing import (
    ExerciseStyle,
    MarketInputs,
    OptionContract,
    OptionType,
    binomial_price,
    black_scholes_price,
    lsmc_price,
)


def test_black_scholes_known_call_price():
    contract = OptionContract(
        spot=100,
        strike=100,
        maturity=1,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
    )
    market = MarketInputs(rate=0.05, volatility=0.2)

    assert black_scholes_price(contract, market) == pytest.approx(10.4506, abs=1e-4)


def test_binomial_converges_to_black_scholes_for_european_call():
    contract = OptionContract(
        spot=100,
        strike=100,
        maturity=1,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
    )
    market = MarketInputs(rate=0.05, volatility=0.2, dividend_yield=0.01)

    bs = black_scholes_price(contract, market)
    tree = binomial_price(contract, market, steps=750)

    assert tree.price == pytest.approx(bs, abs=0.02)


def test_american_put_is_at_least_european_put():
    european = OptionContract(
        spot=100,
        strike=100,
        maturity=1,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
    )
    american = OptionContract(
        spot=100,
        strike=100,
        maturity=1,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    market = MarketInputs(rate=0.05, volatility=0.2)

    european_price = binomial_price(european, market, steps=500).price
    american_price = binomial_price(american, market, steps=500).price

    assert american_price >= european_price


def test_lsmc_prices_american_put_near_binomial():
    contract = OptionContract(
        spot=100,
        strike=100,
        maturity=1,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    market = MarketInputs(rate=0.05, volatility=0.2, dividend_yield=0.01)

    tree = binomial_price(contract, market, steps=300).price
    mc = lsmc_price(contract, market, paths=20_000, steps=50, seed=7)

    assert mc.price == pytest.approx(tree, abs=0.45)
