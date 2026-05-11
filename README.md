# Vanilla Equity Options Pricing

Python pricers for vanilla equity options, with support for European and American exercise and dividends.

Implemented methods:

- Black-Scholes-Merton for European options with continuous dividend yield.
- Cox-Ross-Rubinstein binomial tree for European and American options.
- Longstaff-Schwartz Monte Carlo for American options.

The code is intentionally small and readable, so the repository can be used both as a pricing library and as a learning project (essentially).

## Quick Start

```python
from option_pricing import (
    OptionContract,
    MarketInputs,
    OptionType,
    ExerciseStyle,
    black_scholes_price,
    binomial_price,
    lsmc_price,
)

contract = OptionContract(
    spot=100,
    strike=100,
    maturity=1.0,
    option_type=OptionType.PUT,
    exercise_style=ExerciseStyle.AMERICAN,
)

market = MarketInputs(rate=0.05, volatility=0.20, dividend_yield=0.02)

tree = binomial_price(contract, market, steps=500)
mc = lsmc_price(contract, market, paths=50_000, steps=100, seed=42)

print(tree.price)
print(mc.price)
```

European benchmark:

```python
european_call = OptionContract(
    spot=100,
    strike=100,
    maturity=1.0,
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN,
)

price = black_scholes_price(european_call, market)
```

## Financial Setup

The models use the following inputs:

- `S`: current stock price, or spot.
- `K`: option strike.
- `T`: maturity in years.
- `r`: continuously compounded risk-free rate.
- `sigma`: annualized volatility.
- `q`: continuous dividend yield.
- `option_type`: `call` or `put`.
- `exercise_style`: `european` or `american`.

Payoffs are:

```text
Call payoff = max(S_T - K, 0)
Put payoff  = max(K - S_T, 0)
```

A European option can only be exercised at maturity. An American option can be exercised at any time before or at maturity. That early-exercise feature is what makes American options harder to price.

## Black-Scholes-Merton

Black-Scholes-Merton gives a closed-form price for European options under the geometric Brownian motion model:

```text
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

The formula uses:

```text
d1 = [ln(S / K) + (r - q + 0.5 sigma^2) T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

European call:

```text
C = S exp(-qT) N(d1) - K exp(-rT) N(d2)
```

European put:

```text
P = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)
```

where `N(.)` is the standard normal cumulative distribution function.

This implementation supports a continuous dividend yield `q`. It deliberately rejects American options and discrete cash dividends because the plain Black-Scholes-Merton formula does not generally apply to those cases.

Important special case:

- An American call on a non-dividend-paying stock has the same value as the corresponding European call.
- An American put, even without dividends, generally has no simple Black-Scholes closed form because early exercise can be optimal.

## Binomial Tree

The binomial model implemented here is the Cox-Ross-Rubinstein tree.

At each time step `dt = T / steps`, the stock can move up or down:

```text
u = exp(sigma sqrt(dt))
d = 1 / u
```

The risk-neutral probability is:

```text
p = [exp((r - q) dt) - d] / [u - d]
```

At maturity, the option value is its payoff. The tree is then rolled backward:

```text
continuation = exp(-r dt) * [p V_up + (1 - p) V_down]
```

For a European option:

```text
V = continuation
```

For an American option:

```text
V = max(continuation, immediate exercise value)
```

This makes the binomial tree a natural method for American options because the early-exercise decision is checked at every node.

The binomial method is deterministic. Increasing `steps` usually improves accuracy, especially when comparing a European option to the Black-Scholes benchmark.

## Longstaff-Schwartz Monte Carlo

Longstaff-Schwartz Monte Carlo, or LSMC, prices American options by simulation and regression.

First, the stock paths are simulated under the risk-neutral dynamics:

```text
S_{t+dt} = S_t exp((r - q - 0.5 sigma^2) dt + sigma sqrt(dt) Z)
```

where `Z` is a standard normal random variable.

Then the method works backward from maturity:

1. Compute the payoff on every simulated path.
2. At each earlier exercise date, keep only paths that are in the money.
3. Regress the discounted future cashflows on functions of the current stock price.
4. Use the regression to estimate the continuation value.
5. Exercise when immediate payoff is greater than estimated continuation value.

By default, the regression basis is polynomial with degree 2:

```text
continuation(S) ~= a + bS + cS^2
```

The final price is the average discounted cashflow across all paths. The implementation also returns a Monte Carlo standard error.

LSMC is useful because it scales better than trees for more complex products, but it introduces simulation noise and regression approximation error.

## Dividends

Two dividend representations are supported.

Continuous dividend yield:

```python
market = MarketInputs(rate=0.05, volatility=0.20, dividend_yield=0.02)
```

Discrete cash dividends:

```python
market = MarketInputs(
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.00,
    discrete_dividends=[(0.25, 1.0), (0.50, 1.0)],
)
```

Each tuple is `(time, amount)`, where `time` is measured in years from today.

Black-Scholes-Merton only uses `dividend_yield`. It rejects discrete cash dividends.

The binomial method uses an escrowed-dividend approximation: the present value of future cash dividends is separated from the risky stock component so the tree remains recombining.

The LSMC method subtracts the cash dividend from each simulated stock path at the corresponding dividend date.

## Method Comparison

| Method | Exercise style | Dividends | Strengths | Limits |
| --- | --- | --- | --- | --- |
| Black-Scholes-Merton | European | Continuous yield | Fast, closed form, benchmark | No general American pricing, no discrete cash dividends |
| Binomial CRR | European, American | Continuous yield, discrete cash dividends | Deterministic, handles early exercise | Can be slow for very large trees |
| LSMC | American | Continuous yield, discrete cash dividends | Flexible, simulation-based | Monte Carlo noise, regression approximation |

## Notebook

An exploratory notebook is available at:

```text
notebooks/pricer_tests_and_convergence.ipynb
```

It includes:

- Black-Scholes smoke tests.
- Put-call parity checks.
- Binomial convergence plots.
- American early-exercise premium plots.
- LSMC convergence plots with standard errors.
- Discrete dividend examples.

## Notes

.....
