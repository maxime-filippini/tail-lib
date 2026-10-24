# User guide

The guide walks from the ideas behind the library to running and backtesting
models.

1. [Concepts](concepts.md): how returns are passed in, which date a value
   belongs to, how a value-at-risk is signed, and how models fit together.
2. [Runs in Python](python-runs.md): build models as Python objects, run
   them, and read the results. The way in from notebooks and scripts.
3. [Runs from specs](specs.md): the same runs written as data (JSON or
   Python dicts), checked field by field. The way in for a service.
4. [Models](models.md): the catalogue of value-at-risk models and of the
   models they read (volatilities, means, decay factors), with their
   formulas.
5. [Backtests](backtests.md): overshootings, Kupiec's test, and rolling
   backtests.
6. [Simulation](simulation.md): simulated returns with known volatilities,
   to test models on.

Requirement IDs such as R13 or D1 refer to the
[requirements](../requirements.md), for readers who want the reason behind a
rule.
