# tail-lib

Market risk measures (volatility, value-at-risk), and the backtesting and
comparison of the models that compute them.

## Docs

- `docs/requirements.md`: what the library must do.
- `docs/architecture.md`: how the code is organized, from a spec to the
  numbers.
- `docs/open-points.md`: pending work and design questions.

## Checks

```sh
uv run ty check
uv run ruff check
uv run pytest
```

## Dependency compatibility tests

```sh
uv tool run tox
```
