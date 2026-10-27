# tail-lib

[![Coverage][coverage-badge]][coverage-report]

[coverage-badge]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fmaxime-filippini.github.io%2Ftail-lib%2Fcoverage%2Fcoverage.json&query=%24.totals.percent_covered_display&suffix=%25&label=coverage
[coverage-report]: https://maxime-filippini.github.io/tail-lib/coverage/

Market risk measures (volatility, value-at-risk), and the backtesting and
comparison of the models that compute them.

## Docs

- `docs/requirements.md`: what the library must do.
- `docs/architecture.md`: how the code is organized, from a spec to the
  numbers.
- `docs/open-points.md`: pending work and design questions.

The user documentation (guide and API reference) is a
[Zensical](https://zensical.org) site, configured in `zensical.toml`, with
its pages in `docs/guide/` and `docs/reference/`. CI deploys it to
<https://maxime-filippini.github.io/tail-lib/> on each push to `develop`:

```sh
uv run --group docs zensical serve   # preview on http://localhost:8000
uv run --group docs zensical build   # write the site to site/
```

## Checks

```sh
uv run ty check
uv run ruff check
uv run pytest
```

To measure line and branch coverage locally:

```sh
uv run coverage run -m pytest
uv run coverage report
uv run coverage html  # open htmlcov/index.html
```

CI uploads the coverage report as the `coverage-report` artifact on every
run. Pushes to `develop` also publish it alongside the docs; the badge above
shows that deployment's coverage and links to its HTML report. It becomes
available after the first successful deployment with coverage enabled.

## Dependency compatibility tests

```sh
uv tool run tox
```
