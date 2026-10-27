# Open points

Pending work and design questions, so a new session can pick them up. Read
`CLAUDE.md`, `docs/requirements.md` and `docs/architecture.md` first.

Working rules, from `CLAUDE.md` and the developer:

- One small change at a time, then stop for review.
- Propose design options before any new abstraction or public API.
- Check each proposal against the decisions recorded here and in
  `docs/architecture.md`, e.g. every backtest runs on every model (R4).
- Commit only when the developer says so. Commits have no co-author trailer.

State as of 2026-10-27: branch `feat/context-api`, from `develop`, not
pushed. ty and ruff (excluding the untracked `examples/t.ipynb`) pass, as
do all 667 tests. The notebook still imports the removed `Run` and has an
existing import-order lint error; it is left untouched.

## Next up

1. **Context API**, plan agreed with the developer on 2026-10-26, see
   "Context API" under "Design questions". All steps are committed,
   including `Context.from_spec`, the `var` rename and the final removal
   of `Run`, `RunResult`, `Cache`, `Pool`, `node.run` and `Builder`.
   Follow-up, agreed on 2026-10-27: backtest declarations (`Kupiec(name=...)`
   inside a context, with `Context.add_backtest` underneath) are committed;
   see "Backtest declarations" below. Next is the run date range, from step 2.
2. **Run date range (R21–R23)**, agreed with the developer on 2026-10-25,
   see "Run date range" under "Design questions". Step 1 (docs) done,
   awaiting review. Steps 2 to 5 wait for the context API.
3. **Run report (R19)**, every step built; spec agreed on 2026-10-25, see "Run report" under
   "Design questions". Step 1 (nodes describe their settings) done. Then,
   each reviewed on its own: names as identifiers (R20, done), labels on
   the run (R20, done), and step 2 (`run.graph`, `Run.evaluate`, done).
   Step 3 is split in five parts (see "Steps" below): 3a (`ReportData`)
   done, 3b (`Report`, `save()`), 3c (the frontend scaffold), 3d (the CI
   check), 3e (the graph tab), 4 (the returns vs VaR tab), 5 (the backtest
   table), 6 (the rolling backtests tab) done. Step 7 (node outputs on
   click) done, awaiting review. Left: the "Later" and "Still open" items
   under "Run report".
4. **Serializing results (R12)**, a design to propose first. See "Design
   questions" below.

## Decisions on finished work

Recorded so later changes don't undo them by accident.

### Backtest declarations (R4, R10, R11)

Agreed on 2026-10-27, replacing the functions of overshootings used during
the context migration:

- `Kupiec(name="kupiec", window=...)` is a frozen, model-independent
  declaration with keyword-only fields. Constructing it inside a context
  registers it on the innermost one, as for models (D10). A name is required
  inside a context and is checked even outside one (R20). Outside, a
  declaration may be unnamed and reused. Names are registration metadata,
  excluded from equality.
  `KupiecSpec` remains its JSON form; the executable `_KupiecTest` is
  internal. The result type and the `kupiec_pof` function are unchanged.
- `ctx.add_backtest(name, declaration)` records the declaration; the
  construction hook calls it for direct declarations in a block. The
  constructor's `backtests` mapping uses the same method. It checks names
  are identifiers and rejects duplicate backtest names and collisions
  with named nodes. A failed registration keeps the existing declaration.
- `ctx.graph()` calls each declaration's private `_bind(model)` with node
  registration suspended. Binding happens against every current model,
  so registration order does not matter and declarations can be reused in
  different contexts (R4, D10). No nodes are computed during registration
  or graph construction.
- Kupiec binding builds `_KupiecTest(Overshootings(model), window=...)`.
  The internal node keeps explicit dependencies, pure `compute` and its
  signature check (D11), and reads that model's confidence level. Equal
  nodes, including overshootings and identical tests under different
  names, compute once (D9).
- Specs become the same declarations (`build._build_backtest`) and are
  registered through the context constructor. Unnamed spec declarations
  are created with automatic registration suspended, so an outer Python
  context is untouched on success and failure. The JSON format is unchanged.
- Reports display the public name `Kupiec` for the internal executable
  node. Its settings, paths, graph edges and numerical outputs are unchanged.
- This revisits the earlier rejection of a settings class per backtest:
  the public interface still has two classes for declaring Kupiec tests,
  `Kupiec` and `KupiecSpec`. The concrete executable node is internal.
- Thirteen new cases cover registration before, between and after models,
  collisions with each kind of named node, duplicate registration, missing
  and invalid names, shared outputs under different names, and reuse across
  contexts with different confidence levels and shapes. Existing nesting,
  thread and asyncio tests now check backtest registration too; spec tests
  check that construction does not affect a surrounding context.
  Existing numerical tests use the public declaration; internal graph and
  dependency tests still inspect the concrete node.
- Validation: all 667 tests pass, with regression data unchanged. The
  example prints the same table, and all 20 Python blocks on the home page,
  spec guide, Python-run guide and backtest guide run. The report sample
  regenerates and `npm run check` passes without warnings. ty and ruff pass
  excluding the untouched notebook; full checks still report its removed
  `Run` import and import order. The strict docs build still reports the
  existing README link in the historical review.

### Rolling backtests (R15, R16)

- A backtest takes an optional `window`: `trailing` windows, or `last`, one
  window on the last dates (e.g. Kupiec on the last 250 days as of the last
  date). Named backtests let several windows coexist, and all the backtests
  of a model share its `Overshootings` (D3).
- Results have one value per window: every `KupiecResult` array has the
  shape (n_windows, *batch), and `end_dates` gives each window's last date.
  A whole-sample test is one window ending on the last date. Rejected:
  results as time series on the returns' dates (mostly `NaN` for a
  whole-sample test), and a separate rolling result type.
- Windows start on the first date where every series has a value (R16,
  `Windows(..., skip_leading_nan=True)`), so no window holds warm-up `NaN`s.
  Backtests and filtered historical simulation use it; other models don't
  yet. To be replaced by the run date range (R21–R23): with a `step`, a
  late-starting portfolio shifts every portfolio's windows (risk-manager
  review of 2026-10-25, finding 1).
- Rejected for "the last 250 days": reading `[-1]` of a rolling test
  (computes every window), and windows counted back from the last date (past
  windows move when a date is added, so results aren't reproducible, D4).
- Timing: the window of overshootings ending on `t` uses VaR up to `t - 1`
  and returns up to `t` (D1).

### `Last` windows for models

- Historical simulation (filtered or not) and the decay calibration take
  `Trailing | Last`. With `Last`, a model gives one value, on the last date,
  e.g. today's VaR without computing every window (R6, R7).
- Such a model can't be backtested: its only VaR is the forecast for a date
  after the data. `RunSpec` rejects a run with backtests and a model with a
  `last` window anywhere in its spec, naming the model and the path. So
  `last` models go in their own run, without backtests.
- Every window in a spec needs its `type` (`trailing` or `last`). The
  developer chose strict, as for R9a, over a missing `type` defaulting to
  `trailing`.

### FHS spec (R9)

`FilteredHistoricalSimulationsSpec` nests a `VolatilityFilterSpec` (`vol`,
`mean`) in its `filter` field, mirroring the node's `filter` argument.
Chosen by the developer over flat `vol` and `mean` fields on the FHS spec
(as in `GaussianSpec`), from which the builder would make the filter: that
breaks "a spec's fields mirror its node's arguments". Identical filter
specs build equal filter nodes, computed once by value identity (D9), and
paths go through them, e.g. `models.fhs.filter.vol.decay.window`.

### JSON round trip (R9)

`RunSpec.model_dump_json()` parses back to an equal spec, and spells out
every default. Tested on specs holding every spec class.

### Runs from Python code (R17)

Python code (notebooks, scripts) declares nodes inside a `Context`.
Specs are the form that can be deserialized, for the JSON service. They can
be built in Python too, but only as a data format, with its validation,
to be turned into nodes (`Context.from_spec`): they hold no logic.
This replaces an earlier choice of specs as the one way in. Chosen by the
developer:

- Backtests are model-independent declarations, registered by construction
  inside a context, explicitly with `ctx.add_backtest`, or in the context
  constructor's mapping (see
  "Backtest declarations"). The context binds them against each model;
  future tests can use outputs besides overshootings. This replaces the
  original function-of-overshootings interface. Rejected: backtests bound
  to selected models by hand (loses "every backtest on every model", R4).
- Nodes list the nodes they read (`dependencies()`), so `Context` can find run
  inputs and `last` windows up front. It also gives names for R12. Rejected:
  no graph walk, which reports missing inputs mid-run and can't reject a
  backtested `last` model.
- `Context.from_spec(spec)` builds the nodes inside a context.
  `RunSpec` keeps its own `last`-window check, for errors with a spec path
  (R9b).
- Names are unique in a run's graph (developer's rule, against silent
  overriding): nodes carry their names, model keys in a spec become those names, and
  backtests are named by dict keys. Each input name belongs to one node;
  equal nodes reading one input are one node (D9), while different nodes
  using that name are an error. A model and a backtest can't share a name either (2026-10-25), so
  one `labels` mapping covers both (R20). Rejected: separate model and
  backtest label mappings, and one label applying to both.
- Not now: a node-built run can't be written back as a spec, which saving
  what ran (R12) and reusing results (D4) may need later.

## Small open points

- Specs from HTTP (R17) are untrusted. To discuss: limits on what one spec
  can ask for (number of models, window lengths), and whether the service
  needs spec errors in a structured form (pydantic's `errors()` gives each
  path and message).
- The Python example in `docs/architecture.md` ("From Python") isn't tested.
- The code examples of the user guide (`docs/guide/`) were run by hand
  when written, but aren't tested. Options: a pytest that runs the
  `python` blocks, or `pymdownx.snippets` including tested files.
- The docs site is built in CI on every run, and deployed to GitHub Pages
  (https://maxime-filippini.github.io/tail-lib/) on pushes to `develop`.
  Not deployed yet: the docs commits aren't on `develop`.
- The guide installs from the git repository: tail-lib isn't on PyPI.

- `NaN` in the middle of a window (e.g. a missing return). Proposal: keep
  counting the usable flags (`n_obs < length`).
- Maybe a `to_series()` on backtest results, to put window results on the
  returns' dates.
- Maybe a `load_run_spec(path)` helper (new public API, to propose if
  wanted). `RunSpec.model_validate_json(path.read_text())` works today.
- Integer checks are still written in each class: window length (in both
  `Trailing` and `Last`) and step, backcast window, number of dates and
  burn-in. A `check_at_least(name, value, minimum)` in `_checks.py` would
  write them once.
- `GaussianModel` rejects a zero or negative volatility, the scale of its
  distribution (developer's choice). `VolatilityFilter` and the FHS, which
  divide and rescale by a volatility, don't check it yet.
- `Windows.apply` holds all windows in memory at once. Chunk it for long
  series (R6).
- The tox matrix varies NumPy only. Consider pinning the lowest pydantic
  (2.13) in a test environment.
- The derived requirements D1–D7 are not yet confirmed by the developer.
- The branch hasn't been pushed, and there's no PR to `develop` yet.

## Design questions

### Context API

The developer's sketch, `.scratch/api_update_pure.py` (not in git),
proposes a new way to declare and run models. Migration plan agreed on
2026-10-26; D9–D11 in `docs/requirements.md` describe the target.

**Agreed**

- Nodes are frozen dataclasses, identified by value (D9). A node's inputs
  are its fields that hold nodes; its other fields are its settings. So
  `dependencies()` and `settings()` are no longer written by each node.
- No `evaluate`: the run calls `compute(returns, **inputs)`, one keyword
  argument per input, named after its field, holding that input's output.
  Building a node checks the signature (D11).
- Nodes compute on `TimeSeries` (R24), not bare arrays, so the dates stay
  with the values. Every time series output has the dates and shape of the
  run's returns, checked by the run. Backtest results are not time series.
- Decay nodes output a `DecayFactors`: the decay on each date, the refit
  dates and the first date each fit used. One `EwmaVolatility` re-filters
  from that output alone (D7), and replaces `FittedEwmaVolatility`, which
  today reads its calibration's `window` and `start`.
- `VolatilityFilter` stays a node, so the FHS spec keeps its nested
  `filter` (see "FHS spec" above), and FHS models on 500 and 750 dates
  share one filter. The filter's output holds the standardized returns,
  the mean and the volatility, so the FHS takes only `filter`, as
  `compute` only sees its own inputs.
- Node fields take the spec field names, e.g. `vol` and `mean` rather than
  `vol_model` and `mean_model`, so a path through a run reads like the
  path in its spec.
- `with Context(...) as ctx:` replaces `Run`. Nodes built in the block bind
  to it, through a `ContextVar` (D10). VaR models are named, and names are
  Python identifiers (R20). `ctx.graph()` orders the nodes without
  computing; `ctx.run(returns, inputs)` computes each once and returns
  `Results`, read by node or by name. `Cache`, `Pool` and `node.run` go.
- Backtests run on every VaR model (R4). The original constructor mapping
  of functions of overshootings was replaced on 2026-10-27 by immutable
  declarations, now registered by construction in the block or explicitly
  with `Context.add_backtest`, see "Backtest declarations".
  Internal executable nodes still inherit `Backtest`.
- Run inputs (R13): `run(returns, inputs)` sets the output of each input
  node to `inputs[node.name]`, after checking it is aligned with the
  returns, and never calls its `compute`. Missing inputs are still
  reported before computing.
- A context without VaR models computes its other nodes, e.g. a
  volatility on its own. This replaces `Pool` and `node.run`, and reopens
  "Runs without VaR models" below.
- A context from a spec: `Context.from_spec(spec)` opens a context with
  the spec's backtests and labels, and builds each model with plain
  `match` functions, the model's key in `models` as its name. Value
  identity shares identical parts (D9), so the `Builder` tables go; the
  check on an input declared twice differently, and `RunSpec`'s
  `last`-window check with its paths (R9b), stay. For later: the node
  dataclasses as the specs (pydantic over them, value ranges as field
  annotations), which would also write a context back as a spec (R12,
  D4). Rejected: keeping the `Builder` class.
- Done before steps 2 to 5 of the run date range, which will use the
  context's graph walk.

**Not taken from the sketch.** The sketch simplifies some methods. The
migration keeps today's numbers, and each of these would be its own
change, if the developer asks for it: a grid fit of the decay (today:
golden-section search with `bounds` and `n_iter`); `arch`'s weighted
backcast (today: `Backcast` and `StartingVol`); `Rolling` and `Expanding`
windows with `refit_every` (today: `Trailing` with `step`, and `Last`, to
revisit with the run date range); dropping the HS `interpolation`, the
Gaussian volatility check and `Gaussian().quantile`; the new
`SampleMean` and `SampleVolatility` nodes and a default volatility for
`GaussianModel` (R9a defaults are an open question); names with dashes
(R20).

**Steps**, each reviewed on its own, with every check green:

0. Docs: this section, R24, D9–D11.
1. Pin the numbers: `tests/test_regression.py` computes one node of each
   type, and the Kupiec tests of a run, on fixed returns, against the
   arrays in `tests/data/regression.npz` (returns included, so the
   simulator can change). Later steps rewrite how the test builds and runs
   the nodes, keep its keys and arrays, and stay green, or regenerate the
   file on purpose (`uv run python -m tests.test_regression`) and say why.
2. Nodes as frozen dataclasses, identified by value, still on `Cache` and
   `evaluate`. Arrays in fields are stored as tuples, so they hash.
   `dependencies()` and `settings()` written once, from the fields. Then
   2b: the field renames, with spec and report paths.
3. `DecayFactors`, and one `EwmaVolatility` for constant and calibrated
   decays. The `ewma` spec takes a `constant` or `ewma_calibration`
   decay; `fitted_ewma` goes.
4. Pure `compute`, family by family, through a generic `evaluate` in the
   base class until the last one: 4a means, constant volatility, HS,
   Gaussian, overshootings, Kupiec and the `Backtest` base class; 4b EWMA;
   4c the filter and FHS; 4d input nodes. Then the generic `evaluate`
   goes, and the `CLAUDE.md` rules on `evaluate` and the `Cache` are
   reworded with the developer.
5. `Context` and `Results`: 5a next to `Run`; 5b the report takes a
   context and reads `Results`; 5c `Run`, `RunResult`, `Cache` and `Pool`
   removed, the example and the guide moved over.
6. A context from a spec (R9, R9a, R17), `Context.from_spec`.
7. Then the run date range, from its step 2 (trimming the returns, R21, is
   its step 3).

Done in step 2 (2026-10-27), with no output changed (the regression pins
pass unchanged):

- `Node` and every library node are frozen dataclasses. `dependencies()`
  and `settings()` are written once in `Node`, from the fields set by the
  constructor; the per-node overrides went, except the FHS's, which also
  lists its filter's mean and volatility, read by its `evaluate`, until
  step 4c.
- Arrays given to `ConstantMean`, `ConstantVolatility`, `ConstantDecay` and
  `StartingVol` are stored as nested tuples (`frozen_floats` in
  `graph.py`, type `Floats` in `_typing.py`), one level per axis, so batch
  shapes of any rank keep their shape. `settings()` returns the tuples, and
  the report writes them as lists, as before.
- The mean defaults to `ZeroMean()` (a field default) instead of
  `None`. Two zero means in one graph are now one node, so the report's
  graph has one box fewer.
- `KupiecTest` copies `cf_var` from its overshootings into a field it sets
  itself (`init=False`, not compared), so `compute` reads its own settings
  only. It is not a setting.
- Run inputs: equal input nodes are one node, so two `VolatilityInput("vol")`
  built separately are no longer an error. Two different nodes on one name
  (a volatility and a VaR input) still are.
- A subclass of a node that isn't itself a dataclass can still set new
  attributes, e.g. the counters in the tests. The cycle test builds its
  cycle that way: frozen nodes can't form one otherwise. The cycle check in
  `Cache` stays until the context replaces it.
- Not changed: the `Builder` keeps its tables (now redundant, they go in
  step 6), and field names stay `vol_model` and `mean_model` until 2b.

Done in step 2b (2026-10-27), no output changed:

- `vol_model` and `mean_model` are `vol` and `mean`, on `GaussianModel` and
  `VolatilityFilter`, and in the FHS's dependencies; `ConstantDecay`'s
  `decay_factor` is `value`. So run and report paths read like spec paths,
  e.g. `models.gauss.vol.decay`, and the report shows the decay's setting
  as `value`.
- `Overshootings.var_model` kept its name here: it has no spec to follow.
  Renamed `var` after step 4b, see below.
- `compute` arguments keep their names (`means`, `vols`, `decay`): they
  become the field names in step 4, where `compute` takes one keyword per
  input.

Done in step 3 (2026-10-27), no output changed (the fitted EWMA's pins are
now computed by `EwmaVolatility` on a calibration):

- Decay models (`DecayModel`, no longer a `Preprocessor`) output
  `DecayFactors`: `values`, the decay on each date as a `TimeSeries`;
  `refits`, the index of each refit date; `fit_starts`, for each refit, the
  index of the first date its fit used. A constant decay has one refit, on
  date 0, from date 0; a calibration one per window, on its last date, from
  its first. Refits are index arrays, one entry per refit, rather than the
  sketch's per-date mask: shorter, and the dates are `values.dates[refits]`.
  `DecayFactors` checks they are increasing and that each fit starts on or
  before its refit.
- `EwmaVolatility` takes any decay model and runs the recursion once per
  refit, from its fit's first date, up to the date before the next refit
  (D7). The runs go through one vectorized recursion, the runs on a batch
  axis, as `FittedEwmaVolatility` did, so a calibration with `step = 1`
  doesn't loop per refit (R6). `FittedEwmaVolatility` and the `fitted_ewma`
  spec are gone: `{"type": "ewma", "decay": {"type": "ewma_calibration",
  ...}}` replaces it. A decay in the `ewma` spec now needs its `type`.
- Two starts meet when an EWMA reads a calibration: the EWMA's own `start`
  and the one the calibration fitted with. `FittedEwmaVolatility` used the
  calibration's (D7). Chosen here, to confirm: the two have to be equal,
  checked when the EWMA is built and in the `ewma` spec (with its path,
  R9b), so no number changes silently. Alternative: the EWMA's `start`
  seeds every run and the calibration's only its fits (the sketch's way,
  simpler, but a calibration with a non-default start would then be
  re-filtered from another start than it was fitted with).
- Paths through a calibrated EWMA go through `decay`, e.g.
  `models.today.vol.decay.window`. The report labels decay nodes as
  preprocessors and charts their `values`.

Done in step 4a (2026-10-27), no output changed:

- `Node.evaluate` is written once: it gets each dependency's output from
  the cache and calls `compute(returns, **outputs)`, by field name.
  `compute` is abstract on `Node`, typed `(*args: Any, **kwargs: Any)` so
  each node narrows it, as in the sketch. The means, `ConstantVolatility`,
  HS, Gaussian, `Overshootings` and `KupiecTest` dropped their own
  `evaluate`.
- Building a node on the generic `evaluate` checks its `compute` (D11):
  one positional argument, the returns, then exactly one keyword-only
  argument per dependency; no `*args` or `**kwargs`. Nodes still on their
  own `evaluate` are skipped. `tests/test_dependencies.py` lists those
  (`_OWN_EVALUATE`), and checks every `__post_init__` reaches
  `Node.__post_init__`, where the check runs.
- `compute` keywords are the field names: the Gaussian's `compute` takes
  `vol` and `mean` (were `vols` and `means`), `Overshootings`' took
  `var_model`, the field's name (renamed `var` after step 4b, see below).
  Kupiec takes the returns, unused, like every `compute`.
- The input nodes' `compute` raises `TypeError`: they keep reading the
  run's inputs in `evaluate` until 4d.
- `Backtest[T]` (in `graph.py`) is the base class of backtest nodes.

Done in step 4b (2026-10-27), no output changed:

- `ConstantDecay`, `EwmaDecayCalibration` and `EwmaVolatility` dropped their
  own `evaluate`: the generic one passes the EWMA its `decay`'s
  `DecayFactors`, and the decay models only the returns. Building them now
  checks their `compute` (D11). Their `compute` takes the returns
  positional-only (`returns, /`), like the nodes of 4a.
- Nothing else had to change: step 3 had already made the EWMA read only its
  decay's output, not its calibration's settings.

Done after step 4b (2026-10-27), developer's choice, no output changed:

- `Overshootings.var_model` is `var`, so its `compute` reads the VaR series
  as `var`, not as `var_model`. `var` names both the model the field holds
  and the series `compute` gets. Rejected: `model`, as unclear as
  `var_model` inside `compute`.
- Run and report paths through overshootings end in `var`, e.g.
  `backtests.gauss.kupiec.overshootings.var.vol`. The report page finds a
  model's overshootings by that edge name (`report-app/src/models.ts`), so
  the built page is rebuilt.

Done in step 4c (2026-10-27), no output changed:

- `VolatilityFilter` outputs `FilteredReturns` (in `volatility.py`):
  `standardized`, the standardized returns, with the `mean` and `vol` that
  standardized them, as the plan agreed. It checks the three are aligned.
  The filter is a `Node[FilteredReturns]`, no longer a `Preprocessor`,
  as decay models stopped being one in step 3. Its `compute` takes `vol`
  and `mean` (were `vols` and `means`).
- The FHS's `compute(returns, /, *, filter)` reads the standardized returns,
  mean and volatility from its filter's output. Its `dependencies()`
  override and its `evaluate` went, so no node overrides `dependencies()`
  any more, and `tests/test_dependencies.py` now checks that settings and
  dependencies are exactly the constructor arguments. The keyword `filter`
  shadows the builtin: one `noqa: A002`, as the name follows the field and
  its spec.
- Run and report paths lose `models.fhs.vol` and `models.fhs.mean`: the
  FHS reaches them only through `models.fhs.filter`. The report's graph has
  two edges fewer per FHS model. The report charts a filter's
  `standardized` returns, as it charts a decay model's `values`, and labels
  it a preprocessor. The page itself is unchanged.
- The regression test pins a filter's `standardized` returns, under the
  same keys.

Done in step 4d (2026-10-27), no output changed:

- `RunInput` (in `graph.py`), chosen by the developer over the run seeding
  the cache with each input node's output (its graph walk is in `run.py`,
  which `node.run` and `Pool.run` don't use), and over waiting for the
  context. It is a frozen dataclass with the `name` field, which it checks
  is an identifier, and a `compute` that raises `TypeError`.
  `VolatilityInput` and `ValueAtRiskInput` inherit it, next to their
  model base class; their constructors are unchanged.
- `Cache.get` gives a `RunInput` node `cache.input(node.name)`, aligned
  with the returns, and never calls its `compute`. The input nodes'
  `evaluate` went, so no library node has its own `evaluate`, and building
  any node checks its `compute` (D11).
- `run._input_name` is an `isinstance` check on `RunInput` instead of a
  `match` on the two classes, so the report's `input_name` follows.

Done at the end of step 4 (2026-10-27), no output changed:

- `Node.evaluate` is gone: `Cache.get` gets the output of each of a node's
  `dependencies()`, then calls `node.compute(returns, **outputs)`. So
  building any node checks its `compute` (D11), with no exception for
  nodes overriding `evaluate`. `Run.evaluate` and `Pool.evaluate` stay
  until step 5.
- The two test nodes that overrode `evaluate` (in `tests/test_graph.py`)
  are rewritten: one as a frozen dataclass with `filter` and `vol` fields,
  the cycle one with a `compute` taking its `dependency`.
- The `CLAUDE.md` rules "Nodes for estimable parameters" and "Pure
  `compute`" are reworded, chosen by the developer: an input node is a
  field holding another node, a setting a field holding a plain value, and
  `compute` gets the inputs' outputs from the run. They name neither
  `evaluate` nor the `Cache`, so they hold with the context too.

Decided for step 5 by the developer (2026-10-27):

- Names: every node takes an optional `name=` keyword, not part of its
  identity (as in the sketch). A context registers its VaR models by it,
  and `Results` reads any named node by it. Names are unique in a context.
  Rejected: names given to the context only (`ctx.add("gauss", model)`).
- Run inputs: one name. A run input's name is its node name and the input
  it reads, and stays part of its identity, so `ValueAtRiskInput("var_ext",
  cf=0.99)` is the model `var_ext`. The input spec is unchanged; in step 6
  a spec input model's key has to match its `name`. Rejected: a separate
  field for the input it reads, which changes the spec's JSON.
- A name is required in a context only: building a VaR model without one
  inside a `with Context()` block raises `ValueError` there. Outside a
  context (tests, `Run` until 5c) no name is needed. Rejected: always
  required.
- `Results` is read by node (`results[node]`), by name
  (`results["gauss"]`), and through tables: `results.models` (name to VaR
  series) and `results.backtests[model][test]`, as `RunResult` has, so the
  report moves over with few changes in 5b. Rejected: by node or name only.

Done in step 5a-1 (2026-10-27), no output changed:

- `Node.name`: optional, keyword-only, a Python identifier, checked in
  `Node.__post_init__`. `compare=False`, so not part of the identity, and
  `repr=False`. `settings()` and `dependencies()` skip fields with
  `compare=False`, so the name is neither.
- `RunInput` redefines `name` as a required positional field, part of the
  identity. Two dataclass details made this need care, both commented in
  `graph.py`: the field is written `name: str = field()`, since a bare
  annotation inherits `Node`'s `None` default from the class attribute;
  and `VolatilityInput` and `ValueAtRiskInput` list `RunInput` first among
  their bases (was second), since a dataclass collects its bases' fields in
  reverse MRO order and the model base class would bring back `Node.name`.
  Their constructors are unchanged. A bad name is now reported before a bad
  `cf` on `ValueAtRiskInput`.
- Nothing reads a node's name yet: `Run`, the report and the specs ignore
  it until the context.

Done in step 5a-2 (2026-10-27), no output changed:

- `tail_lib.context`: `Context` and `Results`, next to `Run`, as
  described in `docs/architecture.md` ("Contexts"). `Node.__post_init__`
  binds the node to the open context, through `graph.current_context`.
  The run checks every output but a backtest's against the returns (R24),
  and doesn't use the `Cache`.
- The context reuses `run.py`'s private helpers (`_graph`, `_run_graph`,
  `_check_names_and_labels`, `_check_no_last_window`, `_input_names`) and
  `RunGraph`. They move into `context.py` when `Run` goes (5c).
- Chosen here, to confirm:
  - Bound nodes no model or backtest reaches are graph roots too, with
    paths `nodes.<name>`, or `nodes.<class>` and `nodes.<class>_2`, ...
    without a name, latest built first, so a node's inputs are reached
    through it. Unnamed paths depend on the order nodes were built.
  - `results["name"]` covers the nodes built in the block with a name:
    a named node built outside the context isn't known by its name, as
    value identity can't tell two equal instances' names apart.
  - A node other than a backtest whose output holds no time series is a
    `TypeError` at run time (R24). Known outputs: `TimeSeries`,
    `DecayFactors` (its `values`), `FilteredReturns` (its `standardized`).
  - Names are unique across every node and the backtests, not only the
    models and backtests.
  - The run has no cycle check: frozen nodes can't form a cycle (the
    `Cache` kept its check, for the cycle test's mutable node).
  - `Context` has no `input_names`, unlike `Run`: nothing needs it yet.
- `tests/test_context.py` checks the binding (nesting, threads, asyncio
  tasks), the names, the graph, and that a context gives `Run`'s numbers.

Done in step 5b (2026-10-27), no output changed:

- `Report(context, results, title=..., series_labels=...)` and
  `build_report_data(context, results, ...)`, chosen by the developer over
  `Report(ctx, returns, results)` and over the report running the context
  itself. The report computes nothing, so computing then reporting computes
  once (risk-manager review, finding 2). It raises `ValueError` if the
  results lack a node of the context's graph.
- `Results.returns`: the returns the run computed on, so the report needs
  nothing else.
- `Run` no longer has a report. The report tests, the sample data script
  (`report-app/scripts/export_sample.py`) and the guide's "Reports"
  section use a context; the rest of the guide moves in 5c.
- With one name per run input, the sample's and the tests' input model is
  named `var_ext`, after its input (was `external` or `ext`).

Order of the remaining steps, chosen by the developer (2026-10-27): step 6
(`Context.from_spec`) comes before `Run` is removed, so specs run at every
commit (rejected: a `Context.from_spec` over today's `Builder`, then
rewritten; or a gap without specs). Step 5c is split in three: 5c-1 the
regression test, the example and the guide; 5c-2 the tests off `node.run`,
`Pool` and `Cache`; 5c-3 the removal, after step 6.

Done in step 5c-1 (2026-10-27), no output changed:

- `tests/test_regression.py` builds every node inside two contexts (one
  with the Kupiec backtests, one for the `Last` models) and reads them
  from `Results`. The pinned keys and arrays are unchanged: the VaR input
  is now named `var_input`, after the model key, since a run input's name
  is its model's.
- `examples/backtest_hs_fhs.py` declares its models in a context; it prints
  the same numbers. Its step 3 prints the number of distinct nodes, as
  `Context` has no `input_names`.
- The guide: "Runs in Python" is rewritten around contexts (names, results
  by node or name, the error table, contexts without models, the graph);
  the home page, "Backtests" and "Simulation" use a context. Each page's
  Python blocks were run against simulated returns. "Runs from specs" and
  the home page's spec block still use `Run.from_spec`, until step 6.
- `docs/architecture.md`'s "From Python" shows a context.

Done in step 5c-2 (2026-10-27), no output changed:

- `tests.helpers.run_node` computes one node and its dependencies in a
  context. An unnamed VaR model gets a named copy, equal by value (D9),
  so the helper can read the output by the original node. This is a test
  helper, not a new library API.
- Node tests for decay, volatility, VaR, overshootings, Kupiec and run
  inputs, and the simulation's filtering check, use the helper instead
  of `node.run`. Missing inputs are now checked before computing (R13),
  so the input test expects the context's list of missing names.
- Shared-preprocessor and shared-calibration tests, equal models under
  different names, and backtest sharing tests run in one context (D9,
  D10). The run-input test passes inputs through `Context.run`.
- Tests specifically for `Run`, `Builder`, `Pool` and `Cache` stay until
  step 5c-3, including the cache's mutable-node cycle test and dependency
  recording tests. The context-to-`Run` numerical comparison stays too.
- No library code or pinned regression data changed. Next is step 6,
  `Context.from_spec`, before removing the old API.
- Validation: ty, ruff excluding the untracked notebook, and all 658
  tests pass. The strict docs build still fails on the existing README
  link in `reviews/2026-10-25-risk-manager-review.md`.

Done in step 6 (2026-10-27), no pinned output changed:

- `Context.from_spec(spec)` builds a context from a `RunSpec` (R9, R9a,
  R17), with the spec's labels and one backtest function per backtest spec
  (R4). Plain `match` functions in `build.py` build the models inside the
  context, with model keys as node names. Value identity shares equal
  nodes (D9), without using the `Builder` or its tables.
- Input declarations are checked before building: a name declared twice
  differently is still an error (R13). A VaR input's model key must match
  its input name, as decided for step 5; a mismatch raises `ValueError`
  with the model's path. `RunSpec` keeps its existing `last`-window check
  and error paths (R9b).
- The context's graph is checked before returning, without computing.
  Building from a spec restores any surrounding context, on success and
  on error (D10).
- `tests/test_context_specs.py` checks each model family against Python
  nodes, batch values, both EWMA starts, calibrated decays, trailing and
  `Last` windows, input names, labels, shared computation and each
  backtest's settings. A context can run again on different returns.
- The spec guide, the home page and `architecture.md` now show
  `Context.from_spec`. Input model keys in their examples match their
  input names. `RunSpec` still lives in `tail_lib.run`; its name is
  unchanged. `Run`, `Builder` and their tests stay until step 5c-3.
- Validation: ty, ruff excluding the untracked notebook, and all 674
  tests pass. All five Python blocks on the home page and spec guide run.
  The strict docs build still reports only the existing README link.

Done in step 5c-3 (2026-10-27), no pinned output changed:

- `Run`, `RunResult`, `Cache`, `Pool`, `Node.run` and `Builder` are removed.
  `Context.run` is the execution path for Python-built and spec-built
  graphs, and `Results` holds every output (D9–D11, R10, R24).
- `GraphNode`, `GraphEdge`, `RunGraph`, `BacktestFunction` and the graph
  helpers moved from `run.py` into `context.py`. The report reads its
  input-name helper there. `run.py` keeps only `RunSpec`, at its existing
  import path, and its validation (R9, R9b).
- The remaining spec and graph tests use contexts. Graph tests are now in
  `tests/test_context_graph.py`. Dependency tests record the actual
  `compute` call for every node type and check it receives each input's
  output by field name; run input nodes are never computed (R13, D11).
- Twenty cases were removed: tests specific to the deleted APIs and
  duplicates already covered by the context tests. They include the
  `Run` comparison, shared-cache pools, direct cache lookup, cache input
  names, and the mutable-node cycle check. Frozen dataclass nodes cannot
  form a constructor-field cycle; the context has no cycle check for
  custom mutable subclasses, as agreed in step 5a-2.
- All 654 tests pass, including the unchanged regression pins. ty and
  ruff pass with `examples/t.ipynb` excluded. This untracked notebook still
  imports `Run`, so the full ty check reports that removed import; full
  ruff still reports its existing import-order error. The notebook is
  untouched. The strict docs build still reports the existing README link.
- The HS/FHS example runs, as do all 13 Python blocks on the home page,
  spec guide and Python-run guide. The report example writes into a
  temporary directory. Clean interpreters can import `RunSpec` or
  `Context` first, and the removed public names are absent.

**Still open**

- Whether `RunSpec` keeps its name.
- Hashing: nested frozen dataclasses hash again on every lookup. Cache the
  hash if a large run (R6) shows it.
- Building a node inside a `with` block binds it: one line in the guide,
  so it isn't a surprise (R10).

### Run date range (R21–R23)

Agreed with the developer on 2026-10-25, after the risk-manager review
(`docs/reviews/2026-10-25-risk-manager-review.md`, finding 1). Today,
`Windows(..., skip_leading_nan=True)` starts the windows on the first date
where every series has a value. FHS and rolling backtests use it, so with a
`step` above 1, adding a later-starting portfolio to a run shifts every
portfolio's windows and changes its results.

**Agreed**

- The run trims its returns at the first date where every portfolio has a
  value (R21). The latest-starting portfolio sets the start for all; to treat
  portfolios differently, split them into several runs.
- The run computes its VaR models on every date between a lower and an
  upper date (R22). VaR models lose their `step`; calibrations keep theirs.
- Each node declares, for each dependency, how many dates before its first
  output it needs (e.g. an FHS on 500 dates needs 499 dates of filtered
  returns before; the filter needs the volatility of the date before). The
  `Context` walks the graph back from the VaR range and gives each node its
  span, and the VaR covers exactly the range. Each node computes on its
  span, but its output keeps the dates and shape of the run's returns
  (R24, 2026-10-26, replacing outputs of different lengths per node).
  Proposed, to settle in step 3: dates outside a node's span are `NaN`.
- Each node type has a fixed start rule (R23): the first date it is needed,
  or the first available date. `EwmaVolatility` starts as early as
  possible, every other node on the first date it is needed. Only nodes
  whose values depend on their start are affected by the rule: for the
  others, starting later only skips work.
- A shared node starts on the earliest date any of its consumers needs. It
  is computed once (D3), and its consumers see the same values.
- A calibration counts its `step` from its first needed date. A run whose
  range is one date (e.g. today's VaR) so always recalibrates on that date.
- Before computing, the run fails if the trimmed returns can't feed every
  VaR model on the whole range, naming the model and the date it needs
  (R22). Proposed, not yet confirmed: also name the portfolio that sets the
  trimmed start.
- Backtests use the overshootings of the range: the first flag compares the
  return of the date after the lower date with the VaR of the lower date.
- Accepted: the EWMA starts on the trimmed start, so a portfolio's results
  change slightly when a later-starting portfolio joins the run, or when
  more history is passed. The gap shrinks by the decay factor each date
  (with 0.94: about 0.2% after 100 dates, 2e-7 after 250). This weakens D4.
- `NaN`s inside a portfolio's history are left to the review's finding 2
  (validation).

**Steps**, each reviewed on its own. Steps 2 to 5 wait for the context API
(see "Context API" below, developer's choice on 2026-10-26), and will be
built on its graph walk rather than on `Node.dependencies()`:

1. Docs: R21–R23, R16 and D6 marked superseded, this section.
2. Nodes declare what they need and their start rule; the `Context` derives
   each node's span for a given range. No change to results yet; the spans
   can be inspected before a run.
3. `Context` takes the lower and upper dates, trims the returns, checks them,
   and computes each node on its span. `skip_leading_nan` and the VaR
   `step` go away.
4. Backtests on the overshootings of the range.
5. The report shows the range and each node's span.

**Rejected**

- Schedules: one `Every(n)` per run, or named schedules nodes refer to,
  counted from the first observation. VaR has no schedule (R22).
  Date-based schedules (e.g. month ends) were out of scope anyway.
- A late-starting portfolio getting `NaN` until it has enough history,
  while the others keep their dates: replaced by trimming (R21).
- A burn-in setting on the EWMA start: the EWMA starts as early as
  possible instead.
- A copy of a shared node per consumer: computes it more than once, and
  the same node gives different values to different models.
- The start rule as a constructor argument: fixed per node type for now.

**Still open**

- Where the bounds are given (an argument of `Context.run`, a `RunSpec` field,
  or both) and their defaults. To settle in step 3.
- Whether a one-date range replaces `Last` windows for models.

### Run report (R19)

Spec agreed with the developer on 2026-10-25.

**Python side**

The original run/cache design below was replaced by `Context` and `Results`
in steps 5b and 5c-3 of "Context API". It is kept as decision history.

- `Report(run, returns, inputs=None)` computes the run itself, so it can
  read every node's output, not only `RunResult`. `report.save(path)`
  writes the HTML file. Replaced in step 5b of "Context API" by
  `Report(context, results)`, which computes nothing.
- `Run.evaluate(cache)` becomes public, like `Pool.evaluate`. `Run.run`
  turns into a thin wrapper around it. The report builds the `Cache`, calls
  `run.evaluate(cache)`, then reads each graph node from that same cache.
  These nodes are already computed, so reading them costs nothing.
- `Run` exposes its graph: every distinct node once, including the
  `Overshootings` and backtest nodes, with its paths (a shared node has
  several, D3) and its edges, labelled by argument name. Chosen for step 2
  on 2026-10-25: `run.graph` is a frozen `RunGraph` of `nodes` (`GraphNode`:
  the node and its paths, depth first from the models, then the backtests)
  and `edges` (`GraphEdge`: from the dependency to the node reading it,
  with the argument name). Backtest paths mirror `RunResult`:
  `backtests.<model>.<test>`, then `backtests.<model>.<test>.overshootings`.
  `Run.evaluate(cache)` checks for missing inputs before computing, through
  a new read-only `Cache.input_names`; `run()` wraps it.
- Names and labels (R20): model, backtest and input names are Python
  identifiers, so paths are unambiguous. Friendly names are an optional
  `labels` mapping on `Run` and `RunSpec`, from a model or backtest name to
  its display name, falling back to the name: `run.labels` holds every
  name, filled in (chosen over a `run.label(name)` method). Rejected: labels on the report
  only (a report from a JSON spec, R17, would need its own place to send
  them), and a `label` field per spec (specs differing only by label would
  no longer share a node, D3).
- Nodes describe their constructor settings, e.g. `cf`, `window`, through
  an abstract `settings()` next to `dependencies()` (done in step 1).
  Settings that are nodes stay in `dependencies()`. Graph labels show the
  class and the settings, with full details on hover.
- The data passed to the page is one JSON blob, defined by a pydantic
  `ReportData` model. It is the contract with the frontend, and also the
  shape the data service could serve later (R17). Draft contents:
  metadata (title, creation time, tail-lib version), dates, series labels
  for the batch axes, returns, graph nodes (id, kind, class, settings,
  paths) and edges, each time-series node's output by node id, and backtest
  results by model and backtest name. `NaN` becomes `null`.
- No templating library: Python writes the JSON into a
  `<script type="application/json">` slot of the built page.

**Frontend**

- The page is an SPA built with Svelte, TypeScript and Vite. ECharts draws
  both the charts and the graph. `vite-plugin-singlefile` inlines
  everything into one `index.html`.
- The source lives in `report-app/` at the repo root, with its own
  `package.json`. The built `index.html` is committed as package data under
  `src/tail_lib/report/`. CI rebuilds it and fails if the committed file
  differs, so pip installs never need Node.
- The TypeScript types come from `ReportData`'s JSON Schema, so both sides
  are checked against one contract (R11).
- Tabs:
  - **Graph:** one shared DAG, each node once. Returns and inputs at the
    top, then preprocessors, models, overshootings and backtests.
  - **Returns vs VaR:** the returns, each model's VaR as a line, and
    overshootings marked. Models toggle from the legend.
  - **Backtests:** a table per model and backtest with n_obs,
    overshootings, observed vs expected rate, statistic and p-value,
    coloured by rejection. It shows every series.
  - **Rolling backtests:** for windowed backtests, the p-value and observed
    rate at each window's end date.
  - **Node outputs:** click a graph node (a volatility, a filter, a mean)
    to chart its output.
- When there are batch series, a selector picks the series the charts
  show.

**Steps**, each reviewed on its own:

1. Nodes describe their settings.
2. `Run` exposes its graph, and `Run.evaluate(cache)`.
3. In five parts, chosen by the developer on 2026-10-25:
   - 3a. `ReportData` and `build_report_data(run, cache, ...)`: the JSON
     contract, pure Python.
   - 3b. `Report(run, returns, inputs)` and `save(path)`, writing the JSON
     into a placeholder HTML page.
   - 3c. The frontend scaffold in `report-app/`, TS types from the JSON
     Schema, the built page committed as package data, empty tabs.
   - 3d. The CI check: rebuild the page, fail on a diff.
   - 3e. The graph tab.
4. The returns vs VaR tab.
5. The backtest table.
6. The rolling backtests tab.
7. Node outputs on click.

**Rejected**

- Report inputs: `Run + RunResult + returns` (no intermediate outputs) and
  `RunResult` only (no graph).
- Node outputs returned by `Run.run` in `RunResult`: overlaps with R12,
  which is still undesigned.
- JS loaded from a CDN: needs internet access to render.
- Vendoring `echarts.min.js` in the Python package (first chosen): the SPA
  build bundles ECharts instead.
- Jinja2 or `string.Template` to build the page: replaced by one JSON blob
  read by the SPA.
- Building the page during the package build: building a wheel would need
  Node.
- Plotly + Mermaid, Plotly + Cytoscape.js, Vega-Lite + Mermaid: two
  libraries where ECharts does both. Vanilla TS, React and Vue: Svelte
  chosen.
- One graph tree per model: hides shared nodes (D3). Labels with only the
  class and run name: less informative.
- Batch series: rejecting batched returns, or one chart per series.
- Hand-written types on both sides: they can drift apart.
- Settings (step 1, chosen by the developer: an abstract `settings()`
  returning Python values typed by a `Setting` union): settings returned as
  JSON by each node (ties nodes to the report format, repeats the
  conversion), and settings read by introspecting `__init__` (implicit,
  breaks on renamed or derived attributes). Names `parameters()` (blurs
  estimable parameters, which are nodes) and `config()`.

**Later (R19)**

- Showing the report inline in Jupyter (`_repr_html_`), and serving it, or
  its `ReportData` JSON, from the data service (R17).

**Still open**

- File size: every node's output, on every date and for every series, can
  be large. We may want an option to leave out node outputs, or to
  downsample.
- Decided for step 5 (2026-10-25): rejected below a p-value level set in
  the page, 5% by default and not saved (rejected: fixed 5%, a level set
  from Python); only the p-value cell is coloured, red with "rejected" or
  green (rejected: the whole row); a rolling backtest shows its last window
  (rejected: leaving rolling backtests out of the table).
- Step 6: the rejection level is shared by the backtests and rolling tabs.
  The p-value axis is linear, 0 to 1, so small p-values crowd near 0; a log
  axis is an option.
- Step 7: clicking a box in the graph opens its output in Node outputs, or
  Backtests for a backtest node; the returns and run inputs aren't nodes,
  so clicking them does nothing. How each output reads (percent or plain
  number) is a small table by class in the page.
- Decided for 3a (2026-10-25): a node's id is its first path; series are
  labelled by an optional `series_labels` (one per series, C order over
  the batch axes), by index otherwise; settings that aren't numbers or
  text are written as in the specs. Rejected: numbered ids, index labels
  only, `repr()` of the settings.
- Counts (`n_obs`, `n_overshootings`) are written as floats in the JSON
  (`99.0`), as every series of values is `float | null`. Could get an
  integer type if the page needs it.
- 3b: the page's data slot holds `__TAIL_LIB_REPORT_DATA__`; the built SPA
  keeps that slot in `report-app/index.html`. The page's code never writes
  the placeholder (it checks for JSON instead), so it appears once.
- 3c: TypeScript is pinned to 6: `svelte-check` doesn't support 7 yet
  (only through an experimental `--tsgo` setup). ECharts is added in 3e,
  with the first chart. The build is reproducible (same file on a rebuild),
  which the CI check (3d) relies on. The generated types and the sample
  data aren't committed (developer's call): `npm run types` runs before
  `dev`, `check` and `build`. The sample data, 1000 dates, 2 series
  and 5 models, is 800 kB of JSON: see "File size" below.
- 3e: the graph is laid out by our own code (`report-app/src/graph/
  layout.ts`), chosen by the developer over dagre and a force layout. With
  ECharts the page is 508 kB (173 kB gzipped) before any data. The app's
  TypeScript has no unit tests (e.g. of the layout): adding vitest is to
  discuss. Cosmetic: box borders look slightly thicker on the sides.
- Step 4, chosen by the developer: VaR drawn on the date it forecasts;
  overshooting dots only in runs with backtests (rejected: the page
  computing them, and `Run` always holding every model's overshootings).
  Found while testing: ECharts' default zoom filtering (`filterMode:
  "filter"`) made the tooltip read the lines a few dates off, and point
  thinning (`sampling`) can hide extreme returns; both are off. At full
  width, one-date spikes can look shorter than they are, as there are more
  dates than pixels: zoom in to read them. `GraphTab` still has its own
  ECharts setup code, which could use `Chart.svelte`.
- Nodes read through another node are listed in `dependencies()` too, e.g.
  the FHS lists its filter's `vol` and `mean`, so the graph has
  an edge from the volatility to the FHS as well as through the filter.
  The developer chose (2026-10-25) to keep `dependencies()` free of display
  concerns: revisit in the report only if the drawing needs it.

### Serializing results (R12) and reusing them (D4)

`RunResult` holds only model VaRs and backtest results. R12 also asks for
preprocessor outputs (volatilities, calibrations). To decide:

- how to name them, e.g. by their path in the spec (`gauss_99.vol`);
- the file format for arrays;
- how to save backtest results, now that they have a windows axis and
  `end_dates`.

Frozen specs compare and hash by value, so D4 could key saved outputs by
spec, and the JSON dump of a spec records exactly what ran.

### Returns as a node (pinned by the developer)

Today every node reads `cache.returns`. Proposal: a `Returns` node with a
shared default instance `RETURNS`, taken by every node that reads the
returns. `ConstantVolatility` is the exception: it only needs the dates and
shape.

The gain is composability, not reuse:

- demeaned or portfolio returns (R5, R2), e.g. to filter back the
  volatilities of a simulation with an AR mean (R18);
- backtesting against a different P&L series.

Alternatives: B, returns as a named input (`run(inputs={"returns": ...})`),
which loses the reference series that inputs are aligned with; C, keep it
as it is. The developer asked to revisit this later. `VolatilityFilter`
has a spec, but only nested in the FHS spec (see "FHS spec" above).

Filtered historical simulation, first listed as a gain here, got its own
node instead: `FilteredHistoricalSimulationsModel` takes a `VolatilityFilter`
(`(r[t] - mean[t - 1]) / vol[t - 1]`, zero mean by default), and rescales the
quantile with the filter's mean and volatility. One filter can be shared by
several FHS models, e.g. on 500 and 750 dates.

### Simulator and estimation models (R18)

The shocks are a `Distribution` subclass (`distributions.py`), chosen by
the developer over a `Shocks` union, a `Protocol`, one class with a name
field, and wrapped SciPy distributions (loose types, R11). Distributions
give their quantiles too (`quantile`), which `GaussianModel` reads, so a
Student-t VaR, and ES, can reuse them.

Chosen by the developer (option A): the simulator's processes
(`ArProcess`, `EwmaProcess`, `GarchProcess`) are separate from the model
nodes, and share only the one-step recursions (`ewma_update`,
`garch_update`). Rejected (B): a `simulate` method on the volatility nodes,
whose parameters are nodes, and some of which (`FittedEwmaVolatility`,
`VolatilityInput`, a `Backcast` start) can't simulate.

For later (option C, noted by the developer): shared parameter objects,
e.g. `GarchParams(omega, alpha, beta)`, output by a future GARCH
calibration node and taken by the simulator, so a fit can be simulated
from. To revisit when GARCH estimation lands. The library has no GARCH
volatility model yet, nor an AR mean model.

The simulator's recursions run on the residuals `r - mu`, while the
volatility models run on the raw returns. A model only filters back the
simulated volatilities with a zero mean, until the volatility models can
read demeaned returns: see "Returns as a node".

### Runs without VaR models (declined for now)

A `Run` only takes VaR models, since each gets an `Overshootings` node.
To compute volatilities, means or filters on their own, use `Pool` or
`node.run` (option C, chosen by the developer on 2026-10-25). This skips
the run's up-front input checks, and has no spec form. If it's needed
later, the options were: (A) a separate `preprocessors` argument on `Run`
and in `RunResult`, with one spec dict per kind (`volatilities`, `means`,
`filters`), as volatility and mean specs share `type` values such as
`constant` and `input` (recommended, and overlaps with R12); (B) widen
`models` to any `Node[TimeSeries]` and backtest only the VaR ones
(rejected: which models get backtested depends on hidden type checks).

Reopened by the context API (2026-10-26): `Pool` and `node.run` go, and a
context without VaR models computes its other nodes, with the same input
checks. In Python, a volatility built in a context is computed by its
run. Still open: the spec form, where options (A) and (B) apply again.

### Spec shorthands and defaults (R9a)

Also an open question in the requirements. The developer agreed to start
strict, with every node spelled out, and kept it strict for window types.
Candidates: `"decay": 0.94` for a constant decay, and default preprocessors
for a model.

### Per-backtest model filter (declined for now)

Every backtest runs on every model (R4). A filter would let one run hold
both `last` models and backtests, or run different tests on different
models. Declined by the developer: separate runs cover the `last` case. If
it's needed later, the options were: (A) `"models": [...]` on the backtest
spec, which breaks "a spec's fields mirror its node's arguments" and the
sharing of identical backtests; (B) a run-level entry wrapping the test,
`{"test": {...}, "models": [...]}` (recommended for different tests on
different models); (C) one `backtested_models` list on the run. The
`last`-window check in `RunSpec` would then apply only to backtested models.

## Features not started

- **More backtests:** Christoffersen independence and conditional coverage,
  the Basel traffic light, ES backtests (open question). They should take
  the shared `Overshootings` node and an optional `window`, like Kupiec. See
  "Adding a node type" in `docs/architecture.md`.
- **Expected shortfall (R1).**
- **GARCH volatility and AR mean models:** estimation nodes using
  `garch_update`, and an AR mean model. See "Simulator and estimation
  models" for option C.
- **Benchmarking (R4):** ranking criteria are an open question.
- **Multivariate models (R2):** the axis convention is an open question.
