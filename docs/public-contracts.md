# Public contracts and core test coverage

Inventory of the implemented interface as of 2026-10-27, for the upcoming
test-suite pruning. A contract is behavior callers can rely on: accepted
inputs, numerical meaning, dates and shapes, errors, or an observable
execution guarantee. Each **C** identifier names a contract family, not a
required number of tests.

Sources are the current code, [requirements](requirements.md) and
[user guide](guide/index.md). Requirement IDs retain their existing status:
references to D1–D11 do not promote derived requirements to confirmed ones.
Test ownership and pruning suggestions below are proposals. This inventory
changes neither the interface nor the tests.

## Public surface

Imports use the defining modules; `tail_lib` itself exposes `__version__`
without re-exporting model classes.

- **Execution:** `tail_lib.context.Context`, `Results`, `GraphNode`,
  `GraphEdge`, `RunGraph`; `tail_lib.run.RunSpec`.
- **Data/dates:** `tail_lib.time_series.TimeSeries` (re-exported with `Trailing`, `Last`),
  `tail_lib.calendar.Calendar`; `tail_lib.time_series.windows.Trailing`, `Last`, `Windows`.
- **Models:** `ZeroMean`, `ConstantMean`; `ConstantDecay`,
  `EwmaDecayCalibration`; `ConstantVolatility`, `EwmaVolatility`,
  `VolatilityFilter`, `VolatilityInput`; `HistoricalSimulationsModel`,
  `FilteredHistoricalSimulationsModel`, `GaussianModel`, `ValueAtRiskInput`.
- **Model data/numerical functions:** `Backcast`, `StartingVol`,
  `DecayFactors`, `FilteredReturns`; `starting_variance`, `ewma_update`,
  `ewma_variance`, `ewma_neg_log_likelihood`, `fit_ewma_decay`, `garch_update`.
  Their documented defaults and aliases include `EwmaStart`,
  `InterpolationMethod`, `DEFAULT_EWMA_START`, `DEFAULT_DECAY_BOUNDS`,
  `DEFAULT_FIT_ITERATIONS`.
- **Backtests:** `tail_lib.backtests.overshootings.overshootings`,
  `Overshootings`; `tail_lib.backtests.kupiec.kupiec_pof`, `Kupiec`,
  `KupiecResult`.
- **Configuration:** spec classes, constrained value types, tagged unions,
  and `find_last_window` in `tail_lib.specs` (C28–C32).
- **Extensions:** `Node`, `Preprocessor`, `Backtest`, `RunInput`,
  `Dependencies`, `Setting`, `Settings`, `frozen_floats` in `tail_lib.graph`;
  `MeanModel`, `DecayModel`, `VolatilityModel`, `ValueAtRiskModel` in their
  model modules. `tail_lib.build.BacktestResult` aliases `KupiecResult`.
- **Simulation:** `tail_lib.distributions.Distribution`, `Gaussian`,
  `StudentT`; `tail_lib.simulation.MeanProcess`, `ConstantMeanProcess`,
  `ArProcess`, `VolatilityProcess`, `ConstantVolProcess`, `EwmaProcess`,
  `GarchProcess`, `Simulator`, `Simulation`.
- **Reporting:** `tail_lib.report.report.Report`;
  `tail_lib.report.data.build_report_data`, `ReportData`, `Metadata`,
  `NodeData`, `EdgeData`, `KupiecData`, `BacktestData`, `NodeKind`, `Values`.

Underscored helpers, including all `_build_*` functions and `_KupiecTest`,
are implementation. Their behavior can satisfy a public contract without
their structure becoming a contract. `current_context` and
`DATA_PLACEHOLDER` are orchestration/template hooks, rather than additional
user workflows. Imported dependency names are not additional entry points.
Removed interfaces (`Run`, `RunResult`, `Cache`, `Pool`, `Builder`,
`node.run`) are not compatibility promises.

## Data, dates and windows

Sources: `src/tail_lib/time_series/time_series.py`, `calendar.py`, `time_series/windows.py`.
Requirements: R8, R8a, R21, R24; D5.

- **C01 — Array layout and independence.** Time is axis 0; trailing axes
  are independent series. Model outputs keep `(n_dates, *batch)`.
  Applicable scalar/per-series parameters broadcast to this shape; batched
  computations agree with the corresponding independent computations.
- **C02 — Time-series access/alignment.** `TimeSeries(dates, values)` holds
  caller arrays and exposes `dates`, `values`, `n_dates`. Unequal first-axis
  lengths currently raise `RuntimeError`. Slicing returns a `TimeSeries`
  with dates and values sliced together; non-slice indexing raises
  `TypeError`. `check_aligned` requires equal full shapes and dates, raising
  `ValueError` otherwise. Construction does not copy/freeze arrays or
  generally validate finite values or sorted/unique dates.
- **C03 — Calendar-generated dates.** `Calendar` defaults to Monday–Friday
  without holidays, accepts a custom weekmask/holidays, and exposes their
  normalized values. `dates_from(start, n)` rolls invalid starts forward;
  `dates_to(end, n)` rolls invalid ends backward. Both produce increasing
  valid dates, accept zero dates and reject negative counts.
  `TimeSeries.from_first_date` / `to_last_date` generate one date per value
  on axis 0 using that calendar.
- **C04 — Window selection.** Immutable `Trailing(length, step=1)` and
  `Last(length)` require positive lengths/steps. Trailing windows first end
  at index `length-1`, then every `step` observations; `Last` selects only
  the final `length` observations. Insufficient history raises `ValueError`.
- **C05 — Window layout/warm-up.** `Windows.values` has shape
  `(length, n_windows, *batch)` and `dates` has `(length, n_windows)`;
  `n_windows`, `end_indices`, `end_dates` describe the same selection.
  `skip_leading_nan=False` preserves the original anchor. When true,
  trailing windows/steps start where every series first has a non-NaN
  value; a `Last` window must fit after that point. Interior NaNs remain.
- **C06 — Window results on dates.** `to_series` accepts one result per
  window, restores the original date grid, emits NaN before the first end,
  and holds results until the next window ends. Wrong result counts raise
  `ValueError`. `apply(fn)` passes the complete time-first window array to
  `fn` and maps its output in the same way.

Suggested owners: `tests/test_windows.py`, `test_calendar.py`,
`test_time_series.py`, plus representative batched model/fit cases.
C02 lacks focused construction/slicing/direct-alignment tests today;
date/shape rejection is exercised through consumers.

## Model meaning and numerical behavior

Sources: `src/tail_lib/models/`, [model guide](guide/models.md).
Requirements: R1 (implemented volatility/VaR), R14; D1, D2, D6, D7.

- **C07 — Forecast timing/sign.** A value on observation `t` uses data
  through `t` and forecasts the next observation. Future returns must not
  change earlier forecasts. VaR is the `1-cf` return quantile, negative for
  a loss; `cf` must be in `(0,1)`. Warm-ups are leading NaNs on the original
  grid. Backcasts/fits must not publish before their training data exists.
- **C08 — Constant inputs.** `ZeroMean` produces zeros; `ConstantMean` and
  `ConstantVolatility` repeat their settings on the returns grid.
  `ConstantDecay` repeats a decay in `[0,1]`, with one refit and fit start
  at index 0. Array settings become nested tuples for value identity.
  Incompatible broadcasting fails rather than growing the output shape.
- **C09 — EWMA initialization/recursion.** `Backcast(window=20)` publishes
  the mean squared return of the first `window` observations as variance
  on index `window-1`, with NaN before it. Currently it requires strictly
  more than `window` returns. `StartingVol(vol)` requires positive
  volatility and updates `vol**2` with return 0 on date 0. Subsequently,
  `var[t]=d[t]*var[t-1]+(1-d[t])*r[t]**2`; volatility is its square root.
  These models use raw returns, assuming zero mean. `starting_variance`
  returns the first **update** index and its input variance:
  `(window, backcast)` or `(0, vol**2)`.
- **C10 — EWMA fitting.** `ewma_neg_log_likelihood` scores returns against
  the variance before updating it, using
  `0.5*sum(log(var)+r**2/var)` from the first update index.
  `fit_ewma_decay` fits independently over batch axes and returns the batch
  shape. Defaults: bounds `(0.8,0.9999)`, `Backcast(20)`, 40 iterations.
  Invalid bounds (`0 <= lower < upper <= 1`) or an unfillable backcast
  raise `ValueError`. Numerical agreement within tolerance matters;
  bit-for-bit equality across dependency versions is not promised.
- **C11 — Calibration schedule/data.** `EwmaDecayCalibration` fits every
  training window, publishes at its end, then holds the fit until the next
  refit. `DecayFactors.values` uses the returns grid; `refits` and
  `fit_starts` are matching index arrays. Refits are increasing/in range;
  fit starts are non-negative and no later than their refits. `Last` fits
  once. Steps anchor to common trimmed history, independently of bounds.
- **C12 — Re-filtering.** `EwmaVolatility` re-filters from each fit's
  training start with its decay, publishes from the refit until the next
  refit, and is NaN before the first fit. Calibration/volatility starts
  must be equal; mismatch raises `ValueError` at construction. A `Last`
  calibration gives one final-date volatility.
- **C13 — Volatility filtering.** `VolatilityFilter` computes
  `z[t]=(r[t]-mean[t-1])/vol[t-1]`, with zero mean by default and NaN on
  the first date. `FilteredReturns` carries aligned `standardized`, `mean`,
  `vol`; mismatches raise `ValueError`.
- **C14 — Gaussian VaR.** `GaussianModel` computes
  `mean[t]+vol[t]*Phi_inverse(1-cf)`, with zero mean by default. Inputs must
  match returns' dates/shapes. Zero/negative volatility raises `ValueError`;
  NaN volatility preserves an unknown forecast.
- **C15 — Historical VaR.** `HistoricalSimulationsModel` takes the `1-cf`
  empirical quantile in each window, including its final observation.
  Interpolation defaults to `linear`; `below`/`above` select lower/higher
  order statistics. Full-history execution supports stepped windows with
  held forecasts; `Last` gives one final-date forecast.
- **C16 — Filtered historical VaR.**
  `FilteredHistoricalSimulationsModel` windows standardized returns after
  their common warm-up, then computes
  `mean[t]+vol[t]*Q_(1-cf)(z_window)` using the mean/volatility at the window
  end. Interpolation/held-value behavior matches historical VaR. The
  filter's lag adds one observation to its inputs' warm-up.
- **C17 — GARCH update.** Public `garch_update` computes
  `omega+alpha*eps**2+beta*var`, with broadcasting. This is a one-step
  function; GARCH estimation/volatility nodes are not implemented.

Suggested owners: `tests/test_value_at_risk.py`, `test_volatility.py`,
`test_ewma.py`, and the residual-recursion case in `test_simulation.py`.
Retain independent numerical examples, first-valid/refit transitions and
no-look-ahead checks. Comparing a wrapper only with the function it calls
is not an independent formula oracle. C10's likelihood currently has
indirect fit/regression coverage without an independent likelihood example.

## Context execution, inputs and results

Source: `src/tail_lib/context.py`. Requirements: R3–R6, R13, R17,
R20–R24; D3, D9, D10.

- **C18 — Binding/isolation.** Valid nodes and named `Kupiec` declarations
  built inside `with Context()` bind to the innermost context.
  `Context.current()` returns it or `None`. Nested blocks, exceptions,
  threads and asyncio tasks restore/isolate this state. Nodes built outside
  may be dependencies inside; `bind(node)` supports explicit registration.
  Contexts without VaR models still compute their bound nodes.
- **C19 — Names/labels.** Node/input/backtest names are Python identifiers.
  VaR models/backtest declarations built inside a context require names.
  Unequal nodes cannot share a name; nodes cannot use backtest names.
  Equal nodes may reuse names or have aliases. Duplicate backtest
  registration fails without replacing its declaration. Labels must be
  non-empty and name models/backtests; missing labels default to names.
  `models`/`labels` expose read-only mappings. Some label/graph validation
  is deferred to `graph()`/`run()`.
- **C20 — Once per value per execution.** Equal nodes, including equal
  nested inputs built separately, compute at most once per run.
  Dependencies compute before readers and their outputs are passed under
  dependency field names. Equal models/tests under different names share
  computation. Later runs compute afresh; there is no cross-run cache.
- **C21 — Named external inputs.** `VolatilityInput`/`ValueAtRiskInput`
  read `inputs[name]` without computation. Missing inputs raise `KeyError`
  before computation; used inputs with different dates/full shapes raise
  `ValueError` before computation. Checks use the original grid before
  trimming. Unused inputs are ignored. One input name cannot denote unequal
  nodes, including different roles or VaR confidence levels. Correct
  external-forecast meaning/timing is the caller's responsibility.
- **C22 — Common history.** Returns/used inputs are trimmed to the first
  date where every return series has a non-NaN value. Empty returns or no
  complete date raises `ValueError` before computation. External inputs
  retain later warm-ups; interior missing returns remain. Built-in
  execution leaves caller arrays unchanged. Trimming is per execution.
- **C23 — Inclusive bounds.** Either `lower` or `upper` opts into range
  selection. Bounds accept `datetime.date`/`np.datetime64`, select inclusive
  observations inside trimmed history, and may fall between observations.
  Missing bounds use history endpoints. Bounded dates must be datetime64,
  unique, increasing and without NaT. NaT/reversed/outside bounds or empty
  selections raise `ValueError`. Unbounded execution retains full-history
  warm-ups and stepped VaR.
- **C24 — Bounded-history checks.** Built-in VaRs must forecast from the
  lower date, accounting for dependency, external-input, calibration and
  filter warm-ups. Insufficient history or sparse VaR windows (`step!=1`)
  fails before computation. Calibration steps may exceed 1. Actual VaR is
  also checked afterward: missing selected forecasts, including custom
  warm-ups/interior gaps, raise `ValueError`.
- **C25 — Selection preserves history.** Nodes compute from the full common
  history; bounds neither move refits nor force a refit on the selected
  date. VaR is NaN outside `[lower,upper]` on the full grid, consistently
  through all result accessors and downstream readers. Upstream outputs
  retain history. Flags exist only after the first selected forecast,
  through the upper observation. Backtests receive this flag slice, with
  rolling windows anchored there. One selected observation with backtests
  raises `ValueError`.
- **C26 — Output alignment.** Non-backtest outputs must be `TimeSeries`,
  `DecayFactors` or `FilteredReturns`, containing a series matching returns'
  dates/full shape. Mismatches raise `ValueError`; unsupported types raise
  `TypeError`. Backtest results use a windows axis and are exempt.
- **C27 — Results.** `Results` holds trimmed `returns`, `models[name]`,
  `backtests[model][test]`, every node's `outputs`, and `named` nodes.
  `results[node]` works by node value; `results[name]` resolves a registered
  name. Unknown names/uncomputed nodes raise `KeyError`. `outputs`/`named`
  are read-only; arrays and `models`/`backtests` dicts are not deeply
  immutable. Effective observation bounds belong to each result, or are
  `None` for unbounded execution; reusing contexts preserves earlier bounds.

Suggested owners: `tests/test_context.py`, `test_context_history.py`,
`test_context_range.py`. Duplicates also occur in `test_graph.py`,
`test_inputs.py`, `test_run.py`, `test_build.py`, `test_dependencies.py`.
Nested/thread/async cases exercise distinct isolation guarantees.

## Serialized specifications

Sources: `src/tail_lib/specs.py`, `run.py`, [spec catalogue](guide/specs.md).
Requirements: R9, R9a, R9b, R13, R17, R20; D9.

- **C28 — Spec vocabulary/defaults.** `RunSpec.models` is a non-empty
  name-to-VaR mapping; `backtests`/`labels` default to empty mappings.
  VaR variants: `ValueAtRiskInputSpec` (`input`),
  `HistoricalSimulationsSpec` (`historical`),
  `FilteredHistoricalSimulationsSpec` (`filtered_historical`),
  `GaussianSpec` (`gaussian`). Volatility variants: `VolatilityInputSpec`
  (`input`), `ConstantVolatilitySpec` (`constant`), `EwmaVolatilitySpec`
  (`ewma`). Means: `ZeroMeanSpec` (`zero`), `ConstantMeanSpec` (`constant`).
  Decays: `ConstantDecaySpec` (`constant`), `EwmaDecayCalibrationSpec`
  (`ewma_calibration`). Other forms: `VolatilityFilterSpec`
  (`volatility_filter`), `TrailingSpec` (`trailing`), `LastSpec` (`last`),
  `BackcastSpec` (`backcast`), `StartingVolSpec` (`starting_vol`),
  `KupiecSpec` (`kupiec`). Fields/defaults follow the linked catalogue.
  Union slots require discriminator tags, including windows; Python class
  constructors supply their tags. Single-form filter/Kupiec specs have
  default tags. Numeric decay shorthands are rejected.
- **C29 — Validation/error locations.** Specs reject unknown fields/tags,
  non-finite floats and invalid ranges. Confidence levels are in `(0,1)`,
  decays in `[0,1]`, volatilities positive, counts at least 1, bounds ordered,
  names identifiers and labels non-empty. Public aliases are
  `ConfidenceLevel`, `DecayFactor`, `Positive`, `Count`, `Name`, `Label`.
  `Values[T]` accepts a scalar or non-empty flat list/tuple.
  Pydantic `ValidationError` exposes the faulty field/element location and
  expected constraint. Preserve useful `errors()` locations/categories,
  rather than complete rendered Pydantic messages. Scalar typing uses
  Pydantic coercion, not global strict mode.
- **C30 — Configuration round-trip.** Frozen specs compare by value and
  JSON dumps include defaults and parse back to equal specs. `RunSpec`
  contains dicts: frozen does not imply all specs are hashable/deeply
  immutable. Python/JSON forms describe equivalent configurations.
- **C31 — Spec/Python parity.** `Context.from_spec` produces the same
  defaults/results/sharing/backtest settings as equivalent Python nodes,
  naming models by keys. VaR input keys must equal input names; conflicting
  input declarations raise `ValueError`. Spec construction restores and
  leaves uncontaminated a surrounding context, including on failure.
- **C32 — Whole-run compatibility.** Model/backtest names cannot collide;
  labels must name known models/backtests. Backtests reject `Last` anywhere
  in a model's built-in dependency graph; `Last` on backtests is valid.
  `RunSpec` checks during validation; Python contexts check their graph
  before execution. `find_last_window` returns the first nested `last`
  window's dotted field path or `None`.

Suggested owners: `tests/test_context_specs.py` for parity;
`test_build.py`'s validation cases and `test_run.py`'s spec round-trip /
whole-run checks for the others. Keep one parity matrix spanning distinct
spec variants, rather than overlapping private-builder/conversion tests.

## Overshootings and backtests

Sources: `src/tail_lib/backtests/overshootings.py`, `kupiec.py`.
Requirements: R3–R5, R14, R15; D1, D3.

- **C33 — Overshootings.** `overshootings(var, returns)` requires aligned
  series and flags `r[t] < VaR[t-1]`; equality is not an overshooting.
  Values are `1.0`/`0.0`, or NaN on the first date/when either comparison
  input is missing. `Overshootings(var)` provides the node form and the
  model's `cf_var`.
- **C34 — Kupiec statistic/result.** `kupiec_pof(flags, cf_var, window=None)`
  tests `p=1-cf_var`, omitting NaN flags per series. It returns the
  likelihood ratio against observed `x/n` and chi-squared(1) survival
  probability. Zero/all overshootings use `0*log(0)=0`; no observations
  gives NaN observed rate/statistic/p-value. `KupiecResult` contains
  `end_dates`, `n_obs`, `n_overshootings`, `expected_rate`, `observed_rate`,
  `statistic`, `p_value`. Arrays have `(n_windows,*batch)`, counts are
  integers; invalid `cf_var` raises `ValueError`.
- **C35 — Backtest windows/dates.** Whole-sample testing produces one
  window ending on the final date. `Trailing`/`Last` operate on flags after
  common warm-up. Results belong to the final flag date, not the last VaR
  date. Length/step select samples; results are not expanded to daily
  series. Insufficient post-warm-up history raises `ValueError`.
- **C36 — Declarations apply to every model.** `Kupiec` is frozen and
  keyword-only, defaults to `window=None`, and excludes its optional name
  from equality. Inside a block it needs a name and registers itself;
  outside a block it is reusable through `add_backtest`. Every declaration tests every
  model regardless of registration order, with that model's confidence
  level. Tests share VaR/overshootings. Registration/inspection compute
  nothing and do not leak generated nodes into surrounding contexts.

Suggested owners: `tests/test_overshootings.py`, `test_kupiec.py`, with
registration in `test_context.py` and bounded samples in
`test_context_range.py`. Keep hand-computed statistics and zero/all/empty
cases; retain one sharing check across different backtest windows.

## Extensions and graph inspection

Sources: `src/tail_lib/graph.py`, `context.py`.
Requirements: R6, R10, R11, R19; D9–D11.

- **C37 — Node identity.** Frozen nodes compare by class and comparable
  constructor fields, recursively including inputs. Ordinary names do not
  affect equality/hash; `RunInput.name` does. `frozen_floats` preserves a
  scalar as float and arrays as nested tuples with recoverable values/shape.
  Value identity drives sharing, not object-instance identity.
- **C38 — Dependencies/computation.** `dependencies()` maps node-valued
  comparable constructor fields to input nodes; `settings()` describes
  other comparable fields, including defaults. Metadata and `init=False`
  fields are excluded. `compute` takes one positional returns argument and
  exactly one keyword-only argument per dependency, without variadics.
  Invalid signatures raise `TypeError` at construction. Custom
  `__post_init__` calls the base implementation; computation uses supplied
  outputs/own settings. Direct `RunInput.compute` raises `TypeError`.
- **C39 — Graph inspection.** `Context.graph()` computes nothing and
  returns one `GraphNode` per distinct node, including unused bound nodes,
  with all paths reaching it. `GraphEdge(source,target,name)` points from
  dependency to reader, under the input field name. Inspection does not
  mutate registrations.
- **C40 — Paths/traversal.** Paths start at `models.<name>`,
  `backtests.<model>.<test>`, or `nodes.<name>` / `nodes.<Class>[_N]`, then
  dependency fields. Listing follows first-reached depth-first order from
  models, then backtests, then other roots (latest bound first). Aliases /
  shared inputs contribute multiple paths. This documented order determines
  report IDs; its checks cannot simply be treated as implementation-only.

Suggested owners: `tests/test_graph.py`, `test_context_graph.py`. A small
custom graph can cover wiring/sharing. Spying on every built-in's private
validation call is not necessary to protect the public extension behavior.

## Simulation and distributions

Sources: `src/tail_lib/simulation.py`, `distributions.py`.
Requirements: R8a, R18; D8.

- **C41 — Standardized distributions.** `Distribution` requires
  `draw(rng,shape)`, `quantile(q)`, `check_shape(batch_shape)`. `Gaussian`
  is standard normal. `StudentT(df)` requires `df>2` and scales draws /
  quantiles by `sqrt((df-2)/df)` for unit variance. Quantile probabilities
  are in `(0,1)`; parameters broadcast to the batch shape or raise
  `ValueError` when shape-checked.
- **C42 — Process formulas/initialization.** Every process exposes
  `start(batch_shape)`, `value(state)`, `update(state,x)` and
  `check_shape(batch_shape)`, with an array state; mean processes update
  with returns, volatility processes with residuals. `ConstantMeanProcess`
  (default 0) and `ConstantVolProcess` (positive) keep their value.
  `ArProcess` needs at least one coefficient, forecasts
  `const+sum(ar[k]*past[k])` with newest return first, exposes lag `order`,
  and initializes at `const/(1-sum(ar))`. No coefficients or an exactly unit
  coefficient sum raises `ValueError`; general stationarity is not checked.
  `EwmaProcess` starts at positive `start_vol**2` with decay in `[0,1]`. `GarchProcess`
  starts at `omega/(1-alpha-beta)`, requiring positive omega, non-negative
  alpha/beta, and `alpha+beta<1`. `from_long_run_variance` converts positive
  variance to equivalent omega.
- **C43 — Simulation timing/data.** `Simulator` defaults to a zero
  `ConstantMeanProcess` and Gaussian shocks. It draws `r[t]=mu[t]+sigma[t]*z[t]`, updates variance
  with residuals and mean with returns, then stores next-observation
  means/volatilities on date `t`. `Simulation.returns`, `.means`, `.vols`
  share dates and `(n_dates,*batch_shape)`; custom calendars are honored.
- **C44 — Randomness/burn-in/validation.** `simulate` accepts seed,
  generator or `None`; equal seed/configuration in one environment gives
  equal paths. Burn-in advances the process then discards observations;
  output dates still start at `first_date`. `n_dates>=1`, `burn_in>=0` and
  parameters must broadcast without enlarging batch shape, otherwise
  `ValueError`. Exact draws across arbitrary NumPy versions, EWMA stability
  and general AR stationarity are not guarantees.

Suggested owners: `tests/test_simulation.py`, `test_distributions.py`.
Keep deterministic recurrence/timing, seed, burn-in, batch and invalid-domain
cases. A small seeded statistical smoke set can supplement formula checks.

## Reports and package compatibility

Sources: `src/tail_lib/report/data.py`, `report/report.py`, `pyproject.toml`.
Requirements: R7, R11, R19.

- **C45 — Report existing results.** `build_report_data(context,results,
  title="tail-lib report",series_labels=None)` and `Report(...)` read
  outputs without computation. Results must contain the graph nodes or
  raise `ValueError`; compatibility is by node value, not context-instance
  ID. Reports use stored execution bounds and trimmed history.
  `Report.data` exposes gathered `ReportData`.
- **C46 — Report JSON.** `ReportData` contains `metadata`, `dates`, `series`,
  `returns`, `labels`, `models`, `nodes`, `edges`, `outputs`, `backtests`.
  Nested public models define fields and reject extras. Batch axes flatten
  in C order; values are **series first, then dates/windows**. NaN becomes
  null; dates are ISO day strings. Metadata includes title, UTC creation
  time, version and effective bounds. Missing bounds default to null,
  including older JSON. Series labels must match the flattened count;
  defaults are index strings (`"0"`, `"(0, 1)"`, etc.).
- **C47 — Report graph/output encoding.** IDs are first graph paths. Nodes
  include kind, public-facing class name, settings, paths and input name;
  edges/models reference IDs. Settings use spec-style windows/starts and
  lists for arrays. Outputs include supported time-series nodes, decay
  values and standardized returns. Backtests map model/test names to node
  IDs and windowed result fields. JSON Schema generates frontend types.
- **C48 — Saved reports.** `Report.save(str_or_Path)` writes self-contained
  UTF-8 HTML usable offline without Node/external assets. Embedded JSON
  round-trips, including text containing `</script>`. The package must
  contain a usable template; exact placeholder/bundler text is internal.
  The viewer offers graph, returns/VaR, backtests, rolling and node-output
  views. Styling is not a numerical contract.
- **C49 — Installable typed package.** Declared support is Python `>=3.12`,
  NumPy `>=2.2`, Pydantic `>=2.13`, SciPy `>=1.18.1`.
  `tail_lib.__version__` reads installed metadata. The package ships
  `py.typed` and report HTML; installation needs no frontend build.
  Supported dependency combinations must import/execute the library.

Suggested owners: `tests/test_report_data.py`, `test_report.py`,
`test_dependency_compatibility.py`, plus CI wheel/dependency/report-app
build/type checks. Browser interactions have no dedicated automated suite;
JSON checks alone do not verify the viewer.

## Proposed pruning rules

Baseline: **709 collected tests pass**. This is not a target count. Reduce
repeated scenarios/snapshot multiplication while retaining distinct
contracts, including supporting public interfaces.

1. **Give each family one primary test owner.** Use independent small
   numerical examples for formulas, `Context` behavior for execution, and
   real JSON parse/dump/build/run cases for configuration. A test may cover
   several families; a family may require several meaningful partitions.
2. **Consolidate execution duplication.** `test_context.py`,
   `test_graph.py`, `test_inputs.py`, `test_run.py`, `test_build.py` and
   `test_dependencies.py` overlap on sharing/input/wiring. For example,
   `test_run_computes_each_node_once`,
   `test_context_computes_shared_preprocessors_once` and
   `test_shared_calibration_is_fitted_once_in_a_run` protect C20. Keep a
   shared/nested graph case and spec-based value sharing. Preserve distinct
   multi-window overshootings and thread/task isolation behavior.
3. **Consolidate spec parity.**
   `test_context_specs.py::test_spec_builds_and_runs_like_python` overlaps
   Gaussian/historical/FHS conversion examples in `test_build.py` and
   parity in `test_run.py`. Prefer one variant/default/nesting matrix.
   Preserve validation and round-trips: parity cannot detect a changed
   default shared by both paths or an accepted unknown field.
4. **Replace structural checks with observable behavior.** Candidates:
   `test_dependencies.py`'s subclass enumeration, `_input_name`/`_window`
   recognition and private `_check_compute` spy; `test_build.py`'s direct
   `_build_calibration` use; `test_context.py`'s private `_bind` spy. Preserve
   useful failure modes through construction, `graph()` and `run()` first.
   Public `dependencies()`/`settings()` and documented graph paths remain
   legitimate contract tests.
5. **Reduce pins deliberately.** `test_regression.py` pins every output
   and every Kupiec field across many model/window combinations against
   `tests/data/regression.npz`, originally for the context migration.
   Consider a compact end-to-end numerical fixture, with detailed behavior
   owned by independent formula/timing/refit/range tests. A snapshot records
   current numbers without proving the method. Remove pins only after
   their distinct guarantees have test owners.
6. **Keep meaningful edge partitions.** Scalar/multiple batch axes, first
   forecast, refit transition, last-only output, missing/misaligned inputs,
   no common history, bounds, interior forecast gaps and zero/all/no
   overshootings are distinct risks. Parameterize compactly. Avoid repeating
   the same domain rejection through every consumer when an owner and an
   integration path cover it.
7. **Retain release checks.** Dependency imports, wheel contents, generated
   report types and the bundled report catch failures mathematical tests
   cannot. Keep these in CI without multiplying numerical scenarios.

Before deleting a test, identify its C identifier and the remaining test
that catches the same caller-visible failure. With no applicable contract,
it is an implementation-only candidate. If a contract loses its last
independent check, consolidate/replace that check first.

## Limits and mismatches to resolve separately

These observations do not introduce guarantees or implementation changes.

- **Validation differs between Python and specs.** Specs enforce positive
  constant volatility, finite values and positive iteration counts.
  `ConstantVolatility` does not check positivity; Gaussian VaR checks when
  consuming it. Several NumPy-based checks permit NaN, and some positive
  infinities. `fit_ewma_decay`/`EwmaDecayCalibration` do not directly require
  positive `n_iter`. Do not infer uniform validation from spec tests.
- **Array support differs.** Python nodes accept nested arrays; spec
  `Values[T]` accepts scalars/flat vectors. `TimeSeries` checks only length;
  bounded execution adds date-order validation. General normalization or
  missing-data repair is not promised.
- **EWMA naming can mislead.** `starting_variance` returns the update
  index, one later than a backcast's publication date; its docstring's
  “date the recursion starts on” is ambiguous. The extra-return backcast
  requirement is current behavior, not a mathematical necessity.
- **Report extensibility is limited.** Context execution supports custom
  nodes with supported output types; reporting recognizes specific node
  roles and Kupiec results. Arbitrary backtest serialization is not promised.
- **Future features are not current coverage.** ES, GARCH/AR estimation,
  multivariate models, benchmark rankings, general result persistence /
  reloading, inline notebook reports and a hosted HTTP service are pending.
  R12 needs a separate design; report JSON is not a lossless result archive.
- **Performance is qualitative.** R6 and D3/D9 support batching/shared work;
  there is no stated runtime/memory threshold to pin with unit tests.
