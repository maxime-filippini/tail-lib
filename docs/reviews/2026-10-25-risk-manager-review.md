# Risk-manager review of tail-lib

Reviewed on 2026-10-25 at commit `085ec5f`.

## Assessment

tail-lib has a coherent, transparent foundation for running several univariate VaR models on prepared portfolio return histories. Named models, shared volatility/filter nodes, uniform forecast outputs, reusable runs, and JSON specs fit the intended workflow well. The documentation explains the numerical conventions unusually clearly.

The largest gaps are in turning those calculations into a dependable, convenient comparison across many portfolios: batching can change the available dates and even the forecasts of an otherwise unchanged portfolio; comparisons need substantial manual sample alignment; validation is inconsistent; and rolling quantiles create large temporary arrays. These deserve priority over expanding the model catalogue.

I would use the current library for controlled analysis on complete, aligned histories. For repeated desk-wide model comparisons, I would currently need to build a material amount of validation, sample selection, portfolio bookkeeping and result summarization around it.

This review evaluates library behavior and usability, rather than certifying a risk model or assessing regulatory compliance. Recommendations below are proposals, not newly confirmed project requirements.

## Scope and evidence

- Reviewed source, guides, requirements, examples, tests, JSON construction, and report integration.
- Ran the existing checks: **288 tests passed**; `uv run ty check` and `uv run ruff check` passed.
- Compared a complete `(1500, 6)` panel with six separate runs: historical simulation (HS), filtered historical simulation (FHS), Gaussian/EWMA, whole-sample Kupiec and rolling Kupiec produced matching results.
- Checked that appending observations did not change earlier Gaussian/EWMA and FHS forecasts in a small fixed-decay example.
- Reproduced the batch, missing-value, input-validation and documentation cases described below.
- Measured an illustrative workload, with methodology and limitations recorded below. Fitted-EWMA throughput, report/browser rendering, incremental updates and concurrency were not benchmarked. The report-tab issue is source-inferred, rather than browser-tested.

Project requirements informing this review: [R1–R20 and D1–D7](../requirements.md). In particular, R3–R6 concern backtesting, comparison and performance; R8 assigns array construction to the caller but requires input validation; R10 favors understandable orchestration; R17 covers notebooks and JSON services. D1–D7 remain proposed requirements in the repository.

## Strengths to preserve

1. **The run abstraction matches the task.** A dictionary of named VaR models plus a dictionary of named backtest factories naturally expresses a comparison. Applying each backtest to every model is useful consistency, and follows the recorded design. [Run](../../src/tail_lib/run.py#L180)
2. **Shared computations are real.** One Python node instance is computed once per cache; specs automatically deduplicate equal sub-specifications. Sharing a volatility and filter across Gaussian and FHS variants is straightforward once the rule is understood. [Cache](../../src/tail_lib/graph.py#L208), [Builder](../../src/tail_lib/build.py#L86), [sharing guide](../guide/concepts.md#sharing-work)
3. **Timing and signs are explicit.** VaR is a return quantile; the value stored on observation `t` forecasts the next observation; overshootings compare `r[t]` with `VaR[t-1]`. This is consistent in the inspected model and backtest paths. External forecasts remain the caller's responsibility. [Conventions](../guide/concepts.md), [overshootings](../../src/tail_lib/backtests/overshootings.py#L25)
4. **Specs support service integration.** Tagged, frozen specifications reject unknown fields and invalid values, give useful field paths, serialize defaults and build the same runtime objects used in Python. [Specs](../../src/tail_lib/specs.py#L105), [guide](../guide/specs.md)
5. **Latest-only estimates and reports already exist.** `Last` avoids computing historical HS/FHS windows when only the latest forecast is needed. Offline reports expose node settings, intermediate outputs and backtests. Report JSON serialization exists today; it would be inaccurate to describe serialization as wholly absent. [Windows](../../src/tail_lib/windows.py#L48), [Report](../../src/tail_lib/report/report.py#L36), [ReportData](../../src/tail_lib/report/data.py#L156)

## Prioritized findings

### 1. High: portfolio batching can change an individual portfolio's results

**Observed behavior.** `Windows(..., skip_leading_nan=True)` finds the first date on which *every* series has a value. FHS and rolling/last Kupiec use this behavior. A portfolio with a later start delays the batch's window origin. With `step > 1`, that also shifts the schedule of calculations. This follows an explicit recorded design decision, but it conflicts with the ordinary expectation that batching independent portfolios only changes execution efficiency. [First-full-date logic](../../src/tail_lib/windows.py#L196), [schedule](../../src/tail_lib/windows.py#L94), [recorded decision](../open-points.md#rolling-backtests-r15-r16)

**Reproductions.**

- For 20 overshooting flags, portfolio A complete, portfolio B missing its first five flags, `Trailing(4, step=3)` produced no common window end dates between A alone and A in the batch.
- For FHS with `Trailing(10, step=3)`, unchanged returns and unchanged external volatility for A, introducing five leading missing volatilities in companion B moved A's first forecast from index 10 to 15. Later A forecasts differed by as much as `0.0059004` in return units in the seeded example.
- An entirely missing companion series caused rolling Kupiec to raise `ValueError`, preventing results for an otherwise usable portfolio.

**Impact.** Adding, removing, reordering into groups, or chunking portfolios can change the effective analysis. A performance workaround based on portfolio chunks is therefore unsafe for heterogeneous histories unless the sample policy is fixed first. Whole-sample Kupiec behaves differently: it counts available observations independently per series.

**Recommendation.** Decide and expose the sample policy. A portfolio should retain its own dates/results when run alone or alongside unrelated portfolios under an independent policy. A common sample should be an explicit comparison choice. Options include a common calendar schedule with per-series availability masks, or separate series schedules with labeled/ragged results. Until then, document the homogeneous-history requirement prominently, offer a way to inspect the effective start, and explain why naive portfolio chunking can alter results. **R5/R6/R16.**

### 2. High: validation and missing-data behavior need a consistent contract

**Observed behavior.** `TimeSeries` only checks the first dimension's length, and a mismatch raises a bare `RuntimeError`. Unsorted dates, duplicate dates and infinite returns were accepted in direct experiments. A static type annotation does not validate these inputs at runtime. [TimeSeries](../../src/tail_lib/time_series.py#L22)

Validation also differs between Python nodes and specs. `ConstantDecay(np.nan)` is accepted, although specs reject non-finite numbers. `ConstantVolatility(-0.01)` can feed FHS without rejection: in the seeded example it produced a final 99% VaR of `+0.0188858`, reflecting the wrong tail after sign reversal. Gaussian rejects non-positive volatility. FHS/filter do not enforce the same constraint. [Decay checks](../../src/tail_lib/_checks.py#L47), [filter division](../../src/tail_lib/models/volatility.py#L368), [Gaussian checks](../../src/tail_lib/models/value_at_risk.py#L357)

A single missing return at index 100 permanently propagated through fixed-decay EWMA for that portfolio: none of its 1,399 later Gaussian forecasts was finite. Other portfolio forecasts remained usable, but a combined HS/FHS/Gaussian run with rolling tests failed when FHS had no usable flags for the affected series. [EWMA recurrence](../../src/tail_lib/models/ewma.py#L141)

**Recommendation.** Validate dimensions, usable date dtype, strictly increasing unique dates and the finite-value policy once at the boundary. Reject invalid fixed parameters at construction consistently in nodes and specs. For missing observations, choose an explicit reject, propagate or recovery policy; distinguish legitimate warm-up from missing data. Return errors or diagnostics naming the series/date/node. Avoid silently imputing returns. This respects R8: preparing arrays remains the caller's job. **R8/R9b/R11/R17.**

### 3. High: comparison requires an explicit evaluation-sample API

**Observed behavior.** The guide and example manually intersect rolling test end dates. Whole-sample tests end on the same date while using different histories because of warm-ups. Matching rolling end dates is sufficient only when the test lengths and usable-observation sets also match. [Comparison recipe](../guide/backtests.md#comparing-models-on-the-same-windows), [example](../../examples/backtest_hs_fhs.py#L134), [Kupiec counting](../../src/tail_lib/backtests/kupiec.py#L112)

With HS windows of 100 and 103 observations, and a 50-observation Kupiec test stepping by five, the models' test schedules never intersected in the experiment. The current intersection recipe therefore returned no comparison despite ample data.

**Recommendation.** Add a small comparison layer accepting an evaluation interval, schedule and missing-observation policy. Keep training/warm-up history available while restricting the tested flags: slicing the return history first can change model initialization. Return model, portfolio, confidence level, effective sample dates, usable counts, coverage and test outcomes. For common-sample comparisons, check the actual observation masks, rather than just the dates or counts. **R3–R5/R15/R16.**

### 4. High for scale: rolling quantiles need bounded-memory execution

The current sharing architecture avoids repeated preprocessing, but it does not make every numerical operation memory-efficient. `Windows` creates read-only sliding views, which are cheap. HS and FHS then pass the full logical window tensor into `np.quantile`; NumPy's default non-overwriting implementation copies its input. The expansion happens during the reduction, rather than in `sliding_window_view` itself. [Windows](../../src/tail_lib/windows.py#L116), [HS](../../src/tail_lib/models/value_at_risk.py#L180), [FHS](../../src/tail_lib/models/value_at_risk.py#L291), [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)

For 2,520 observations, a 500-observation window and 100 portfolios, the raw float64 returns are about **1.92 MiB**. The logical HS window tensor is about **771 MiB**. Measured process peak memory approached 884 MiB for the workload below.

**Recommendation.** Process rolling quantiles in bounded blocks of window endpoints, preserving the existing timing and schedules. Consider cumulative counts for rolling Kupiec: the current reductions repeatedly visit overlapping windows and allocate temporary arrays. Profile repeated quantiles at different confidence levels on the same sample, because distinct model nodes currently perform separate quantile calls. Benchmark fitted EWMA separately; its likelihood loops and refiltering of training segments present a different cost profile. **R6/D3.**

Prefer controlling working memory before introducing parallel workers, which could multiply these temporary allocations. Do not promise that vectorization automatically beats a portfolio loop: the measurements show that the benefit depends on the model and workload.

## Measured performance

**Workload.** Seed 42; independent normal float64 returns of shape `(2520, P)`, standard deviation 1%; HS(500), FHS(500) with shared fixed-decay EWMA(0.94) and filter, plus Gaussian using the same EWMA. Each model had whole-sample and rolling 250-observation Kupiec tests. Runs used the default 20-observation backcast and daily window steps. Model construction and data generation were outside the timed region; computation used a fresh cache on every call. No report serialization or file I/O was timed.

**Environment.** Linux shared virtual machine, 3 logical CPUs available; Python 3.13.15, NumPy 2.5.4, SciPy 1.18.1. Each row ran in a fresh subprocess, with three repeated computations. Time is the median of three; memory is the process high-water RSS across those computations, including the roughly 102–104 MiB import baseline. These are indicative measurements, rather than controlled capacity or cross-library benchmarks. No hard performance budget is currently specified.

| Portfolios | Execution | Median seconds | Peak RSS, MiB |
|---:|---|---:|---:|
| 1 | One batch | 0.069 | 110.5 |
| 10 | One batch | 0.541 | 189.6 |
| 50 | One batch | 2.739 | 536.7 |
| 100 | One batch | 6.299 | 883.6 |
| 50 | Sequential single-portfolio calls | 2.814 | 111.7 |
| 100 | Latest HS/FHS only, Gaussian history; no backtests | 0.035 | 116.5 |

At 50 portfolios, the full rolling workload had similar batch and sequential runtime, with considerably more batch memory. In a separate Gaussian/EWMA-only check, 50 portfolios took a median 0.047 seconds batched versus 0.098 seconds sequentially: batch support does provide a useful benefit for some workloads. The latest-only row performs a different task and should not be interpreted as an acceleration of the full backtest.

Publish a small benchmark matrix covering portfolio count, history length, window length, model count, confidence-level count and recalibration step, reporting both elapsed time and peak memory. Include current output equivalence as a condition for any optimization. An incremental daily-update API may eventually help repeated production runs, but it is a separate design from the current per-execution cache and the proposed persisted-result reuse D4.

## Backtesting depth and interpretation

Kupiec POF is the only implemented statistical backtest. It handles zero/all overshootings and leaves unusable observations out of its counts; its result includes the quantities needed to inspect coverage. It only evaluates the count of exceptions. [Implementation](../../src/tail_lib/backtests/kupiec.py#L61), [available backtests](../../src/tail_lib/specs.py#L301)

In an experiment with 250 flags and five exceptions, placing all five together or spacing them 50 observations apart gave the same p-value, `0.1618549`, as expected from the formula. A user cannot diagnose exception clustering from that test alone.

Prioritize an independence/conditional-coverage diagnostic and an out-of-sample quantile-loss comparison. Model ranking should state its objective and hold confidence level, horizon and evaluation sample fixed where appropriate. A larger Kupiec p-value does not establish that a model is better. Explain that the chi-squared reference is asymptotic, that sparse tail events limit power, and that overlapping rolling rejection rates are descriptive, rather than independent repeated validations. The small-sample verification difficulty is a central point of [Kupiec's original paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6697).

State catalogue boundaries plainly: HS, FHS, Gaussian with constant/EWMA/fitted-EWMA volatility, and external VaR forecasts are available. Student-t shocks and GARCH simulation do not constitute fitted Student-t/GARCH VaR models. ES is stated in R1 but has no implemented model in the current catalogue. Multivariate portfolio construction is explicitly deferred by R2; its absence is not a defect in this univariate review. [Model specs](../../src/tail_lib/specs.py), [models guide](../guide/models.md), [requirements](../requirements.md)

## Streamlining the API without replacing the transparent core

**Keep the existing `Run` and node layer.** Explicit dependencies and cache sharing serve R10 and are useful to advanced users. The all-tests-on-all-models rule also suits benchmarking. A generic orchestration rewrite would add complexity without resolving the main usage gaps.

The most useful additions to discuss with the author are:

1. **A comparison/summary method** with the explicit sample behavior described above. Provide dependency-light records or arrays; pandas can remain optional. Every row should identify model, portfolio and test, with effective sample and status.
2. **A supported compute-once/report path.** `run.run(...)` creates an internal cache, and `Report(run, returns)` creates a second cache, so a normal calculate-then-report sequence repeats the expensive work. `build_report_data(run, cache)` already supports low-level reuse. Choose either a report-from-cache/data constructor or a small execution object carrying the result and the information needed to report it. A bare `RunResult` alone lacks intermediate node outputs. [Run.run](../../src/tail_lib/run.py#L321), [Report](../../src/tail_lib/report/report.py#L68), [build_report_data](../../src/tail_lib/report/data.py#L188)
3. **Portfolio identity in the result/export workflow.** Core results currently carry positions, whereas friendly series labels are introduced during report creation. Document one mapping from batch positions to portfolio IDs, and preserve it through summaries and exports. Clarify that report JSON flattens batch axes and stores series first, unlike the numerical time-first arrays. [RunResult](../../src/tail_lib/run.py#L90), [report conversion](../../src/tail_lib/report/data.py#L327)
4. **A small curated import surface or optional convenience factories.** The documented first run needs many deep imports and a fair amount of setup. Make the common HS/EWMA-Gaussian/FHS workflow shorter while still exposing the constructed nodes and their settings. Sharing should remain inspectable. This is a usability proposal for author selection, rather than an instruction to rename existing APIs or weaken strict JSON specs.
5. **A clearer run artifact.** Specs and report JSON already save useful information, but a node-built run cannot round-trip into its original spec. Record input identifiers/hashes, portfolio IDs, evaluation policy and a reliable code version alongside settings and outputs. Keep audit/export separate from persistent cache reuse. [Report metadata](../../src/tail_lib/report/data.py#L66), [documented node/spec limitation](../guide/python-runs.md#runs-built-in-python-vs-from-specs)

## Documentation and reporting changes

### Fix existing batch examples first

The backtesting guide indexes `rolling.end_dates` with `rolling.p_value < 0.05`. Dates are one-dimensional, while batched p-values have shape `(n_windows, P)`; this raises an `IndexError`. The adjacent `.mean()` also collapses portfolios and windows into one scalar. [Affected example](../guide/backtests.md#reading-a-result)

For a complete two-dimensional batch, teach per-portfolio aggregation and selection explicitly:

```python
p = rolling.p_value                    # (n_windows, n_portfolios)
share_rejected = (p < 0.05).mean(axis=0)
portfolio_index = 0
dates_rejected = rolling.end_dates[p[:, portfolio_index] < 0.05]
```

That rate example assumes finite p-values. Where p-values are missing, use a finite-value denominator and display insufficient-data status: `NaN < 0.05` evaluates to false and otherwise understates the rejection rate.

### Add one executable multi-portfolio walkthrough

Start with `(T, P)` **already aggregated portfolio return series** and a list of portfolio IDs. Run several models and confidence levels with shared preprocessing, extract a portfolio-by-model summary, compare on a stated evaluation sample, then save a report without recomputation. Explain that columns are independent portfolio series, not component assets being combined or a covariance model. Include histories of different lengths and explain the supported policy.

The present concepts page introduces batching well, but the quickstart and standalone comparison example use one series. The runs guide encourages repeated per-series calls without highlighting the batch option there. The README leads with contributor documentation and omits a minimal consumer run. [Batch concepts](../guide/concepts.md#many-series-at-once), [quickstart](../index.md#a-first-run), [runs guide](../guide/python-runs.md), [README](../../README.md)

### Make operational conventions discoverable

- Show a tiny dated example for importing external VaR, distinguishing forecast/estimation date from the date of the return it forecasts. The existing timing prose is correct; an integration recipe reduces one-observation alignment errors.
- Explain `step` separately for calibration, VaR and backtests. Holding a VaR between calculations changes the forecast process; it is not a numerical optimization that preserves daily estimates.
- Give history requirements and effective output starts before a large run. The current warm-up error is useful, but a preflight summary would make model grids easier to plan.
- Publish the performance/memory guidance above, sharing rules and `Last` behavior, with a pinned version or commit installation example.
- Execute the documentation snippets in CI. A documentation-site build does not detect the reproduced indexing error. The repository already records untested guide snippets as an open point. [Open points](../open-points.md#small-open-points)

### Improve report triage

Offline inspection, portfolio selectors and graph settings are useful. For many books, add filtering by portfolio/model/rejection, sorting and a compact summary of insufficient data and failures. Current table construction eagerly includes every model/test/series row; rendering scale was not measured here. [Backtests table](../../report-app/src/tabs/BacktestsTab.svelte#L48)

One source-inferred correctness issue deserves a targeted check: the rolling tab discovers tests only from the first model and requires that model to have more than one window. With 150 observations, HS(100) first, constant-volatility Gaussian second and rolling Kupiec(50), generated results had one window for HS and 100 for Gaussian. The current predicate hides that rolling test. Determine rolling type from settings, or inspect all models. [Rolling-tab predicate](../../report-app/src/tabs/RollingTab.svelte#L21)

## Suggested delivery order and acceptance criteria

1. **Correctness and sample semantics:** consistent parameter/input validation, explicit missing-data behavior, and a clear independent-versus-common sample choice. Accept when adding an unrelated portfolio cannot change an existing portfolio's results under the independent policy, and invalid sigma/decay/date inputs fail clearly.
2. **Comparison and batch documentation:** an executable batch recipe, corrected indexing and finite-value aggregation, plus a comparison helper returning the effective common observation set. Accept when different warm-ups and sparse schedules either compare correctly or explain precisely why they cannot.
3. **Memory-bounded rolling execution:** chunk numerical work without changing the defined calculation dates, and establish runtime/peak-memory measurements. Accept only with output equivalence on complete and heterogeneous histories under the selected policy.
4. **Workflow completion:** compute once, summarize/export with portfolio identity, and produce a report. Add rejection filtering and the rolling-tab correction.
5. **Backtesting depth and convenience:** add independence/conditional-coverage and quantile-loss tools, followed by curated imports/factories and further model families according to demand.

These changes strengthen the existing explicit design. The main author-facing opportunity is to make the path from prepared portfolio returns to a fair, auditable model comparison as coherent as the path from a node to its numerical output.
