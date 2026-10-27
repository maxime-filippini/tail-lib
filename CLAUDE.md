# tail-lib

Read `docs/requirements.md` before designing, implementing or reviewing a
feature. Cite requirement IDs (R1, D2, ...) in plans and hand-offs.
`docs/architecture.md` explains how the code is orchestrated, and
`docs/open-points.md` lists pending work: keep both up to date with each
change.
Stated requirements record only what the developer said. Anything proposed
during design goes under Derived until the developer confirms it.

## Working with the core developer

The core developer reviews every change and must always understand the
orchestration and the methodology (R10). Work in small increments:

- **One concern per change.** Make a change small enough to review in one
  sitting: a refactor, a fix or a feature, kept separate. After each change,
  stop and hand it back for review.
- **Decide design with the developer.** For any new abstraction, public API, or
  change to how models are orchestrated, propose the options and trade-offs
  first, then implement the option the developer picks.
- **Traceable orchestration.** Use plain, explicit control flow, so a reader
  can follow a call from the entry point to the numbers by reading the code.
- **Documented methodology.** Each model's docstring states its formula, its
  timing convention (which dates feed the forecast for date `t`) and its
  parameters. It also cites a reference when the method comes from the
  literature.
- **Nodes for estimable parameters.** A model parameter that can be estimated
  from the returns (a volatility, a decay factor) is an input node: a field
  holding another node, whose output `compute` receives. A parameter always
  set by hand (a confidence level, a window length, fit bounds) is a setting:
  a field holding a plain value. If any variant of an input can be estimated,
  every variant is a node, e.g. `ConstantDecay` next to
  `EwmaDecayCalibration`.
- **Pure `compute`.** A node's `compute` uses only its arguments and the node's
  own settings, never another node's settings. It takes the returns, then one
  keyword-only argument per input field, holding that input's output; the run
  computes the inputs first and passes them in.
- **Green checks.** Before proposing a change, run `uv run ty check`,
  `uv run ruff check` and `uv run pytest`. Hand off with all three passing, or
  list each remaining failure and why it remains.
- **Hand-off.** Summarize what changed, why, and which requirements it serves.
  Also list anything you deferred or noticed but left untouched.
