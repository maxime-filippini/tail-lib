# tail-lib

Read `docs/requirements.md` before designing, implementing or reviewing a
feature. Cite requirement IDs (R1, D2, ...) in plans and hand-offs.

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
- **Hand-off.** Summarize what changed, why, and which requirements it serves.
  Also list anything you deferred or noticed but left untouched.
