# Report app

The page of tail-lib's run reports: a Svelte + TypeScript app, built
by Vite into one self-contained HTML file,
`src/tail_lib/_report/index.html`, which `Report.save` fills with a run's
data. The built file is committed, so installing tail-lib never needs Node.

## Working on it

From this directory, with Node 22 and `uv`:

```sh
npm install
npm run sample   # writes dev/sample-data.json from a sample run
npm run dev      # serves the page, reading the sample data
```

After a change:

```sh
npm run check    # svelte-check: types, with no warnings
npm run build    # rewrites src/tail_lib/_report/index.html: commit it
```

CI runs `npm ci`, `npm run check` and `npm run build` (job "Check the
report page"), and fails if the built page differs from the committed one.
The Node version is in `.nvmrc`.

## The data contract

The page reads one JSON document, `ReportData`
(`src/tail_lib/_report/data.py`), from its data slot,
`<script id="report-data" type="application/json">`. Keep that slot and its
placeholder in `index.html`: `Report.save` replaces the placeholder.

The TypeScript types come from `ReportData`'s JSON Schema: `npm run types`
writes `src/report-data.schema.json` and `src/report-data.d.ts`. It runs
before `dev`, `check` and `build`, so the types always match the Python
side, and the two files aren't committed. Run it by hand after changing
`ReportData` while `npm run dev` is running.
