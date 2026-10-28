<script lang="ts">
  // The backtest table: every backtest of every model, for every series.
  // A backtest with several windows shows its last one; the rolling
  // backtests tab shows them all.
  import { modelEntries, percent } from "../models";
  import type { KupiecData, ReportData } from "../report-data";

  // `level`: rejected when the p-value is below it, in percent. Shared with
  // the rolling backtests tab.
  let { data, level = $bindable() }: { data: ReportData; level: number } = $props();

  const models = $derived(modelEntries(data));
  const settings = $derived(new Map(data.nodes.map((node) => [node.id, node.settings])));

  /** Which window the row shows, from the backtest's `window` setting. */
  function windowText(nodeId: string, result: KupiecData): string {
    const window = settings.get(nodeId)?.window as
      | { type: "trailing"; length: number; step: number }
      | { type: "last"; length: number }
      | null
      | undefined;
    const end = result.end_dates.at(-1) ?? "";
    if (window === null || window === undefined) return `whole sample, to ${end}`;
    const n = result.end_dates.length;
    const last = window.type === "trailing" ? ` (last of ${n} windows)` : "";
    return `${window.length} dates to ${end}${last}`;
  }

  /** The last window's value of a result field, for one series. */
  function last(values: (number | null)[][], series: number): number | null {
    return values[series]?.at(-1) ?? null;
  }

  function count(value: number | null): string {
    return value === null ? "–" : value.toFixed(0);
  }

  /** A p-value to 3 decimals, `<0.001` below, so it never reads as 0. */
  function pValueText(value: number | null): string {
    if (value === null) return "–";
    return value < 0.001 ? "<0.001" : value.toFixed(3);
  }

  function fixed(value: number | null, digits: number): string {
    return value === null ? "–" : value.toFixed(digits);
  }

  const rows = $derived(
    models.flatMap((model) =>
      Object.entries(data.backtests[model.name] ?? {}).flatMap(([name, backtest]) =>
        data.series.map((seriesLabel, series) => ({
          key: `${model.name}/${name}/${series}`,
          model: model.label,
          backtest: data.labels[name] ?? name,
          series: seriesLabel,
          window: windowText(backtest.node, backtest.result),
          nObs: last(backtest.result.n_obs, series),
          nOvershootings: last(backtest.result.n_overshootings, series),
          observedRate: last(backtest.result.observed_rate, series),
          expectedRate: backtest.result.expected_rate,
          statistic: last(backtest.result.statistic, series),
          pValue: last(backtest.result.p_value, series),
        })),
      ),
    ),
  );
</script>

{#if rows.length === 0}
  <p class="hint">The run has no backtests.</p>
{:else}
  <p class="hint">
    Kupiec's test of the overshooting rate. A p-value below the level rejects the model.
    Backtests on rolling windows show their last window; see Rolling backtests for every
    window.
  </p>
  <label class="level">
    Rejection level
    <input type="number" bind:value={level} min="0" max="100" step="0.5" /> %
  </label>
  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Backtest</th>
          {#if data.series.length > 1}<th>Series</th>{/if}
          <th>Window</th>
          <th class="num">Dates</th>
          <th class="num">Overshootings</th>
          <th class="num">Observed rate</th>
          <th class="num">Expected rate</th>
          <th class="num">Statistic</th>
          <th class="num">p-value</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as row (row.key)}
          {@const rejected = row.pValue !== null && row.pValue < level / 100}
          <tr>
            <td>{row.model}</td>
            <td>{row.backtest}</td>
            {#if data.series.length > 1}<td>{row.series}</td>{/if}
            <td>{row.window}</td>
            <td class="num">{count(row.nObs)}</td>
            <td class="num">{count(row.nOvershootings)}</td>
            <td class="num">{percent(row.observedRate)}</td>
            <td class="num">{percent(row.expectedRate)}</td>
            <td class="num">{fixed(row.statistic, 2)}</td>
            <td
              class="num"
              class:rejected
              class:accepted={row.pValue !== null && !rejected}
            >
              {pValueText(row.pValue)}{#if rejected}&nbsp;rejected{/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}

<style>
  .hint {
    color: #59636e;
    font-size: 0.9rem;
  }
  .level {
    font-size: 0.9rem;
  }
  .level input {
    width: 4.5rem;
    font: inherit;
  }
  .scroll {
    overflow-x: auto;
    margin-top: 0.75rem;
  }
  table {
    border-collapse: collapse;
    font-size: 0.9rem;
  }
  th,
  td {
    padding: 0.35rem 0.75rem;
    border-bottom: 1px solid #d1d9e0;
    text-align: left;
    white-space: nowrap;
  }
  th {
    font-weight: 600;
    background: #f6f8fa;
  }
  .num {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .rejected {
    background: #ffebe9;
    color: #a40e26;
    font-weight: 600;
  }
  .accepted {
    background: #dafbe1;
  }
</style>
