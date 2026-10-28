<script lang="ts">
  import type { ReportData } from "./report-data";
  import BacktestsTab from "./tabs/BacktestsTab.svelte";
  import GraphTab from "./tabs/GraphTab.svelte";
  import NodeOutputsTab from "./tabs/NodeOutputsTab.svelte";
  import ReturnsVarTab from "./tabs/ReturnsVarTab.svelte";
  import RollingTab from "./tabs/RollingTab.svelte";

  let { data }: { data: ReportData | null } = $props();

  // The tabs agreed for R19.
  const tabs = [
    "Graph",
    "Returns vs VaR",
    "Backtests",
    "Rolling backtests",
    "Node outputs",
  ] as const;
  type Tab = (typeof tabs)[number];

  let active: Tab = $state("Graph");
  // The series the charts show, when the returns have several. The graph
  // and the backtest table show every series, so they have no selector.
  let series = $state(0);
  // The p-value level below which a backtest is rejected, in percent, set
  // in the backtests tab. Not saved: a report always opens at 5%.
  let level = $state(5);
  // The node charted in the node outputs tab, picked there or in the graph.
  let selectedNode: string | null = $state(null);

  // A click in the graph: a backtest's result is a table, in Backtests;
  // any other node's output is charted in Node outputs.
  function selectNode(id: string): void {
    if (data?.nodes.find((n) => n.id === id)?.kind === "backtest") {
      active = "Backtests";
    } else {
      selectedNode = id;
      active = "Node outputs";
    }
  }

  const seriesTabs: readonly Tab[] = ["Returns vs VaR", "Rolling backtests", "Node outputs"];
</script>

{#if data === null}
  <main>
    <h1>tail-lib report</h1>
    <p>This page holds no report data. Save a report with <code>Report.save</code>.</p>
  </main>
{:else}
  <header>
    <h1>{data.metadata.title}</h1>
    <p class="meta">
      {Object.keys(data.models).length} models · {data.series.length} series ·
      tail-lib {data.metadata.version} ·
      {new Date(data.metadata.created).toLocaleString()}
    </p>
    <dl class="ranges">
      <div>
        <dt>Forecast range</dt>
        <dd>
          {#if data.metadata.lower != null && data.metadata.upper != null}
            {data.metadata.lower} to {data.metadata.upper}
          {:else}
            Full history
          {/if}
        </dd>
      </div>
      <div>
        <dt>Available history</dt>
        <dd>{data.dates[0]} to {data.dates.at(-1)}</dd>
      </div>
    </dl>
  </header>
  <nav>
    {#each tabs as tab (tab)}
      <button class:active={tab === active} onclick={() => (active = tab)}>{tab}</button>
    {/each}
    {#if data.series.length > 1 && seriesTabs.includes(active)}
      <label class="series">
        Series
        <select bind:value={series}>
          {#each data.series as label, index (index)}
            <option value={index}>{label}</option>
          {/each}
        </select>
      </label>
    {/if}
  </nav>
  <main>
    {#if active === "Graph"}
      <GraphTab {data} onselect={selectNode} />
    {:else if active === "Returns vs VaR"}
      <ReturnsVarTab {data} {series} />
    {:else if active === "Backtests"}
      <BacktestsTab {data} bind:level />
    {:else if active === "Rolling backtests"}
      <RollingTab {data} {series} {level} />
    {:else if active === "Node outputs"}
      <NodeOutputsTab {data} {series} bind:selected={selectedNode} />
    {/if}
  </main>
{/if}

<style>
  :global(body) {
    margin: 0;
    font-family: system-ui, sans-serif;
    color: #1f2328;
  }
  header,
  nav,
  main {
    padding: 0 1.5rem;
  }
  h1 {
    margin: 1rem 0 0.25rem;
    font-size: 1.4rem;
  }
  .meta {
    margin: 0 0 0.5rem;
    color: #59636e;
    font-size: 0.9rem;
  }
  .ranges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 2rem;
    margin: 0 0 1rem;
    font-size: 0.9rem;
  }
  .ranges div {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  .ranges dt {
    font-weight: 600;
  }
  .ranges dd {
    margin: 0;
    color: #59636e;
  }
  nav {
    display: flex;
    gap: 0.25rem;
    border-bottom: 1px solid #d1d9e0;
  }
  nav button {
    padding: 0.5rem 0.9rem;
    border: none;
    border-bottom: 2px solid transparent;
    background: none;
    font: inherit;
    cursor: pointer;
  }
  nav button.active {
    border-bottom-color: #0969da;
    font-weight: 600;
  }
  .series {
    margin-left: auto;
    align-self: center;
    font-size: 0.9rem;
  }
  .series select {
    margin-left: 0.4rem;
    font: inherit;
  }
</style>
