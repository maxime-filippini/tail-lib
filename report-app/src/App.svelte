<script lang="ts">
  import type { ReportData } from "./report-data";
  import GraphTab from "./tabs/GraphTab.svelte";

  let { data }: { data: ReportData | null } = $props();

  // The tabs agreed for R19. Each is filled in its own step.
  const tabs = [
    "Graph",
    "Returns vs VaR",
    "Backtests",
    "Rolling backtests",
    "Node outputs",
  ] as const;
  type Tab = (typeof tabs)[number];

  let active: Tab = $state("Graph");
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
      {data.dates[0]} to {data.dates.at(-1)} · tail-lib {data.metadata.version} ·
      {new Date(data.metadata.created).toLocaleString()}
    </p>
  </header>
  <nav>
    {#each tabs as tab (tab)}
      <button class:active={tab === active} onclick={() => (active = tab)}>{tab}</button>
    {/each}
  </nav>
  <main>
    {#if active === "Graph"}
      <GraphTab {data} />
    {:else}
      <p class="todo">{active}: not built yet.</p>
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
    margin: 0 0 1rem;
    color: #59636e;
    font-size: 0.9rem;
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
  .todo {
    color: #59636e;
  }
</style>
