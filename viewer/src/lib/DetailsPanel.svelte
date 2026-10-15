<!--
  Details for one clicked component: its score in every view, with ranks, and (for heads)
  which tokens the last position attends to (V3). More panels (tokens it pushes up or
  down, causal scores, logit lens) get added here as the views in planning.md §7 are built.
-->
<script>
  import { showToken } from './api.js';

  let { result, selected, onclose = () => {} } = $props();

  let isMlp = $derived(selected.head === null);
  let name = $derived(isMlp ? `MLP ${selected.layer}` : `Layer ${selected.layer}, head ${selected.head}`);

  // Rank among components of the same kind (heads with heads, MLPs with MLPs), because
  // head and MLP outputs are on different scales. Signed views rank by size (absolute value).
  function score(map, signed) {
    const value = isMlp ? map.mlp[selected.layer] : map.heads[selected.layer][selected.head];
    const peers = isMlp ? map.mlp : map.heads.flat();
    const size = (v) => (signed ? Math.abs(v) : v);
    return { value, rank: 1 + peers.filter((v) => size(v) > size(value)).length, of: peers.length };
  }

  let activation = $derived(score(result.activation, false));
  let dla = $derived(result.dla ? score(result.dla, true) : null);
  let attention = $derived(
    !isMlp && result.attention ? result.attention[selected.layer][selected.head] : null,
  );
  let maxAttention = $derived(attention ? Math.max(...attention) : 1);
</script>

<aside class="panel">
  <header>
    <h3>{name}</h3>
    <button type="button" onclick={onclose} aria-label="Close details">×</button>
  </header>

  <dl>
    <dt>Activation</dt>
    <dd>{activation.value.toFixed(2)} <small>rank {activation.rank} of {activation.of}</small></dd>
    {#if dla}
      <dt>DLA</dt>
      <dd>
        {dla.value > 0 ? '+' : ''}{dla.value.toFixed(3)}
        <small>
          {dla.value >= 0 ? 'toward' : 'away from'} the correct answer · rank {dla.rank} of {dla.of} by size
        </small>
      </dd>
    {/if}
  </dl>

  {#if attention}
    <h4>What the last token reads <small>(attention)</small></h4>
    <ol class="attention">
      {#each attention as weight, i}
        <li>
          <span class="token">{showToken(result.tokens[i])}</span>
          <span class="bar" style:width={`${(weight / maxAttention) * 100}%`}></span>
          <span class="weight">{(weight * 100).toFixed(0)}%</span>
        </li>
      {/each}
    </ol>
  {:else if isMlp}
    <p class="note">MLPs work on one token at a time, so they have no attention pattern.</p>
  {/if}
</aside>

<style>
  .panel {
    border: 1px solid var(--grid-line);
    border-radius: 8px;
    padding: 12px 16px;
    background: var(--surface);
    min-width: 260px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  h3 {
    margin: 0;
    font-size: 15px;
  }
  h4 {
    font-size: 13px;
    margin: 12px 0 6px;
  }
  small {
    color: var(--muted);
    font-weight: normal;
  }
  button {
    border: none;
    background: none;
    font-size: 20px;
    cursor: pointer;
    color: var(--muted);
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    margin: 12px 0 0;
    font-size: 14px;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
  }
  .attention {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: 12px;
  }
  .attention li {
    display: grid;
    grid-template-columns: 90px 1fr 36px;
    align-items: center;
    gap: 6px;
    margin: 2px 0;
  }
  .token {
    font-family: ui-monospace, Consolas, monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar {
    height: 10px;
    background: var(--accent);
    border-radius: 2px;
    transition: width 300ms ease;
  }
  .weight {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
  }
  .note {
    font-size: 12px;
    color: var(--muted);
    margin: 12px 0 0;
  }
</style>
