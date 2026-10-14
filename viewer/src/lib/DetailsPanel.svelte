<!--
  Details for one clicked component. New panels (attention pattern, tokens it
  pushes up or down, DLA and causal scores, logit lens) get added here as the
  views from planning.md §7 are built.
-->
<script>
  let { result, selected, onclose = () => {} } = $props();

  let isMlp = $derived(selected.head === null);
  let name = $derived(isMlp ? `MLP ${selected.layer}` : `Layer ${selected.layer}, head ${selected.head}`);
  let value = $derived(
    isMlp
      ? result.activation.mlp[selected.layer]
      : result.activation.heads[selected.layer][selected.head],
  );
  // Rank among components of the same kind (heads with heads, MLPs with MLPs),
  // because head and MLP output sizes are on different scales.
  let peers = $derived(isMlp ? result.activation.mlp : result.activation.heads.flat());
  let rank = $derived(1 + peers.filter((v) => v > value).length);
</script>

<aside class="panel">
  <header>
    <h3>{name}</h3>
    <button type="button" onclick={onclose} aria-label="Close details">×</button>
  </header>
  <dl>
    <dt>Activation strength</dt>
    <dd>{value.toFixed(3)}</dd>
    <dt>Rank</dt>
    <dd>{rank} of {peers.length} {isMlp ? 'MLPs' : 'heads'}</dd>
  </dl>
  <p class="planned">
    Planned here: attention pattern{isMlp ? '' : ' of this head'}, which output words it pushes up or
    down, and its DLA and causal scores next to this one.
  </p>
</aside>

<style>
  .panel {
    border: 1px solid var(--grid-line);
    border-radius: 8px;
    padding: 12px 16px;
    background: var(--surface);
    min-width: 220px;
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
  button {
    border: none;
    background: none;
    font-size: 20px;
    cursor: pointer;
    color: var(--muted);
  }
  dl {
    display: grid;
    grid-template-columns: auto auto;
    gap: 4px 12px;
    margin: 12px 0;
    font-size: 14px;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
  }
  .planned {
    font-size: 12px;
    color: var(--muted);
    margin: 0;
  }
</style>
