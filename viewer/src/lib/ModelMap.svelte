<!--
  V1 model map: rows = layers, columns = attention heads plus one MLP column.
  Values are drawn with a single-hue scale (activation is never negative).
  Colors animate when the values change, so the eye can follow what moved.
-->
<script>
  import * as d3 from 'd3';

  let {
    heads, // number[layer][head]
    mlp, // number[layer]
    sharedScale = false, // one color scale for heads and MLPs, or one each
    selected = null, // { layer, head } where head === null means the MLP
    onselect = () => {},
  } = $props();

  const cell = 28;
  const gap = 2;
  const mlpGap = 14;
  const left = 36;
  const top = 22;

  let nLayers = $derived(heads.length);
  let nHeads = $derived(heads[0]?.length ?? 0);
  let headMax = $derived(d3.max(heads.flat()) ?? 0);
  let mlpMax = $derived(d3.max(mlp) ?? 0);
  let headColor = $derived(
    d3.scaleSequential([0, sharedScale ? Math.max(headMax, mlpMax) : headMax], d3.interpolateBlues),
  );
  let mlpColor = $derived(
    d3.scaleSequential([0, sharedScale ? Math.max(headMax, mlpMax) : mlpMax], d3.interpolateBlues),
  );

  let mlpX = $derived(left + nHeads * (cell + gap) + mlpGap);
  let width = $derived(mlpX + cell + 8);
  let height = $derived(top + nLayers * (cell + gap) + 4);

  let hovered = $state(null);

  const y = (layer) => top + layer * (cell + gap);
  const headX = (head) => left + head * (cell + gap);
  const isSelected = (layer, head) => selected?.layer === layer && selected?.head === head;
  const name = (layer, head) => (head === null ? `MLP ${layer}` : `L${layer} H${head}`);

  function pick(layer, head, value) {
    onselect({ layer, head, value });
  }

  function onKey(event, layer, head, value) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pick(layer, head, value);
    }
  }

  // Legend: a few swatches from 0 to the scale's maximum.
  function swatches(scale) {
    const [, max] = scale.domain();
    return d3.range(6).map((i) => ({ value: (max * i) / 5, color: scale((max * i) / 5) }));
  }
</script>

<div class="map">
  <svg {width} {height} role="group" aria-label="Model map: layers by attention heads plus MLP">
    {#each d3.range(nHeads) as head}
      <text class="axis" x={headX(head) + cell / 2} y={top - 7} text-anchor="middle">{head}</text>
    {/each}
    <text class="axis" x={mlpX + cell / 2} y={top - 7} text-anchor="middle">MLP</text>

    {#each heads as row, layer}
      <text class="axis" x={left - 6} y={y(layer) + cell / 2 + 4} text-anchor="end">L{layer}</text>
      {#each row as value, head}
        <rect
          x={headX(head)}
          y={y(layer)}
          width={cell}
          height={cell}
          rx="3"
          fill={headColor(value)}
          class:selected={isSelected(layer, head)}
          role="button"
          tabindex="0"
          aria-label={`${name(layer, head)}: ${value.toFixed(2)}`}
          onmouseenter={() => (hovered = { layer, head, value })}
          onmouseleave={() => (hovered = null)}
          onclick={() => pick(layer, head, value)}
          onkeydown={(event) => onKey(event, layer, head, value)}
        >
          <title>{name(layer, head)}: {value.toFixed(3)}</title>
        </rect>
      {/each}
      <rect
        x={mlpX}
        y={y(layer)}
        width={cell}
        height={cell}
        rx="3"
        fill={mlpColor(mlp[layer])}
        class:selected={isSelected(layer, null)}
        role="button"
        tabindex="0"
        aria-label={`${name(layer, null)}: ${mlp[layer].toFixed(2)}`}
        onmouseenter={() => (hovered = { layer, head: null, value: mlp[layer] })}
        onmouseleave={() => (hovered = null)}
        onclick={() => pick(layer, null, mlp[layer])}
        onkeydown={(event) => onKey(event, layer, null, mlp[layer])}
      >
        <title>{name(layer, null)}: {mlp[layer].toFixed(3)}</title>
      </rect>
    {/each}
  </svg>

  <p class="hover" aria-live="polite">
    {#if hovered}
      <strong>{name(hovered.layer, hovered.head)}</strong> · activation {hovered.value.toFixed(3)}
    {:else}
      Hover a square for its exact value; click it for details.
    {/if}
  </p>

  <div class="legends">
    {#each sharedScale ? [['Heads and MLPs', headColor]] : [['Heads', headColor], ['MLPs', mlpColor]] as [label, scale]}
      <div class="legend">
        <span class="label">{label}</span>
        {#each swatches(scale) as s}
          <span class="swatch" style:background={s.color} title={s.value.toFixed(2)}></span>
        {/each}
        <span class="range">0 – {scale.domain()[1].toFixed(1)}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .map {
    display: inline-block;
  }
  svg {
    display: block;
    overflow: visible;
  }
  .axis {
    font-size: 11px;
    fill: var(--muted);
  }
  rect {
    cursor: pointer;
    transition: fill 350ms ease;
    stroke: var(--grid-line);
    stroke-width: 0.5;
  }
  rect:hover,
  rect:focus-visible {
    stroke: var(--text);
    stroke-width: 1.5;
    outline: none;
  }
  rect.selected {
    stroke: var(--accent);
    stroke-width: 2.5;
  }
  .hover {
    min-height: 1.4em;
    margin: 6px 0;
    font-size: 13px;
    color: var(--muted);
  }
  .legends {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    font-size: 12px;
    color: var(--muted);
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .label {
    margin-right: 6px;
  }
  .swatch {
    width: 16px;
    height: 12px;
    border: 0.5px solid var(--grid-line);
  }
  .range {
    margin-left: 6px;
    font-variant-numeric: tabular-nums;
  }
</style>
