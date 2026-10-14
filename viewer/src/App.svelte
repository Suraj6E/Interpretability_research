<script>
  import { onMount } from 'svelte';
  import ModelMap from './lib/ModelMap.svelte';
  import DetailsPanel from './lib/DetailsPanel.svelte';
  import { getInfo, analyze, checkResult, showToken } from './lib/api.js';

  // 'checking' until we know whether a live server answers; then 'live' or 'static'.
  let mode = $state('checking');
  let info = $state(null);
  let prompt = $state('When Mary and John went to the store, John gave a drink to');
  let result = $state(null);
  let error = $state('');
  let busy = $state(false);
  let selected = $state(null);
  let sharedScale = $state(false);
  let dragging = $state(false);

  onMount(async () => {
    try {
      info = await getInfo();
      mode = 'live';
    } catch {
      mode = 'static';
    }
  });

  // Live mode: run the model shortly after the prompt stops changing.
  let controller;
  let requestId = 0;
  $effect(() => {
    if (mode !== 'live') return;
    const text = prompt;
    if (!text.trim()) return;
    const timer = setTimeout(() => run(text), 250);
    return () => clearTimeout(timer);
  });

  async function run(text) {
    controller?.abort();
    controller = new AbortController();
    const id = ++requestId;
    busy = true;
    try {
      result = await analyze(text, controller.signal);
      error = '';
    } catch (e) {
      if (e.name !== 'AbortError') error = e.message;
    } finally {
      if (id === requestId) busy = false;
    }
  }

  // Static mode: load a saved result file by drag-and-drop or the file picker.
  async function loadFile(file) {
    if (!file) return;
    try {
      result = checkResult(JSON.parse(await file.text()));
      selected = null;
      error = '';
    } catch (e) {
      error = `Could not read ${file.name}: ${e.message}`;
    }
  }

  function onDrop(event) {
    event.preventDefault();
    dragging = false;
    loadFile(event.dataTransfer.files[0]);
  }
</script>

<svelte:window
  ondragover={(event) => {
    event.preventDefault();
    dragging = true;
  }}
  ondragleave={() => (dragging = false)}
  ondrop={onDrop}
/>

<main class:dragging>
  <header>
    <h1>Model map</h1>
    {#if mode === 'live'}
      <span class="badge live">Live · {info.model} on {info.device}</span>
    {:else if mode === 'static'}
      <span class="badge">Static · saved results</span>
    {/if}
  </header>

  {#if mode === 'live'}
    <label class="prompt">
      <span>Prompt</span>
      <textarea bind:value={prompt} rows="2" spellcheck="false"></textarea>
    </label>
  {:else if mode === 'static'}
    <div class="drop">
      Drop a result file (<code>results/&lt;model&gt;/&lt;id&gt;.json</code>) anywhere on this page, or
      <label class="file">choose one<input type="file" accept=".json,application/json" onchange={(e) => loadFile(e.currentTarget.files[0])} /></label>.
      To type your own prompts, run <code>python -m interp.serve</code> locally.
    </div>
  {/if}

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if result}
    <section class="tokens" class:stale={busy}>
      <div>
        <h2>Tokens</h2>
        <div class="chips">
          {#each result.tokens as token, i}
            <span class="chip" class:last={i === result.tokens.length - 1}>{showToken(token)}</span>
          {/each}
        </div>
      </div>
      {#if result.top_predictions}
        <div>
          <h2>Model's next-word guesses</h2>
          <ol class="guesses">
            {#each result.top_predictions as guess}
              <li><span class="chip">{showToken(guess.token)}</span> {(guess.prob * 100).toFixed(1)}%</li>
            {/each}
          </ol>
        </div>
      {/if}
    </section>

    <section class="views" class:stale={busy}>
      <div>
        <h2>Activation strength <small>at the last token</small></h2>
        <ModelMap
          heads={result.activation.heads}
          mlp={result.activation.mlp}
          {sharedScale}
          {selected}
          onselect={(s) => (selected = s)}
        />
        <label class="toggle">
          <input type="checkbox" bind:checked={sharedScale} /> Same color scale for heads and MLPs
        </label>
      </div>

      <div class="side">
        {#if selected}
          <DetailsPanel {result} {selected} onclose={() => (selected = null)} />
        {/if}
        <aside class="howto">
          <h3>How to read this</h3>
          <p>
            Each square is one part of the model, measured at the <strong>last token</strong> of the
            prompt. Rows are layers (L0 is first). The 12 columns are attention heads; the last
            column is that layer's MLP. Darker means the part's output was bigger (its L2 norm).
          </p>
          <p>
            <strong>Correlation only.</strong> A dark square fired strongly. That does not show it
            mattered for the answer. The causal view (planned) tests that.
          </p>
        </aside>
      </div>
    </section>
  {:else if mode === 'live'}
    <p class="muted">Running the model…</p>
  {/if}
</main>

<style>
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px 16px 48px;
  }
  main.dragging {
    outline: 3px dashed var(--accent);
    outline-offset: -8px;
  }
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }
  h1 {
    margin: 0;
    font-size: 24px;
  }
  h2 {
    font-size: 14px;
    margin: 0 0 8px;
  }
  h2 small {
    font-weight: normal;
    color: var(--muted);
  }
  h3 {
    font-size: 14px;
    margin: 0 0 6px;
  }
  .badge {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--grid-line);
    color: var(--muted);
  }
  .badge.live {
    color: var(--live);
    border-color: var(--live);
  }
  .prompt {
    display: block;
    margin-bottom: 16px;
  }
  .prompt span {
    display: block;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 4px;
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    font: inherit;
    font-size: 16px;
    padding: 8px 10px;
    border: 1px solid var(--grid-line);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
    resize: vertical;
  }
  .drop {
    padding: 16px;
    border: 1px dashed var(--grid-line);
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 14px;
    line-height: 1.6;
  }
  .file {
    color: var(--accent);
    text-decoration: underline;
    cursor: pointer;
  }
  .file input {
    display: none;
  }
  .error {
    color: var(--error);
    font-size: 14px;
  }
  .tokens {
    display: flex;
    flex-wrap: wrap;
    gap: 24px 40px;
    margin-bottom: 20px;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .chip {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 13px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--surface);
    border: 1px solid var(--grid-line);
  }
  .chip.last {
    border-color: var(--accent);
  }
  .guesses {
    margin: 0;
    padding-left: 20px;
    font-size: 13px;
    line-height: 1.8;
  }
  .views {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
  }
  .side {
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-width: 340px;
  }
  .howto {
    font-size: 13px;
    line-height: 1.5;
    color: var(--muted);
  }
  .howto p {
    margin: 0 0 8px;
  }
  .toggle {
    display: block;
    margin-top: 8px;
    font-size: 13px;
    color: var(--muted);
  }
  .stale {
    opacity: 0.6;
    transition: opacity 150ms;
  }
  .muted {
    color: var(--muted);
  }
</style>
