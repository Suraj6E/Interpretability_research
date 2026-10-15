<script>
  import { onMount } from 'svelte';
  import ModelMap from './lib/ModelMap.svelte';
  import DetailsPanel from './lib/DetailsPanel.svelte';
  import QuestionPicker from './lib/QuestionPicker.svelte';
  import QuestionInfo from './lib/QuestionInfo.svelte';
  import { getInfo, analyze, checkResult, showToken, savedModels, savedResult } from './lib/api.js';

  // 'checking' until we know whether a live server answers; then 'live' or 'static'.
  let mode = $state('checking');
  let info = $state(null);
  let savedModel = $state(null); // model whose saved results the picker lists
  let prompt = $state('When Mary and John went to the store, John gave a drink to');
  let answers = $state({ correct: null, wrong: null }); // null = the model's top two guesses
  let result = $state(null);
  let source = $state('live'); // 'live', 'saved' or 'file'
  let view = $state('activation'); // 'activation' or 'dla'
  let error = $state('');
  let busy = $state(false);
  let selected = $state(null);
  let sharedScale = $state(false);
  let dragging = $state(false);

  onMount(async () => {
    try {
      info = await getInfo();
      mode = 'live';
      savedModel = info.model;
    } catch {
      mode = 'static';
      savedModel = await savedModels().then((models) => models[0]).catch(() => null);
    }
  });

  // Live mode: run the model shortly after the prompt (or the chosen answers) stop changing.
  let controller;
  let requestId = 0;
  $effect(() => {
    if (mode !== 'live') return;
    const text = prompt;
    const chosen = { ...answers };
    if (!text.trim()) return;
    const timer = setTimeout(() => run(text, chosen), 250);
    return () => clearTimeout(timer);
  });

  async function run(text, chosen) {
    controller?.abort();
    controller = new AbortController();
    const id = ++requestId;
    busy = true;
    try {
      result = await analyze(text, controller.signal, chosen);
      source = 'live';
      error = '';
    } catch (e) {
      if (e.name !== 'AbortError') error = e.message;
    } finally {
      if (id === requestId) busy = false;
    }
  }

  function onPromptInput() {
    // A new prompt has new top guesses, so go back to comparing those.
    answers = { correct: null, wrong: null };
  }

  async function pickSaved(id) {
    try {
      result = await savedResult(savedModel, id);
      source = 'saved';
      error = '';
    } catch (e) {
      error = e.message;
    }
  }

  // Static mode: load a saved result file by drag-and-drop or the file picker.
  async function loadFile(file) {
    if (!file) return;
    try {
      result = checkResult(JSON.parse(await file.text()));
      source = 'file';
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

  let hasDla = $derived(Boolean(result?.dla));
  let shownView = $derived(view === 'dla' && hasDla ? 'dla' : 'activation');
  let isGreaterThan = $derived(result?.metric?.type === 'prob_diff');
  let dlaTarget = $derived(
    !result
      ? ''
      : isGreaterThan
        ? `years after ${String(result.year).padStart(2, '0')}`
        : `"${showToken(result.correct ?? '')}" rather than "${showToken(result.wrong ?? '')}"`,
  );
  // When a head is selected, shade each token by how much the last token attends to it.
  let attentionRow = $derived(
    selected && selected.head !== null && result?.attention
      ? result.attention[selected.layer][selected.head]
      : null,
  );
  let guessTokens = $derived(result?.top_predictions?.map((g) => g.token) ?? []);
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
      <span>Prompt {#if source !== 'live'}<em>(showing a saved question; edit to go back to live)</em>{/if}</span>
      <textarea bind:value={prompt} oninput={onPromptInput} rows="2" spellcheck="false"></textarea>
    </label>
  {:else if mode === 'static'}
    <div class="drop">
      Pick a saved question below, or drop a result file (<code>results/&lt;model&gt;/&lt;id&gt;.json</code>)
      anywhere on this page, or
      <label class="file">choose one<input type="file" accept=".json,application/json" onchange={(e) => loadFile(e.currentTarget.files[0])} /></label>.
      To type your own prompts, run <code>python -m interp.serve</code> locally.
    </div>
  {/if}

  {#if savedModel}
    <QuestionPicker model={savedModel} onpick={pickSaved} />
  {/if}

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if result}
    {#if source === 'saved' || (source === 'file' && result.metric)}
      <QuestionInfo {result} />
    {/if}

    <section class="tokens" class:stale={busy}>
      <div>
        <h2>Tokens {#if attentionRow}<small>shaded by L{selected.layer} H{selected.head}'s attention from the last token</small>{/if}</h2>
        <div class="chips">
          {#each result.tokens as token, i}
            <span
              class="chip"
              class:last={i === result.tokens.length - 1}
              style:background={attentionRow
                ? `color-mix(in srgb, var(--accent) ${Math.round(attentionRow[i] * 100)}%, var(--surface))`
                : null}
              title={attentionRow ? `${(attentionRow[i] * 100).toFixed(1)}% attention` : null}
            >{showToken(token)}</span>
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
        <div class="view-bar">
          <div class="tabs" role="tablist" aria-label="Map view">
            <button type="button" role="tab" aria-selected={shownView === 'activation'} onclick={() => (view = 'activation')}>
              Activation
            </button>
            <button type="button" role="tab" aria-selected={shownView === 'dla'} disabled={!hasDla} onclick={() => (view = 'dla')}>
              Contribution (DLA)
            </button>
          </div>
          {#if shownView === 'dla' && source === 'live' && guessTokens.length > 1}
            <div class="answers">
              <label>
                toward
                <select value={result.correct} onchange={(e) => (answers = { correct: e.currentTarget.value, wrong: result.wrong })}>
                  {#each guessTokens as t}<option value={t}>{showToken(t)}</option>{/each}
                </select>
              </label>
              <label>
                rather than
                <select value={result.wrong} onchange={(e) => (answers = { correct: result.correct, wrong: e.currentTarget.value })}>
                  {#each guessTokens as t}<option value={t}>{showToken(t)}</option>{/each}
                </select>
              </label>
            </div>
          {/if}
        </div>

        <h2>
          {#if shownView === 'dla'}
            Direct push toward {dlaTarget} <small>at the last token</small>
          {:else}
            Activation strength <small>at the last token</small>
          {/if}
        </h2>
        <ModelMap
          values={shownView === 'dla' ? result.dla : result.activation}
          kind={shownView === 'dla' ? 'diverging' : 'sequential'}
          label={shownView === 'dla' ? 'DLA' : 'activation'}
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
            column is that layer's MLP. Click a head to see which tokens it reads.
          </p>
          {#if shownView === 'dla'}
            <p>
              <strong>Blue</strong> parts push the model directly toward {dlaTarget};
              <strong>red</strong> parts push the other way. White means little direct effect.
            </p>
            <p>
              <strong>Direct effects only.</strong> A part can also matter by changing later parts,
              which DLA does not count. Still correlation: the causal view (planned) tests what is
              actually needed.
            </p>
          {:else}
            <p>Darker means the part's output was bigger (its L2 norm).</p>
            <p>
              <strong>Correlation only.</strong> A dark square fired strongly. That does not show it
              mattered for the answer, or even which way it pushed: see the DLA view and, later, the
              causal view.
            </p>
          {/if}
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
    max-width: 640px;
  }
  .chip {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 13px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--surface);
    border: 1px solid var(--grid-line);
    transition: background 300ms ease;
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
  .view-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }
  .tabs {
    display: inline-flex;
    border: 1px solid var(--grid-line);
    border-radius: 6px;
    overflow: hidden;
  }
  .tabs button {
    font: inherit;
    font-size: 13px;
    padding: 5px 12px;
    border: none;
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
  }
  .tabs button[aria-selected='true'] {
    background: var(--accent);
    color: #fff;
  }
  .tabs button:disabled {
    color: var(--muted);
    cursor: not-allowed;
  }
  .answers {
    display: flex;
    gap: 10px;
    font-size: 13px;
    color: var(--muted);
  }
  .answers select {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 13px;
    margin-left: 4px;
    padding: 2px 4px;
    border: 1px solid var(--grid-line);
    border-radius: 4px;
    background: var(--surface);
    color: var(--text);
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
