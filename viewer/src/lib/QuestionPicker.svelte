<!--
  Pick a saved question from the batch results (results/<model>/index.json).
  Works in both modes: the files come from the Python server or from GitHub Pages.
-->
<script>
  import { savedIndex } from './api.js';

  let { model, onpick = () => {} } = $props();

  const SKILLS = {
    induction: 'Copying (induction)',
    indirect_object: 'Indirect object (IOI)',
    greater_than: 'Greater-than',
    fact_recall: 'Fact recall',
    agreement: 'Grammar agreement',
  };

  let questions = $state([]);
  let unavailable = $state(false);
  let skill = $state('');
  let chosen = $state('');

  $effect(() => {
    unavailable = false;
    savedIndex(model)
      .then((index) => {
        questions = index.questions;
        skill = questions[0]?.skill ?? '';
      })
      .catch(() => (unavailable = true));
  });

  let skills = $derived([...new Set(questions.map((q) => q.skill))]);
  let inSkill = $derived(questions.filter((q) => q.skill === skill));
</script>

{#if unavailable}
  <p class="muted">No saved results for {model}. Run <code>python -m interp.run --config configs/gpt2_v0.yaml</code>.</p>
{:else if questions.length}
  <div class="picker">
    <label>
      <span>Skill</span>
      <select bind:value={skill} onchange={() => (chosen = '')}>
        {#each skills as s}
          <option value={s}>{SKILLS[s] ?? s} ({questions.filter((q) => q.skill === s).length})</option>
        {/each}
      </select>
    </label>
    <label class="grow">
      <span>Saved question</span>
      <select bind:value={chosen} onchange={() => chosen && onpick(chosen)}>
        <option value="" disabled>Choose a question…</option>
        {#each inSkill as q}
          <option value={q.id}>{q.id} · {q.clean}</option>
        {/each}
      </select>
    </label>
  </div>
{/if}

<style>
  .picker {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
  }
  label span {
    display: block;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .grow {
    flex: 1;
    min-width: 260px;
  }
  select {
    width: 100%;
    font: inherit;
    font-size: 14px;
    padding: 6px 8px;
    border: 1px solid var(--grid-line);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
  }
  .muted {
    color: var(--muted);
    font-size: 13px;
  }
</style>
