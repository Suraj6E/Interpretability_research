<!-- Summary of a saved question: the clean/twin pair, the answers, and the task metric. -->
<script>
  import { showToken } from './api.js';

  let { result } = $props();

  let isGreaterThan = $derived(result.metric?.type === 'prob_diff');
  const signed = (v) => (v > 0 ? '+' : '') + v.toFixed(2);
</script>

<section class="info">
  <h2>{result.id} <small>{result.skill.replace('_', ' ')}</small></h2>
  <dl>
    <dt>Clean</dt>
    <dd>{result.clean}</dd>
    <dt>Twin</dt>
    <dd>{result.twin}</dd>
    <dt>Answer</dt>
    <dd>
      {#if isGreaterThan}
        any year after {String(result.year).padStart(2, '0')}
      {:else}
        <span class="chip">{showToken(result.correct)}</span> (twin: <span class="chip">{showToken(result.wrong)}</span>)
      {/if}
    </dd>
    <dt>Top guess</dt>
    <dd>
      <span class="chip">{showToken(result.top1.clean)}</span> (twin: <span class="chip">{showToken(result.top1.twin)}</span>)
    </dd>
    <dt>{isGreaterThan ? 'Prob. difference' : 'Logit difference'}</dt>
    <dd>clean {signed(result.metric.clean)} · twin {signed(result.metric.twin)}</dd>
  </dl>
</section>

<style>
  .info {
    margin-bottom: 20px;
  }
  h2 {
    font-size: 14px;
    margin: 0 0 8px;
  }
  small {
    font-weight: normal;
    color: var(--muted);
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 3px 14px;
    margin: 0;
    font-size: 13px;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
  }
  .chip {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12px;
    padding: 0 5px;
    border-radius: 4px;
    background: var(--surface);
    border: 1px solid var(--grid-line);
  }
</style>
