// Talks to the local Python server (`python -m interp.serve`).
// Paths are relative, so this works both behind the Vite dev proxy and when the
// built viewer is served by the Python server. On GitHub Pages there is no API,
// getInfo() fails, and the viewer falls back to static mode.

export async function getInfo() {
  const response = await fetch('api/info');
  const type = response.headers.get('content-type') ?? '';
  if (!response.ok || !type.includes('json')) {
    throw new Error('No live server');
  }
  return response.json();
}

// `answers` = { correct, wrong } tokens for DLA; leave them null for the top two guesses.
export async function analyze(prompt, signal, answers = {}) {
  const response = await fetch('api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, ...answers }),
    signal,
  });
  if (!response.ok) {
    let message = `Server error ${response.status}`;
    try {
      message = (await response.json()).detail ?? message;
    } catch {
      // Body was not JSON; keep the status message.
    }
    throw new Error(message);
  }
  return response.json();
}

// Saved batch results (written by `python -m interp.run`). Served by the Python server
// in live mode, and published next to the viewer on GitHub Pages.
async function getJson(path) {
  const response = await fetch(path);
  const type = response.headers.get('content-type') ?? '';
  if (!response.ok || !type.includes('json')) throw new Error(`Could not load ${path}`);
  return response.json();
}

export const savedModels = () => getJson('results/index.json').then((d) => d.models);
export const savedIndex = (model) => getJson(`results/${model}/index.json`);
export const savedResult = (model, id) => getJson(`results/${model}/${id}.json`).then(checkResult);

// Check that a dropped file looks like a result file before drawing it.
export function checkResult(data) {
  const heads = data?.activation?.heads;
  const mlp = data?.activation?.mlp;
  if (!Array.isArray(heads) || !Array.isArray(heads[0]) || !Array.isArray(mlp)) {
    throw new Error('missing activation.heads / activation.mlp');
  }
  if (heads.length !== mlp.length) {
    throw new Error('heads and mlp have different numbers of layers');
  }
  return data;
}

// Make spaces and newlines visible in tokens (" Mary" -> "·Mary").
export function showToken(token) {
  return token.replaceAll(' ', '·').replaceAll('\n', '↵');
}
