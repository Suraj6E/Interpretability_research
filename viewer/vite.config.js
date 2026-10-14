import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  // Relative asset paths, so the same build works on GitHub Pages (served from a
  // sub-path) and when served by `python -m interp.serve`.
  base: './',
  server: {
    // During `npm run dev`, forward API calls to the local Python server.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
});
