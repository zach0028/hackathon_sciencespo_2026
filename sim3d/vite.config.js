import { defineConfig } from 'vite';
import path from 'node:path';

export default defineConfig({
  base: './',
  build: {
    outDir: path.resolve(__dirname, '../src/hackathon_osmo/app/static/sim_dist'),
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://127.0.0.1:8765',
        ws: true,
      },
      '/api': {
        target: 'http://127.0.0.1:8765',
      },
    },
  },
});
