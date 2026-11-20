import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  publicDir: false,
  assetsInclude: ['**/*.onnx'],
  build: {
    assetsInlineLimit: 0,
    sourcemap: false,
  },
});
