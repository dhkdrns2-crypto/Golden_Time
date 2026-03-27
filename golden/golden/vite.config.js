import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // 개발 시 localhost:5173 에서 /api 가 Spring(1111)으로 가도록 — 상대 경로 fetch 가 동작함
      '/api': {
        target: 'http://localhost:1111',
        changeOrigin: true,
      },
    },
  },
})
