import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Forward API calls to the FastAPI backend so the frontend can just
    // fetch('/ingest') and fetch('/query') with no CORS hassle.
    // Use 127.0.0.1, not localhost: on Windows, Vite may resolve localhost
    // to IPv6 (::1) while uvicorn only listens on IPv4, causing ECONNREFUSED.
    proxy: {
      '/ingest': 'http://127.0.0.1:8000',
      '/query': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
