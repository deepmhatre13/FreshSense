import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// SmartFreshAI frontend — Vite + Vitest configuration.
// The backend is a separate FastAPI process; the frontend targets it via
// VITE_API_BASE_URL (see frontend/.env.example). No proxy coupling required.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    cors: {
      origin: false,
    },
  },
  build: {
    target: "es2022",
  },
  test: {
    environment: "jsdom",
    globals: false,
    css: false,
    setupFiles: ["./src/testUtils/setup.js"],
    // Backend URL is optional in tests; inferenceApi reads it at call time.
  },
});
