import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // The API is proxied rather than called cross-origin. Same-origin requests
    // need no CORS preflight and no absolute URL in the client, so the same
    // build works in dev, in Docker and behind a reverse proxy.
    proxy: {
      "/api": { target: process.env.VITE_API_TARGET ?? "http://localhost:8000", changeOrigin: true },
      "/health": { target: process.env.VITE_API_TARGET ?? "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        // Recharts and its d3 dependencies are most of the bundle and change
        // far less often than application code. Splitting them out means a
        // deploy invalidates the app chunk without forcing every returning
        // visitor to re-download the charting library.
        manualChunks: {
          charts: ["recharts"],
          vendor: ["react", "react-dom", "react-router-dom", "@tanstack/react-query"],
        },
      },
    },
  },
});
