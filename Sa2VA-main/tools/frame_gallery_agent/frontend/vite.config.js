import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendTarget = process.env.FRAME_GALLERY_BACKEND || "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": backendTarget,
      "/frame": backendTarget,
      "/mask": backendTarget,
      "/download": backendTarget,
    },
  },
});
