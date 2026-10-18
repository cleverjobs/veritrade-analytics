import type { NextConfig } from "next";

// Static export: FastAPI serves `out/` from the same origin as /api, so no Node server runs in production.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
