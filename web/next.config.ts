import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits a self-contained server with only the files it actually imports, so
  // the runtime image carries no build toolchain and no unused dependency tree.
  output: "standalone",

  images: {
    // Company logos are served from the provider's own CDN. Listed explicitly
    // rather than wildcarded: next/image will render any host named here, and a
    // wildcard would turn the app into an open image proxy.
    remotePatterns: [
      { protocol: "https", hostname: "static2.finnhub.io" },
      { protocol: "https", hostname: "static.finnhub.io" },
    ],
  },
};

export default nextConfig;
