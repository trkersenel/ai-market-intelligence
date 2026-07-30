import type { NextConfig } from "next";

const nextConfig: NextConfig = {
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
