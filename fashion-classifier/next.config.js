/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Bundle everything needed to run without node_modules at start time.
  // Render starts the app with: node .next/standalone/server.js
  output: "standalone",

  webpack(config, { isServer }) {
    // @tensorflow/tfjs-node is a native addon used only by training scripts.
    // Prevent it from being pulled into the Next.js bundle.
    config.externals = [
      ...(config.externals || []),
      "@tensorflow/tfjs-node",
    ];
    return config;
  },
};

module.exports = nextConfig;
