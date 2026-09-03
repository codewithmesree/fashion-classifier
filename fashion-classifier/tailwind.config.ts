import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#E4DCC8", // raw muslin fabric
        ink: "#201C16", // screen-print ink
        denim: {
          DEFAULT: "#2E4A73",
          deep: "#17273E",
          light: "#5A7BA6",
        },
        thread: "#B23A2E", // stitching thread, used sparingly
        paper: "#FBF8F2",
        muted: "#8A8171",
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        body: ["var(--font-body)", "sans-serif"],
        tag: ["var(--font-tag)", "monospace"],
      },
      letterSpacing: {
        tagwide: "0.14em",
      },
      boxShadow: {
        swatch: "0 1px 0 rgba(32,28,22,0.06), 0 8px 24px -12px rgba(32,28,22,0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
