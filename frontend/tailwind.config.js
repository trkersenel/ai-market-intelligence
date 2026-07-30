/**
 * Design tokens from the validated data-viz palette.
 *
 * The chart colours are not decorative choices. `up`/`down` are blue and red,
 * not the conventional green and red: green/red is precisely the pair that
 * red-green colour blindness collapses, which makes the finance convention the
 * single worst option for the one encoding a trading screen most depends on.
 * Blue↔red reads as opposite to everyone.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        surface: { DEFAULT: "#1a1a19", page: "#0d0d0d", raised: "#222221" },
        ink: { DEFAULT: "#ffffff", secondary: "#c3c2b7", muted: "#898781" },
        hairline: { DEFAULT: "#2c2c2a", strong: "#383835" },
        series: { 1: "#3987e5", 2: "#d95926", 3: "#199e70" },
        // Diverging poles for returns, plus the neutral midpoint.
        up: "#3987e5",
        down: "#d03b3b",
        flat: "#383835",
        status: {
          good: "#0ca30c",
          warning: "#fab219",
          serious: "#ec835a",
          critical: "#d03b3b",
        },
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      fontSize: { "2xs": ["0.6875rem", { lineHeight: "1rem" }] },
    },
  },
  plugins: [],
};
