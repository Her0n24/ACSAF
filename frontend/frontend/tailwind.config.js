/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        sunset: {
          50: "#fff5f1",
          100: "#ffe1d2",
          200: "#ffc2a6",
          300: "#ff9a6e",
          400: "#ff7a45",
          500: "#ff5c18",
          600: "#f3490c",
          700: "#c43610",
          800: "#9a2d16",
          900: "#7c2817"
        }
      }
    }
  },
  plugins: [],
};
