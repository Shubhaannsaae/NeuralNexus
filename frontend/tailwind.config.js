/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#21808D',
          50: '#f0f9fa',
          100: '#daf1f3',
          200: '#b8e3e8',
          300: '#8bd0d8',
          400: '#57b4c1',
          500: '#21808D',
          600: '#1e7380',
          700: '#1d606a',
          800: '#1e4f57',
          900: '#1c424a',
        },
        secondary: {
          DEFAULT: '#50b8c6',
          50: '#f1fbfc',
          100: '#ddf4f7',
          200: '#c0eaf0',
          300: '#94dae4',
          400: '#50b8c6',
          500: '#3ca3b3',
          600: '#358596',
          700: '#316d7b',
          800: '#305a66',
          900: '#2c4c57',
        },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
