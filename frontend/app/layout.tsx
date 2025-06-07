import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'NeuralNexus - Neurological Drug Discovery Platform',
  description:
    'AI-powered platform for neurological drug discovery combining protein analysis, knowledge graphs, and hypothesis generation.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div id="app">{children}</div>
      </body>
    </html>
  );
}
