import type { Metadata } from "next";
import { Space_Grotesk, Inter, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const display = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "700"],
  variable: "--font-display",
});

const body = Inter({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-body",
});

const tag = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-tag",
});

export const metadata: Metadata = {
  title: "Garment Scanner — Fashion-MNIST Classifier",
  description:
    "Draw or upload a clothing sketch and a CNN reads it like a garment tag.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${tag.variable}`}>
      <body className="font-body text-ink antialiased">{children}</body>
    </html>
  );
}
