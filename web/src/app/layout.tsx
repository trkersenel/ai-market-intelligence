import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { AppShell } from "@/components/shell/app-shell";
import { Providers } from "./providers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Market Intelligence",
  description:
    "Anomaly detection, news correlation and grounded answers across the AI infrastructure and semiconductor ecosystem.",
};

/**
 * Applies the stored theme before first paint.
 *
 * A blocking inline script rather than an effect, because an effect runs after
 * hydration -- which means a light-theme user sees a dark flash on every page
 * load. Defaults to dark, matching the stylesheet.
 */
const THEME_SCRIPT = `
try {
  var stored = localStorage.getItem("theme");
  document.documentElement.dataset.theme = stored === "light" ? "light" : "dark";
} catch (e) {
  document.documentElement.dataset.theme = "dark";
}
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      data-theme="dark"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
