export const metadata = {
  title: "MY LIST · Live Watchlist",
  description: "Live Yahoo Finance watchlist for phone and desktop",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    title: "MY LIST",
    statusBarStyle: "black-translucent",
  },
  themeColor: "#0b0f14",
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 1,
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="he" dir="rtl">
      <body>{children}</body>
    </html>
  );
}
