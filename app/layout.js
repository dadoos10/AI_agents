export const metadata = {
  title: "Finance Hub · מניות והוצאות Max",
  description: "מעקב מניות חי וייבוא פירוט חיובים מ־Max",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    title: "Finance Hub",
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
