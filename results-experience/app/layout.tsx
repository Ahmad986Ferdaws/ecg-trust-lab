import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://ecg-trust-results-motion.byw-123.chatgpt.site"),
  title: "ECG Trust Lab — The Signal Ledger",
  description:
    "An editorial account of sealed PTB-XL discrimination, frozen SPH transport, and an honestly preserved source-support gate result.",
  openGraph: {
    title: "ECG Trust Lab — The Signal Ledger",
    description:
      "Evidence before confidence: sealed model comparison, no-adaptation transport, and a one-shot support gate.",
    type: "website",
    images: [
      {
        url: "/og.png",
        width: 1729,
        height: 910,
        alt: "ECG Trust Lab — Evidence before confidence.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "ECG Trust Lab — The Signal Ledger",
    description:
      "Audited PTB-XL, SPH transport, and source-support evidence presented as an interactive signal ledger.",
    images: [
      {
        url: "/og.png",
        alt: "ECG Trust Lab — Evidence before confidence.",
      },
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="data:," />
      </head>
      <body>{children}</body>
    </html>
  );
}
