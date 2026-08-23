import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ECG Trust Lab — The Signal Ledger",
  description:
    "An editorial, interactive account of sealed PTB-XL discrimination and frozen SPH transport evidence.",
  openGraph: {
    title: "ECG Trust Lab — The Signal Ledger",
    description:
      "Two architectures, one sealed test, and one no-adaptation transport stress test.",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: "ECG Trust Lab — The Signal Ledger",
    description:
      "Audited PTB-XL and SPH model evidence presented as an interactive signal ledger.",
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
