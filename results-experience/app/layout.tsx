import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const title = "ECG Trust Lab — Results in Motion";
const description =
  "An immersive, audited exploration of PTB-XL ECG classification and independent SPH transport results.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = (
    requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host")
  )
    ?.split(",", 1)[0]
    .trim();
  const forwardedProtocol = requestHeaders
    .get("x-forwarded-proto")
    ?.split(",", 1)[0]
    .trim();
  const protocol =
    forwardedProtocol === "http" || forwardedProtocol === "https"
      ? forwardedProtocol
      : host?.startsWith("localhost") || host?.startsWith("127.0.0.1")
        ? "http"
        : "https";
  const metadataBase = new URL(`${protocol}://${host ?? "localhost:3000"}`);
  const socialImage = new URL("/og.png", metadataBase).toString();

  return {
    metadataBase,
    title,
    description,
    openGraph: {
      title,
      description:
        "Explore audited ECG model evidence as an immersive 3D data universe.",
      type: "website",
      images: [
        {
          url: socialImage,
          width: 1672,
          height: 941,
          alt: "Luminous ECG signals orbiting two translucent model monoliths",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description:
        "Audited PTB-XL and SPH results, rendered as an immersive 3D experience.",
      images: [socialImage],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
