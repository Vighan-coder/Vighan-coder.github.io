import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import ClientWrapper from "@/components/ClientWrapper";
import Analytics from "@/components/Analytics";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Vighan Raj Verma | Data Scientist & R&D Intern",
  description: "Portfolio of Vighan Raj Verma, B.Tech Computer Science student at Truba Institute and R&D Intern at IISER Bhopal. Specialized in Data Science, Computer Vision, and Machine Learning.",
  keywords: [
    "Vighan Raj Verma", 
    "Data Scientist Portfolio", 
    "IISER Bhopal Intern", 
    "Computer Science Bhopal", 
    "Machine Learning Student", 
    "Gaussian Splatting", 
    "Computer Vision Developer"
  ],
  authors: [{ name: "Vighan Raj Verma" }],
  creator: "Vighan Raj Verma",
  metadataBase: new URL("https://vighan.vercel.app"), // Fallback domain for deployment
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://vighan.vercel.app",
    title: "Vighan Raj Verma | Data Science & Research Portfolio",
    description: "Explore the computer science research and machine learning engineering work of Vighan Raj Verma, R&D Intern at IISER Bhopal.",
    siteName: "Vighan Raj Verma Portfolio",
  },
  twitter: {
    card: "summary_large_image",
    title: "Vighan Raj Verma | Data Science Portfolio",
    description: "R&D Intern at IISER Bhopal. Aspiring Data Scientist exploring Machine Learning and Computer Vision.",
  },
  alternates: {
    canonical: "/",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased`}
      style={{ scrollBehavior: "auto" }}
    >
      <body className="min-h-full flex flex-col selection:bg-[#00D084] selection:text-black">
        <ClientWrapper>{children}</ClientWrapper>
        <Analytics />
      </body>
    </html>
  );
}
