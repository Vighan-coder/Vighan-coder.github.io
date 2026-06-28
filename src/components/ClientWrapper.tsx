"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { AudioProvider, useAudio } from "@/context/AudioContext";
import Navbar from "./Navbar";
import LoadingScreen from "./LoadingScreen";
import Lenis from "lenis";

// Lazy load ThreeCanvas to prevent SSR errors and optimize load time
const ThreeCanvas = dynamic(() => import("./ThreeCanvas"), {
  ssr: false,
  loading: () => <div className="fixed inset-0 w-full h-full bg-[#050505] -z-10" />
});

function AppContent({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true);
  const pathname = usePathname();
  const { playTransition } = useAudio();

  // Initialize Lenis smooth scrolling
  useEffect(() => {
    if (isLoading) return;

    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: "vertical",
      gestureOrientation: "vertical",
      smoothWheel: true,
      wheelMultiplier: 1.0,
      touchMultiplier: 1.5,
    });

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    return () => {
      lenis.destroy();
    };
  }, [isLoading]);

  // Play transition sweep sound on route changes
  useEffect(() => {
    if (!isLoading) {
      playTransition();
    }
  }, [pathname, isLoading, playTransition]);

  return (
    <>
      <LoadingScreen onComplete={() => setIsLoading(false)} />
      
      {!isLoading && (
        <>
          <ThreeCanvas />
          <Navbar />
          
          {/* Main page transition wrapper */}
          <AnimatePresence mode="popLayout">
            <motion.main
              key={pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className="relative w-full min-h-screen pt-24 px-4 sm:px-6 lg:px-8 pb-16 flex flex-col"
            >
              {children}
            </motion.main>
          </AnimatePresence>

          {/* Simple Cinematic Footer */}
          <footer className="w-full text-center py-8 border-t border-white/5 bg-black/40 text-xs font-mono text-[#9CA3AF] tracking-widest mt-auto">
            <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between space-y-2 sm:space-y-0">
              <p>&copy; {new Date().getFullYear()} Vighan Raj Verma. All Rights Reserved.</p>
              <p className="hover:text-accent transition-colors">
                <Link href="/privacy">Privacy Policy</Link>
              </p>
            </div>
          </footer>
        </>
      )}
    </>
  );
}

export default function ClientWrapper({ children }: { children: React.ReactNode }) {
  return (
    <AudioProvider>
      <AppContent>{children}</AppContent>
    </AudioProvider>
  );
}
