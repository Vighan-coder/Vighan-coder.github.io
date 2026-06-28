"use client";

import React, { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Define constant particle random offsets at module level to guarantee render purity
const STATIC_PARTICLES = Array.from({ length: 20 }, (_, i) => ({
  xSeed: Math.abs(Math.sin(i * 1234.56)), // pseudo-random deterministic seeds
  ySeed: Math.abs(Math.cos(i * 4567.89)),
  speedSeed: Math.abs(Math.sin(i * 9876.54)),
  delay: Math.abs(Math.cos(i * 3456.78)) * 1.5,
  swayIndex: i
}));

interface LoadingScreenProps {
  onComplete?: () => void;
}

export default function LoadingScreen({ onComplete }: LoadingScreenProps) {
  const [percent, setPercent] = useState(0);

  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  // Initialize to server-safe defaults to prevent hydration mismatch
  const [isSkipped, setIsSkipped] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  const [isMounted, setIsMounted] = useState(false);

  const [dimensions, setDimensions] = useState({ width: 1000, height: 800 });

  // Update dimensions in an async animation frame
  useEffect(() => {
    const handleResize = () => {
      requestAnimationFrame(() => {
        setDimensions({
          width: window.innerWidth,
          height: window.innerHeight
        });
      });
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    let visited = false;
    try {
      visited = localStorage.getItem("vighan-portfolio-visited") === "true";
    } catch (e) {
      console.warn("localStorage get visited error:", e);
    }

    console.log("LoadingScreen: Client mount check. visited =", visited);
    let timeoutId: NodeJS.Timeout | undefined;
    let frameId: number | undefined;

    if (visited) {
      timeoutId = setTimeout(() => {
        setIsMounted(true);
        setIsSkipped(true);
        setIsVisible(false);
        if (onCompleteRef.current) onCompleteRef.current();
      }, 0);
    } else {
      timeoutId = setTimeout(() => {
        setIsMounted(true);
      }, 0);

      // Accumulate percentage over 2.5 seconds
      const startTime = Date.now();
      const duration = 2400; // 2.4 seconds loading duration

      const updateLoader = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Add a nice ease to the percentage counting
        const easedProgress = 1 - Math.pow(1 - progress, 3);
        const nextPercent = Math.floor(easedProgress * 100);
        console.log("LoadingScreen: updateLoader loop tick. progress =", progress, "percent =", nextPercent);
        setPercent(nextPercent);

        if (progress < 1) {
          frameId = requestAnimationFrame(updateLoader);
        } else {
          // Complete loading
          console.log("LoadingScreen: loader finished, transitioning out...");
          timeoutId = setTimeout(() => {
            setIsVisible(false);
            try {
              localStorage.setItem("vighan-portfolio-visited", "true");
            } catch (e) {
              console.warn("localStorage set visited error:", e);
            }
            if (onCompleteRef.current) onCompleteRef.current();
          }, 500); // Small pause at 100%
        }
      };

      frameId = requestAnimationFrame(updateLoader);
    }

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, []);

  if (isSkipped || !isVisible) return null;

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          key="loader"
          initial={{ opacity: 1 }}
          exit={{ 
            opacity: 0, 
            scale: 1.1,
            transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } 
          }}
          className="fixed inset-0 bg-[#050505] z-[9999] flex flex-col items-center justify-center overflow-hidden"
        >
          {/* Volumetric background glowing emerald radial light */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-[#00D084]/10 rounded-full blur-[120px] pointer-events-none" />

          {/* Floating emerald particles */}
          <div className="absolute inset-0 pointer-events-none">
            {isMounted && STATIC_PARTICLES.map((p, i) => (
              <motion.div
                key={i}
                initial={{ 
                  opacity: 0, 
                  x: p.xSeed * dimensions.width, 
                  y: dimensions.height + 50 
                }}
                animate={{ 
                  opacity: [0, 0.7, 0], 
                  y: -50,
                  x: `calc(${p.xSeed * dimensions.width}px + ${Math.sin(p.swayIndex) * 50}px)`
                }}
                transition={{ 
                  duration: 2.5 + p.speedSeed * 1.5,
                  repeat: Infinity,
                  delay: p.delay,
                  ease: "linear"
                }}
                className="absolute w-[4px] h-[4px] rounded-full bg-[#00D084]"
              />
            ))}
          </div>

          <div className="flex flex-col items-center z-10 space-y-6">
            {/* Animated Logo (VRV stylized letters) */}
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
              className="relative w-24 h-24 flex items-center justify-center"
            >
              {/* Outer pulsing ring */}
              <motion.div 
                animate={{ rotate: 360 }}
                transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                className="absolute inset-0 border border-dashed border-[#00D084]/40 rounded-full"
              />
              
              {/* Outer solid glow ring */}
              <motion.div 
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                className="absolute inset-2 border-2 border-[#00D084] rounded-full shadow-[0_0_20px_rgba(0,208,132,0.4)]"
              />

              {/* Logo SVG Text representation */}
              <svg className="w-10 h-10 text-[#F3F4F6]" viewBox="0 0 100 100" fill="none">
                <motion.path
                  d="M20 30 L50 75 L80 30"
                  stroke="currentColor"
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1.5, delay: 0.3, ease: "easeInOut" }}
                />
                <motion.path
                  d="M35 30 L50 55 L65 30"
                  stroke="#00D084"
                  strokeWidth="6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1.2, delay: 0.6, ease: "easeInOut" }}
                />
              </svg>
            </motion.div>

            {/* Title / Name */}
            <div className="text-center space-y-1">
              <motion.h1
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.5 }}
                className="text-lg font-semibold tracking-widest text-[#F3F4F6] uppercase"
              >
                Vighan Raj Verma
              </motion.h1>
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.6 }}
                transition={{ duration: 0.8, delay: 0.8 }}
                className="text-xs font-mono tracking-wider text-[#9CA3AF]"
              >
                Curiosity to Intelligence
              </motion.p>
            </div>

            {/* Progress Percentage Counter */}
            <div className="w-48 flex flex-col items-center space-y-2">
              <span className="text-sm font-mono font-medium text-[#00D084]">
                {percent}%
              </span>
              {/* Progress bar background */}
              <div className="w-full h-[2px] bg-[#0F1115] rounded-full overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${percent}%` }}
                  transition={{ duration: 0.1 }}
                  className="h-full bg-gradient-to-r from-[#00D084] to-[#4AFFB8] shadow-[0_0_8px_#00D084]"
                />
              </div>
            </div>
          </div>

          {/* Prompt to skip if user clicks */}
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.4 }}
            whileHover={{ opacity: 0.8 }}
            onClick={() => {
              setIsVisible(false);
              try {
                localStorage.setItem("vighan-portfolio-visited", "true");
              } catch (e) {
                console.warn("localStorage set visited error:", e);
              }
              if (onCompleteRef.current) onCompleteRef.current();
            }}
            className="absolute bottom-8 text-[10px] font-mono tracking-widest text-[#9CA3AF] uppercase hover:text-[#00D084] transition-colors"
          >
            Skip Intro
          </motion.button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
