"use client";

import React, { useState } from "react";
import LoadingScreen from "@/components/LoadingScreen";
import { RefreshCw } from "lucide-react";
import { useAudio } from "@/context/AudioContext";
import { motion } from "framer-motion";

export default function LoadingPageShowcase() {
  const { playClick } = useAudio();
  const [key, setKey] = useState(0);

  const forceReplay = () => {
    playClick();
    try {
      localStorage.removeItem("vighan-portfolio-visited");
    } catch (e) {
      console.warn("localStorage remove visited error:", e);
    }
    setKey((prev) => prev + 1);
  };

  return (
    <div className="max-w-md mx-auto w-full flex-grow flex flex-col justify-center items-center relative z-10 pt-16 select-none text-center">
      
      {/* Loading Showcase Trigger */}
      <LoadingScreen key={key} />

      <div className="glass-panel p-8 space-y-6 w-full">
        <h2 className="text-sm font-mono font-bold text-white tracking-widest uppercase">
          LAUNCH SEQUENCE CONTROLLER
        </h2>
        
        <p className="text-xs text-gray-400 leading-relaxed">
          The cinematic entry animation is configured to play only once on the user&apos;s initial visit to optimize repeat routing times. To see the animation, click below.
        </p>

        <div className="flex flex-col gap-2 pt-2">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={forceReplay}
            className="w-full py-3.5 text-xs font-mono font-bold tracking-widest text-black bg-[#00D084] hover:bg-[#4AFFB8] rounded-xl transition-all cursor-pointer flex items-center justify-center space-x-2"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>FORCE REPLAY LOADER</span>
          </motion.button>
        </div>
      </div>

    </div>
  );
}
