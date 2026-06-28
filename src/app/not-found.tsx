"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { Home, AlertTriangle } from "lucide-react";

// Glass shard positions and vectors for dispersion
const SHARDS = [
  { clipPath: "polygon(50% 0%, 0% 100%, 100% 100%)", size: "w-20 h-20", x: -80, y: -60, r: -35 },
  { clipPath: "polygon(0% 0%, 100% 0%, 50% 100%)", size: "w-24 h-24", x: 90, y: -80, r: 45 },
  { clipPath: "polygon(0% 20%, 80% 0%, 100% 100%)", size: "w-16 h-28", x: -140, y: 30, r: -20 },
  { clipPath: "polygon(20% 0%, 100% 20%, 0% 100%)", size: "w-28 h-16", x: 130, y: 60, r: 60 },
  { clipPath: "polygon(50% 50%, 0% 100%, 100% 100%)", size: "w-20 h-20", x: -40, y: 120, r: -15 },
  { clipPath: "polygon(0% 0%, 100% 100%, 0% 100%)", size: "w-16 h-16", x: 60, y: 140, r: 25 },
  { clipPath: "polygon(100% 0%, 100% 100%, 0% 50%)", size: "w-24 h-20", x: -180, y: -120, r: -50 },
  { clipPath: "polygon(50% 0%, 100% 100%, 0% 50%)", size: "w-20 h-24", x: 190, y: -130, r: 55 }
];
// Deterministic pseudo-random coordinates for rendering purity
const STATIC_NOT_FOUND_PARTICLES = Array.from({ length: 12 }, (_, i) => ({
  top: `${20 + Math.abs(Math.sin(i * 123.45)) * 60}%`,
  left: `${20 + Math.abs(Math.cos(i * 678.90)) * 60}%`
}));

export default function NotFound() {
  const { playClick, playHover } = useAudio();
  const [shattered, setShattered] = useState(false);

  // Trigger shattering effect shortly after mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setShattered(true);
    }, 600);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="max-w-xl mx-auto w-full flex-grow flex flex-col justify-center items-center relative z-10 pt-12 select-none text-center">
      
      {/* 3D Shattered Glass Arena */}
      <div 
        onClick={() => {
          playClick();
          setShattered(!shattered);
        }}
        className="relative w-72 h-72 flex items-center justify-center cursor-pointer mb-12"
      >
        {/* Floating background particles */}
        <div className="absolute inset-0 pointer-events-none opacity-40">
          {STATIC_NOT_FOUND_PARTICLES.map((p, i) => (
            <motion.div
              key={i}
              animate={{ 
                y: [-20, 20, -20],
                x: [-10, 10, -10],
                scale: [0.8, 1.2, 0.8]
              }}
              transition={{ 
                duration: 3 + i, 
                repeat: Infinity, 
                ease: "easeInOut" 
              }}
              className="absolute w-1.5 h-1.5 rounded-full bg-accent"
              style={{
                top: p.top,
                left: p.left
              }}
            />
          ))}
        </div>

        {/* Center error text indicator */}
        <div className="z-10 flex flex-col items-center space-y-2 pointer-events-none">
          <AlertTriangle className="w-8 h-8 text-accent animate-pulse" />
          <h2 className="text-3xl font-sans font-black tracking-tight text-white uppercase">
            404
          </h2>
          <span className="text-[9px] font-mono tracking-widest text-[#9CA3AF] uppercase">
            PATHWAY SEVERED
          </span>
        </div>

        {/* Glass Shards */}
        {SHARDS.map((shard, idx) => (
          <motion.div
            key={idx}
            initial={{ x: 0, y: 0, rotate: 0, opacity: 0.85, scale: 1 }}
            animate={shattered ? {
              x: shard.x,
              y: shard.y,
              rotate: shard.r,
              opacity: 0.25,
              scale: 0.9
            } : {
              x: 0,
              y: 0,
              rotate: 0,
              opacity: 0.85,
              scale: 1
            }}
            transition={{ 
              type: "spring", 
              stiffness: 80, 
              damping: 12, 
              mass: 0.8,
              delay: idx * 0.02 
            }}
            className={`absolute bg-white/5 border border-white/20 backdrop-blur-md shadow-2xl pointer-events-none ${shard.size}`}
            style={{ 
              clipPath: shard.clipPath,
              transformOrigin: "center center"
            }}
          />
        ))}
      </div>

      {/* Narrative details */}
      <div className="space-y-6">
        <p className="text-xs text-gray-400 max-w-sm mx-auto leading-relaxed">
          The requested coordinate pathway does not connect to any active page arrays. The logic grid has collapsed.
        </p>

        <Link href="/" onClick={playClick} onMouseEnter={playHover}>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="w-48 py-3.5 text-xs font-mono font-bold tracking-widest text-black bg-[#00D084] hover:bg-[#4AFFB8] rounded-xl hover:shadow-[0_0_20px_rgba(0,208,132,0.4)] transition-all cursor-pointer flex items-center justify-center space-x-1.5 mx-auto"
          >
            <Home className="w-3.5 h-3.5" />
            <span>RETURN HOME</span>
          </motion.button>
        </Link>
      </div>

    </div>
  );
}
