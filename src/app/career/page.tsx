"use client";

import React, { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { 
  Briefcase, 
  GraduationCap, 
  Sparkles, 
  Calendar 
} from "lucide-react";

export default function CareerPage() {
  const { playClick, playHover } = useAudio();
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Track scroll depth in this career section to trigger camera adjustments and glows
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  // Example scroll-linked color transforms for ambient background glowing
  const glowOpacity = useTransform(scrollYProgress, [0.1, 0.4, 0.7, 1.0], [0, 0.8, 0.3, 0]);

  return (
    <div ref={containerRef} className="max-w-4xl mx-auto w-full space-y-24 relative z-10 pt-8">
      
      {/* Scroll-Linked Ambient Glow behind the timeline */}
      <motion.div 
        style={{ opacity: glowOpacity }}
        className="fixed inset-0 bg-radial from-[#00D084]/5 via-transparent to-transparent pointer-events-none -z-20 transition-all duration-700"
      />

      <div className="text-center space-y-4">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          className="text-xs font-mono tracking-widest text-[#9CA3AF] uppercase"
        >
          Timeline
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl sm:text-5xl font-sans font-black tracking-tight text-white uppercase"
        >
          Career & <span className="text-accent">Milestones</span>
        </motion.h1>
      </div>

      <div className="relative border-l-2 border-white/5 pl-8 sm:pl-12 max-w-2xl mx-auto space-y-16">
        
        {/* ENTRY 1: B.Tech Computer Science */}
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8 }}
          className="relative group"
        >
          {/* Node Icon indicator */}
          <div className="absolute -left-[50px] sm:-left-[66px] top-1.5 w-9 h-9 rounded-xl bg-[#050505] border border-white/10 flex items-center justify-center group-hover:border-accent/40 transition-colors">
            <GraduationCap className="w-4 h-4 text-[#9CA3AF] group-hover:text-accent transition-colors" />
          </div>

          <div className="glass-panel p-6 space-y-3 hover:border-white/20 transition-all">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
              <div>
                <span className="text-[10px] font-mono font-bold text-accent tracking-widest uppercase">
                  ACADEMIC FOUNDATION
                </span>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  B.Tech Computer Science & Engineering
                </h3>
              </div>
              <span className="text-[10px] font-mono text-[#9CA3AF] flex items-center gap-1.5 shrink-0">
                <Calendar className="w-3.5 h-3.5" />
                2024 - 2028 (Expected)
              </span>
            </div>

            <p className="text-xs text-accent/80 font-mono">
              Truba Institute of Engineering & Information Technology, Bhopal
            </p>
            <p className="text-xs text-gray-400 leading-relaxed">
              Acquiring principles of algorithms, complexity bounds, compiler architectures, and discrete mathematics. Actively translating theory into practice through code bases, system configurations, and statistics toolkits.
            </p>
          </div>
        </motion.div>

        {/* ENTRY 2: R&D Intern @ IISER Bhopal (Special Cinematic Treatment) */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 1.0, type: "spring", stiffness: 100 }}
          className="relative group"
        >
          {/* Active node glowing indicator */}
          <div className="absolute -left-[50px] sm:-left-[66px] top-1.5 w-9 h-9 rounded-xl bg-black border-2 border-accent flex items-center justify-center shadow-[0_0_15px_rgba(0,208,132,0.4)]">
            <Briefcase className="w-4 h-4 text-accent animate-pulse" />
          </div>

          {/* Highlighted Card Container */}
          <div 
            onClick={playClick}
            onMouseEnter={playHover}
            className="glass-panel p-8 space-y-4 border-accent bg-[#00D084]/5 hover:bg-[#00D084]/8 hover:shadow-[0_0_30px_rgba(0,208,132,0.15)] transition-all cursor-pointer relative overflow-hidden"
          >
            {/* Sparkles / volumetric flare absolute top corner */}
            <div className="absolute -top-12 -right-12 w-28 h-28 bg-accent/20 rounded-full blur-2xl pointer-events-none" />

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
              <div>
                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[9px] font-mono font-bold bg-accent/20 text-accent border border-accent/30 tracking-widest uppercase">
                  <Sparkles className="w-2.5 h-2.5" />
                  CINEMATIC ACTIVE INTERNSHIP
                </span>
                <h3 className="text-xl font-extrabold text-white tracking-tight mt-1.5">
                  Research & Development Intern
                </h3>
              </div>
              <span className="text-[10px] font-mono text-accent flex items-center gap-1.5 shrink-0">
                <Calendar className="w-3.5 h-3.5" />
                24 May - 21 July
              </span>
            </div>

            <p className="text-xs text-white font-mono">
              Indian Institute of Science Education and Research (IISER), Bhopal
            </p>

            <div className="text-xs text-gray-300 space-y-2 leading-relaxed">
              <p>
                Contributing to a research paper focused on fusing independently trained aerial and street-view Gaussian splats into a unified scene. Involved in raw data collection, editing data to convert to the correct bit depth, sampling sequences into video frames, and performing multi-view structure-from-motion reconstruction via COLMAP. Trained 3DGS models, solved coordinate scaling issues, and executed scene fusion.
              </p>
              <p className="text-[11px] text-accent/90 font-mono">
                Acquired hands-on experience in Linux environments (terminal commands) and Git/GitHub version control workflows.
              </p>
              <div className="pt-2 flex flex-wrap gap-1.5">
                {["Gaussian Splats (3DGS)", "COLMAP", "Linux Terminal", "Git / GitHub", "Data Calibration", "SfM Reconstruction"].map((chip) => (
                  <span key={chip} className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[9px] font-mono text-gray-400">
                    {chip}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

      </div>

    </div>
  );
}
