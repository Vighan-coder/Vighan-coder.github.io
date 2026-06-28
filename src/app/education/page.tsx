"use client";

import React from "react";
import { motion } from "framer-motion";
import { GraduationCap } from "lucide-react";

export default function EducationPage() {
  return (
    <div className="max-w-5xl mx-auto w-full space-y-20 relative z-10 pt-8">
      
      {/* Page Title */}
      <div className="text-center space-y-4">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          className="text-xs font-mono tracking-widest text-[#9CA3AF] uppercase"
        >
          Academics
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl sm:text-5xl font-sans font-black tracking-tight text-white uppercase"
        >
          Education & <span className="text-accent">Foundation</span>
        </motion.h1>
      </div>

      {/* Secondary Education Block (10th Grade) */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="glass-panel p-8 flex flex-col md:flex-row md:items-center justify-between gap-6"
      >
        <div className="space-y-4">
          <div className="flex items-center space-x-2 text-accent">
            <GraduationCap className="w-5 h-5" />
            <span className="text-xs font-mono font-bold tracking-widest uppercase">Secondary Schooling</span>
          </div>
          <div>
            <h2 className="text-2xl font-black text-white tracking-tight leading-none uppercase">
              Model Higher Secondary School
            </h2>
            <p className="text-xs font-mono text-accent mt-2">
              Bhopal, Madhya Pradesh, India
            </p>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed max-w-2xl">
            Completed Secondary Schooling (10th Grade) with First Division, establishing computational interest and choosing Physics, Chemistry, and Mathematics (PCM) as the primary streams for senior secondary study.
          </p>
        </div>
        <div className="shrink-0 flex flex-col items-center justify-center p-4 bg-white/5 border border-white/10 rounded-2xl text-center min-w-44">
          <span className="text-[10px] font-mono text-[#9CA3AF] uppercase tracking-wider">COMPLETED IN</span>
          <span className="text-xl font-mono font-black text-white mt-1">2022</span>
          <span className="inline-flex items-center gap-1 mt-2 px-2.5 py-0.5 rounded-full text-[9px] font-mono bg-accent/20 text-accent font-bold">
            First Division (10th Grade)
          </span>
        </div>
      </motion.div>

      {/* Schooling Block (12th Grade) */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
        className="glass-panel p-8 flex flex-col md:flex-row md:items-center justify-between gap-6"
      >
        <div className="space-y-4">
          <div className="flex items-center space-x-2 text-accent">
            <GraduationCap className="w-5 h-5" />
            <span className="text-xs font-mono font-bold tracking-widest uppercase">Higher Secondary Schooling</span>
          </div>
          <div>
            <h2 className="text-2xl font-black text-white tracking-tight leading-none uppercase">
              Model Higher Secondary School
            </h2>
            <p className="text-xs font-mono text-accent mt-2">
              Bhopal, Madhya Pradesh, India
            </p>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed max-w-2xl">
            Completed High School (12th Grade) under the MP Board, focusing on Physics, Chemistry, and Mathematics (PCM). Achieved strong analytical skills and academic excellence, laying the groundwork for undergraduate studies in Computer Science.
          </p>
        </div>
        <div className="shrink-0 flex flex-col items-center justify-center p-4 bg-white/5 border border-white/10 rounded-2xl text-center min-w-44">
          <span className="text-[10px] font-mono text-[#9CA3AF] uppercase tracking-wider">COMPLETED IN</span>
          <span className="text-xl font-mono font-black text-white mt-1">2024</span>
          <span className="inline-flex items-center gap-1 mt-2 px-2.5 py-0.5 rounded-full text-[9px] font-mono bg-accent/20 text-accent font-bold">
            First Division (12th Grade)
          </span>
        </div>
      </motion.div>

      {/* Main Degree Block */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        className="glass-panel p-8 border-accent/20 bg-accent/2 flex flex-col md:flex-row md:items-center justify-between gap-6"
      >
        <div className="space-y-4">
          <div className="flex items-center space-x-2 text-accent">
            <GraduationCap className="w-5 h-5 animate-pulse" />
            <span className="text-xs font-mono font-bold tracking-widest uppercase">DEGREE PROGRAM</span>
          </div>
          <div>
            <h2 className="text-2xl font-black text-white tracking-tight leading-none uppercase">
              B.Tech Computer Science & Engineering
            </h2>
            <p className="text-xs font-mono text-accent mt-2">
              Truba Institute of Engineering & Information Technology, Bhopal
            </p>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed max-w-2xl">
            Focusing on computation limits, algorithm optimization, software engineering, databases, and scientific computing models. Building a mathematical and programmatic foundation to solve data-centric problems in machine learning and computer vision.
          </p>
        </div>
        <div className="shrink-0 flex flex-col items-center justify-center p-4 bg-white/5 border border-white/10 rounded-2xl text-center min-w-44">
          <span className="text-[10px] font-mono text-[#9CA3AF] uppercase tracking-wider">PROGRAM DURATION</span>
          <span className="text-xl font-mono font-black text-white mt-1">2024 - 2028</span>
          <span className="inline-flex items-center gap-1 mt-2 px-2.5 py-0.5 rounded-full text-[9px] font-mono bg-accent/20 text-accent font-bold">
            2nd Year CSE Student
          </span>
        </div>
      </motion.div>

    </div>
  );
}
