"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { 
  Mail, 
  ArrowRight, 
  GraduationCap, 
  Microscope, 
  Target, 
  ChevronDown 
} from "lucide-react";

const GithubIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

const LinkedinIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect width="4" height="12" x="2" y="9" />
    <circle cx="4" cy="4" r="2" />
  </svg>
);

// Subtitle texts for cycling
const SUBTITLES = [
  "Computer Science Student",
  "Aspiring Data Scientist",
  "AI Enthusiast",
  "Computer Vision Explorer",
  "R&D Intern",
  "Future Machine Learning Engineer"
];

// Viewport-aware count-up counter component
function StatCounter({ value, duration = 1500, suffix = "" }: { value: number; duration?: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const elementRef = useRef<HTMLSpanElement>(null);
  const [triggered, setTriggered] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTriggered(true);
        }
      },
      { threshold: 0.1 }
    );

    const currentRef = elementRef.current;
    if (currentRef) observer.observe(currentRef);

    return () => {
      if (currentRef) observer.unobserve(currentRef);
    };
  }, []);

  useEffect(() => {
    if (!triggered) return;

    let start = 0;
    const end = value;
    if (start === end) {
      const t = setTimeout(() => setCount(end), 0);
      return () => clearTimeout(t);
    }

    const totalSteps = Math.min(end, 50); // cap steps
    const stepTime = duration / totalSteps;
    const increment = Math.ceil(end / totalSteps);

    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        clearInterval(timer);
        setCount(end);
      } else {
        setCount(start);
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [triggered, value, duration]);

  return (
    <span ref={elementRef} className="font-mono">
      {count}
      {suffix}
    </span>
  );
}

export default function HomePage() {
  const { playClick, playHover } = useAudio();
  const [subtitleIndex, setSubtitleIndex] = useState(0);

  // Cycle subtitles every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setSubtitleIndex((prev) => (prev + 1) % SUBTITLES.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-7xl mx-auto w-full flex-grow flex flex-col justify-center items-center relative z-10 select-none">
      
      {/* Spacer to push content down and make space for floating avatar */}
      <div className="h-44 sm:h-52 md:h-64 lg:h-72 w-full flex items-center justify-center pointer-events-none" />

      <div className="w-full text-center space-y-6 max-w-3xl">
        {/* Sub-header/Greetings */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="text-xs font-mono font-medium tracking-widest text-[#00D084] uppercase"
        >
          Curiosity to Intelligence
        </motion.p>

        {/* Display Name */}
        <motion.h1
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
          className="text-4xl sm:text-5xl md:text-7xl font-sans font-black tracking-tight text-white uppercase leading-none"
        >
          Vighan Raj Verma
        </motion.h1>

        {/* Cycling Subtitles */}
        <div className="h-8 flex items-center justify-center overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.p
              key={subtitleIndex}
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 0.8 }}
              exit={{ y: -20, opacity: 0 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              className="text-base sm:text-lg md:text-xl font-mono text-accent/80 font-semibold uppercase tracking-wide"
            >
              {SUBTITLES[subtitleIndex]}
            </motion.p>
          </AnimatePresence>
        </div>

        {/* Brief Introduction Pitch */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.7 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-sm sm:text-base text-gray-300 max-w-xl mx-auto leading-relaxed"
        >
          I build intelligent systems that combine data, machine learning, and computer vision to solve real-world problems. Currently exploring spatial computing, 3D reconstruction and neural rendering.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4"
        >
          <Link href="/projects" onClick={playClick} onMouseEnter={playHover}>
            <button className="w-48 py-3.5 text-xs font-mono font-bold tracking-widest text-black bg-[#00D084] hover:bg-[#4AFFB8] rounded-xl hover:shadow-[0_0_20px_rgba(0,208,132,0.4)] transition-all cursor-pointer flex items-center justify-center space-x-1.5">
              <span>EXPLORE WORK</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </Link>

          <Link href="/contact" onClick={playClick} onMouseEnter={playHover}>
            <button className="w-48 py-3.5 text-xs font-mono font-bold tracking-widest text-white border border-white/10 hover:border-accent/40 bg-white/5 hover:bg-white/10 rounded-xl transition-all cursor-pointer">
              CONTACT ME
            </button>
          </Link>

          <Link href="/resume" onClick={playClick} onMouseEnter={playHover}>
            <button className="w-48 py-3.5 text-xs font-mono font-bold tracking-widest text-gray-400 hover:text-white border border-white/5 hover:border-white/20 bg-transparent rounded-xl transition-all cursor-pointer">
              READ RESUME
            </button>
          </Link>
        </motion.div>

        {/* Social Icons */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="flex items-center justify-center space-x-6 pt-4 text-[#9CA3AF]"
        >
          <a
            href="https://github.com/Vighan-coder"
            target="_blank"
            rel="noopener noreferrer"
            onClick={playClick}
            onMouseEnter={playHover}
            className="hover:text-accent transition-colors"
            aria-label="GitHub"
          >
            <GithubIcon className="w-5 h-5" />
          </a>
          <a
            href="https://www.linkedin.com/in/vighan-raj-verma-4992b2317"
            target="_blank"
            rel="noopener noreferrer"
            onClick={playClick}
            onMouseEnter={playHover}
            className="hover:text-accent transition-colors"
            aria-label="LinkedIn"
          >
            <LinkedinIcon className="w-5 h-5" />
          </a>
          <a
            href="mailto:vighnrajverma00893@gmail.com"
            onClick={playClick}
            onMouseEnter={playHover}
            className="hover:text-accent transition-colors"
            aria-label="Email"
          >
            <Mail className="w-5 h-5" />
          </a>
        </motion.div>
      </div>

      {/* Quick Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl mt-24">
        {[
          {
            icon: <GraduationCap className="w-6 h-6 text-accent" />,
            title: "Student Portfolio",
            tag: "TRUBA INSTITUTE",
            desc: "2nd Year Computer Science student in Bhopal, developing foundational analytical skills and algorithmic structures."
          },
          {
            icon: <Microscope className="w-6 h-6 text-accent" />,
            title: "Research & Development",
            tag: "IISER BHOPAL",
            desc: "Active R&D Intern working on data science pipelines, deep learning datasets, and computer vision tools."
          },
          {
            icon: <Target className="w-6 h-6 text-accent" />,
            title: "Future Objectives",
            tag: "DATA SCIENCE",
            desc: "Focused on solving dense physical space reconstructions (Gaussian Splats) and complex machine learning applications."
          }
        ].map((card, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 + idx * 0.15 }}
            className="glass-panel p-6 space-y-4 hover:border-accent/30 hover:shadow-[0_0_30px_rgba(0,208,132,0.1)] transition-all group"
          >
            <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center group-hover:border-accent/40 transition-colors">
              {card.icon}
            </div>
            <div>
              <span className="text-[9px] font-mono font-extrabold tracking-widest text-accent/60 uppercase">
                {card.tag}
              </span>
              <h3 className="text-lg font-bold tracking-tight text-white mt-0.5">
                {card.title}
              </h3>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              {card.desc}
            </p>
          </motion.div>
        ))}
      </div>

      {/* Numerical Count-up Statistics Section */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-8 w-full max-w-5xl mt-24 px-4 text-center border-t border-white/5 pt-12">
        {[
          { value: 3, suffix: "+", label: "Years Learning" },
          { value: 12, suffix: "+", label: "Projects Completed" },
          { value: 1, suffix: "+", label: "Research Experience" },
          { value: 15, suffix: "+", label: "Technologies" },
          { value: 1, suffix: "", label: "R&D Internships" }
        ].map((stat, idx) => (
          <div key={idx} className="space-y-1">
            <h4 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
              <StatCounter value={stat.value} suffix={stat.suffix} />
            </h4>
            <p className="text-[10px] font-mono tracking-widest text-[#9CA3AF] uppercase">
              {stat.label}
            </p>
          </div>
        ))}
      </div>

      {/* Animated Scroll Indicator */}
      <motion.div
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
        className="mt-20 flex flex-col items-center space-y-1 text-gray-500 hover:text-white transition-colors cursor-pointer"
        onClick={() => {
          playClick();
          window.scrollBy({ top: window.innerHeight - 80, behavior: "smooth" });
        }}
      >
        <span className="text-[9px] font-mono tracking-widest uppercase">Scroll Down</span>
        <ChevronDown className="w-4 h-4" />
      </motion.div>
    </div>
  );
}
