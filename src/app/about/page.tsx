"use client";

import React from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { 
  Brain, 
  Database, 
  Eye, 
  LineChart, 
  GitBranch, 
  Sigma, 
  Cuboid as Cube, 
  Search, 
  Lightbulb, 
  Zap, 
  BookOpen, 
  Award 
} from "lucide-react";

// Journey Timeline Steps
const TIMELINE_STEPS = [
  {
    year: "2022",
    title: "Secondary School Graduation",
    desc: "Passed 10th Grade with First Division from Model Higher Secondary School, Bhopal, selecting the PCM stream for senior secondary classes."
  },
  {
    year: "2024",
    title: "High School Graduation",
    desc: "Graduated with First Division (12th Passed) from Model Higher Secondary School, Bhopal, strengthening numerical and analytical foundations."
  },
  {
    year: "2024",
    title: "B.Tech Computer Science @ Truba Institute",
    desc: "Enrolled in B.Tech Computer Science (Expected Graduation: 2028). Immersed in computer systems, math foundations, and code."
  },
  {
    year: "2025",
    title: "Focus on AI & Machine Learning",
    desc: "Dived into probability, linear algebra, numerical analysis, statistics, data analytics libraries, and model design."
  },
  {
    year: "24 May - 21 July",
    title: "Research & Development Intern @ IISER Bhopal",
    desc: "Contributing to a research paper on fusing aerial and street-view Gaussian splats. Handled bit-depth conversions, frame-sampling, COLMAP SfM, and 3DGS training/fusion. Learned Linux commands and Git/GitHub."
  },
  {
    year: "2028 & Beyond",
    title: "Future Aspirations",
    desc: "Transitioning toward graduate studies, publishing research in deep learning, and engineering scalable AI solutions."
  }
];

// Interactive Domain Interest Cards
const INTERESTS = [
  {
    icon: <Brain className="w-5 h-5 text-accent" />,
    title: "Artificial Intelligence",
    desc: "Understanding neural representations, cognitive models, and programmatic reasoning systems."
  },
  {
    icon: <Database className="w-5 h-5 text-accent" />,
    title: "Data Science",
    desc: "Extracting insights from high-dimensional datasets using linear estimation, regression, and statistics."
  },
  {
    icon: <Eye className="w-5 h-5 text-accent" />,
    title: "Computer Vision",
    desc: "Designing feature descriptors, image classification algorithms, and convolutional networks."
  },
  {
    icon: <Cube className="w-5 h-5 text-accent" />,
    title: "3D Vision / Gaussian Splatting",
    desc: "Reconstructing dense spatial points from sparse camera views and optimizing density gradient kernels."
  },
  {
    icon: <Search className="w-5 h-5 text-accent" />,
    title: "Academic Research",
    desc: "Formulating novel hypotheses, structuring robust testing methodologies, and writing papers."
  },
  {
    icon: <Sigma className="w-5 h-5 text-accent" />,
    title: "Mathematics & Statistics",
    desc: "Deep-diving into multivariate calculus, linear algebra, bayesian models, and optimization bounds."
  },
  {
    icon: <GitBranch className="w-5 h-5 text-accent" />,
    title: "Open Source",
    desc: "Contributing libraries, debugging core architectures, and engaging with technical communities."
  },
  {
    icon: <LineChart className="w-5 h-5 text-accent" />,
    title: "Data Analysis",
    desc: "Mastering pipelines with NumPy, Pandas, Scikit-learn, and configuring real-time visual dashboards."
  }
];

// Values
const VALUES = [
  {
    icon: <Lightbulb className="w-5 h-5 text-accent" />,
    title: "Innovation First",
    desc: "Always pushing past traditional algorithms to discover creative, neural-driven solutions."
  },
  {
    icon: <Zap className="w-5 h-5 text-accent" />,
    title: "Performance & Detail",
    desc: "Structuring clean, optimized operations to maximize compute efficiency and frame rate output."
  },
  {
    icon: <BookOpen className="w-5 h-5 text-accent" />,
    title: "Perpetual Learning",
    desc: "Approaching computer science with deep curiosity, staying current with ArXiv publications and open repos."
  },
  {
    icon: <Award className="w-5 h-5 text-accent" />,
    title: "Scientific Integrity",
    desc: "Maintaining strict validation guidelines, clear statistical claims, and reproducible code structures."
  }
];

export default function AboutPage() {
  const { playClick, playHover } = useAudio();

  return (
    <div className="max-w-5xl mx-auto w-full space-y-24 relative z-10 pt-8">
      {/* Epic Header */}
      <div className="text-center space-y-4">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          className="text-xs font-mono tracking-widest text-[#9CA3AF] uppercase"
        >
          Who I Am
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl sm:text-5xl font-sans font-black tracking-tight text-white uppercase leading-none"
        >
          Curious. Creative. <br className="sm:hidden" />
          <span className="text-accent">Always Learning.</span>
        </motion.h1>
      </div>

      {/* Profile/Intro Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-start">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="md:col-span-1 glass-panel p-6 flex flex-col items-center text-center space-y-4"
        >
          <div className="relative w-36 h-36 rounded-full border border-white/15 flex items-center justify-center overflow-hidden shadow-inner group">
            {/* Glowing border ring */}
            <div className="absolute inset-0 border border-dashed border-[#00D084]/30 rounded-full animate-[spin_20s_linear_infinite] z-10 pointer-events-none" />
            <Image 
              src="/profile.jpg" 
              alt="Vighan Raj Verma" 
              width={144}
              height={144}
              priority
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
            />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Vighan Raj Verma</h3>
            <p className="text-xs font-mono text-[#9CA3AF]">Expected B.Tech Grad: 2028</p>
          </div>
          <p className="text-xs text-gray-400">
            Truba Institute of Engineering & IT, Bhopal
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="md:col-span-2 space-y-6 text-sm text-gray-300 leading-relaxed"
        >
          <h2 className="text-xl font-bold tracking-tight text-white">The Journey Forward</h2>
          <p>
            I am a second-year Computer Science engineering undergraduate, driven by the challenge of bridging data structures with statistical learning. My focus centers on building intelligence into computer systems, transforming high-dimensional numeric arrays into concrete answers.
          </p>
          <p>
            Currently, as a Research & Development Intern at <span className="text-[#00D084] font-semibold">IISER Bhopal</span>, I assist in designing data structures, cleaning image arrays, and building pipeline processes. This experience has deepened my respect for clean mathematical modeling and reproducible algorithms.
          </p>
          <p>
            Whether implementing a basic neural network layer in raw NumPy, structuring SQL queries to ingest raw files, or tuning 3D coordinates inside Gaussian Splats, I treat every project as a step toward mastering data science and engineering workflows.
          </p>
        </motion.div>
      </div>

      {/* Vertical Journey Timeline */}
      <div className="space-y-12">
        <h2 className="text-2xl font-black tracking-tight text-white uppercase text-center">My Educational Journey</h2>
        <div className="relative border-l border-white/10 max-w-xl mx-auto pl-6 sm:pl-8 space-y-12">
          {TIMELINE_STEPS.map((step, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.6, delay: idx * 0.15 }}
              className="relative space-y-2"
            >
              {/* Timeline Bullet Node */}
              <div className="absolute -left-[31px] sm:-left-[39px] top-1 w-4 h-4 rounded-full bg-[#050505] border border-accent flex items-center justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-ping" />
              </div>

              <span className="text-xs font-mono font-bold text-accent">{step.year}</span>
              <h3 className="text-base font-bold text-white tracking-tight">{step.title}</h3>
              <p className="text-xs text-gray-400 leading-relaxed">{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Interactive Glass Domain Cards */}
      <div className="space-y-8">
        <h2 className="text-2xl font-black tracking-tight text-white uppercase text-center">Domain Interests</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {INTERESTS.map((item, idx) => (
            <motion.div
              key={idx}
              whileHover={{ y: -4, scale: 1.02 }}
              onClick={playClick}
              onMouseEnter={playHover}
              className="glass-panel p-5 space-y-3 cursor-pointer hover:border-accent/40 hover:shadow-[0_0_20px_rgba(0,208,132,0.08)] transition-all"
            >
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center">
                {item.icon}
              </div>
              <h3 className="text-sm font-bold text-white tracking-tight">{item.title}</h3>
              <p className="text-[11px] text-gray-400 leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Core Values Section */}
      <div className="space-y-8">
        <h2 className="text-2xl font-black tracking-tight text-white uppercase text-center">Core Engineering Values</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {VALUES.map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 15 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: idx * 0.1 }}
              className="glass-panel p-6 flex space-x-4 items-start"
            >
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
                {item.icon}
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-bold text-white tracking-tight">{item.title}</h3>
                <p className="text-xs text-gray-400 leading-relaxed">{item.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Elegant Quote Section */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-panel p-8 max-w-2xl mx-auto text-center border-accent/20 bg-accent/2 relative overflow-hidden"
      >
        <div className="absolute top-0 right-0 w-24 h-24 bg-accent/5 rounded-full blur-xl pointer-events-none" />
        <span className="text-4xl text-accent font-serif leading-none block h-2 select-none">&ldquo;</span>
        <blockquote className="text-base sm:text-lg italic font-medium text-white max-w-xl mx-auto leading-relaxed mt-2">
          From data models to visual representations, coding isn&apos;t just instructions; it is the craft of parsing chaos into structured intelligence.
        </blockquote>
        <cite className="block text-xs font-mono text-accent/80 mt-4 not-italic uppercase tracking-widest">
          &mdash; Vighan Raj Verma
        </cite>
      </motion.div>
    </div>
  );
}
