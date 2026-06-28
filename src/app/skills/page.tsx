"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { 
  Code2, 
  Database, 
  Layout, 
  Settings, 
  Map, 
  Layers, 
  Compass,
  ArrowRight
} from "lucide-react";

interface SkillItem {
  name: string;
  desc: string;
  stage: string;
  projects: string;
  roadmap: string;
}

const SKILL_CATEGORIES: {
  category: string;
  icon: React.ReactNode;
  skills: SkillItem[];
}[] = [
  {
    category: "Programming",
    icon: <Code2 className="w-5 h-5 text-accent" />,
    skills: [
      { name: "Python", desc: "Core language for scientific computing, model training, and scripting.", stage: "Advanced", projects: "Real-Time Object Detection, Vectorization Engine", roadmap: "Parallel concurrency loops, C-bindings (ctypes)." },
      { name: "C++", desc: "High-performance processing, algorithms, and point-cloud ingestion modules.", stage: "Intermediate", projects: "3D Cloud Representation", roadmap: "GPU compiler setups, CUDA kernels." },
      { name: "SQL", desc: "Relational database querying, schema construction, and analytics pipelines.", stage: "Intermediate", projects: "Relational Data Cleaning & Schemas", roadmap: "Vector indexing, analytical tree execution." },
      { name: "JavaScript", desc: "Frontend interaction logic, Next.js setups, and React component management.", stage: "Intermediate", projects: "Frosted Portfolio Website", roadmap: "Vanilla WebGL shaders, async WebWorkers." }
    ]
  },
  {
    category: "Data Science",
    icon: <Database className="w-5 h-5 text-accent" />,
    skills: [
      { name: "ML / DL", desc: "Structuring classification neural networks, regressions, and validation tests.", stage: "Intermediate", projects: "YOLO Pipeline, CNN experiments", roadmap: "Mathematical validation bounds, loss derivations." },
      { name: "Computer Vision", desc: "Feature detectors, convolutional pipelines, and spatial representations.", stage: "Intermediate", projects: "Dataset Preprocessing & Alignment", roadmap: "Gaussian Splatting, NeRF rendering parameters." },
      { name: "Statistics", desc: "Probability distribution maps, regressions, and statistical tests.", stage: "Intermediate", projects: "Vectorization analytics engine", roadmap: "Bayesian optimization models, sampling limits." },
      { name: "NumPy / Pandas", desc: "High-dimensional array transformations and vector operations.", stage: "Advanced", projects: "Scientific Data Cleaning Pipelines", roadmap: "C-level memory alignments for arrays." },
      { name: "Scikit-Learn", desc: "Applying regression models, clustering classifiers, and validations.", stage: "Intermediate", projects: "Vectorization analytics engine", roadmap: "Custom estimators, optimizer interfaces." },
      { name: "PyTorch", desc: "Neural network layer configurations, optimizer configurations, and model testing.", stage: "Intermediate", projects: "YOLO Detection Pipeline", roadmap: "Custom autograd functions, CUDA integrations." },
      { name: "OpenCV", desc: "Image arrays normalization, colorspace conversions, and image filtering.", stage: "Intermediate", projects: "Object Detection preprocessing", roadmap: "Real-time GPU matrices, optical flow kernels." }
    ]
  },
  {
    category: "Frontend",
    icon: <Layout className="w-5 h-5 text-accent" />,
    skills: [
      { name: "React / Next.js", desc: "Modular interface rendering, App Router rendering, and SEO structures.", stage: "Intermediate", projects: "Frosted Portfolio Website", roadmap: "Server Actions, RSC caching configurations." },
      { name: "Three.js / R3F", desc: "GPU Canvas rendering, lighting setups, mesh structures, and camera control rigs.", stage: "Intermediate", projects: "Frosted Portfolio 3D canvas", roadmap: "Custom GLSL vertex & fragment shaders." },
      { name: "Tailwind CSS", desc: "Tailwind v4 theme configurations, custom styles, and responsive layout grids.", stage: "Advanced", projects: "Frosted Portfolio layout", roadmap: "Custom CSS transitions, postCSS variables." }
    ]
  },
  {
    category: "Tools & OS",
    icon: <Settings className="w-5 h-5 text-accent" />,
    skills: [
      { name: "Git / GitHub", desc: "Branching loops, version logging, pull requests, and file checks.", stage: "Advanced", projects: "All repository codes", roadmap: "Automated GitHub Actions, CI/CD setups." },
      { name: "Linux / Bash", desc: "Server shells, shell commands, script automations, and file system loops.", stage: "Intermediate", projects: "Development workflow setups", roadmap: "Bash script cron triggers, file watchers." },
      { name: "Jupyter", desc: "Document reporting, modeling iterations, and localized charts.", stage: "Advanced", projects: "Data exploration notebooks", roadmap: "Reproducible document outputs, remote servers." },
      { name: "Blender", desc: "Configuring mesh vertices, scale adjustments, and lighting rigs.", stage: "Beginner", projects: "3D reference meshes", roadmap: "Procedural geometry nodes, texture maps." }
    ]
  }
];

export default function SkillsPage() {
  const { playClick, playHover } = useAudio();
  
  // Default selected skill for display card
  const [selectedSkill, setSelectedSkill] = useState<SkillItem>(SKILL_CATEGORIES[0].skills[0]);

  return (
    <div className="max-w-6xl mx-auto w-full space-y-16 relative z-10 pt-8">
      
      {/* Title */}
      <div className="text-center space-y-4">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          className="text-xs font-mono tracking-widest text-[#9CA3AF] uppercase"
        >
          Toolbox
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl sm:text-5xl font-sans font-black tracking-tight text-white uppercase"
        >
          Skills & <span className="text-accent">Competencies</span>
        </motion.h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        
        {/* Left/Middle: Skills Spheres divided by category */}
        <div className="lg:col-span-2 space-y-10">
          {SKILL_CATEGORIES.map((cat, catIdx) => (
            <div key={catIdx} className="space-y-4">
              <div className="flex items-center space-x-2 text-white justify-center sm:justify-start">
                {cat.icon}
                <h2 className="text-sm font-mono font-bold tracking-widest uppercase">{cat.category}</h2>
              </div>
              
              {/* Spheres Grid */}
              <div className="flex flex-wrap gap-4 justify-center sm:justify-start">
                {cat.skills.map((skill, sIdx) => {
                  const isCurrent = selectedSkill.name === skill.name;
                  return (
                    <motion.div
                      key={sIdx}
                      whileHover={{ scale: 1.1, y: -5 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        playClick();
                        setSelectedSkill(skill);
                      }}
                      onMouseEnter={() => {
                        playHover();
                        setSelectedSkill(skill);
                      }}
                      className={`w-20 h-20 sm:w-24 sm:h-24 rounded-full flex items-center justify-center text-center p-3 cursor-pointer transition-all duration-200 ${
                        isCurrent
                          ? "border-accent text-white bg-accent/10 shadow-[inset_0_4px_12px_rgba(0,208,132,0.3),_0_0_20px_rgba(0,208,132,0.3)]"
                          : "border-white/10 text-gray-400 bg-white/5 hover:border-accent/40 hover:text-white shadow-[inset_0_4px_12px_rgba(255,255,255,0.05),_inset_0_-4px_12px_rgba(0,0,0,0.4)]"
                      }`}
                    >
                      <span className="text-[10px] sm:text-xs font-mono font-bold select-none leading-tight">
                        {skill.name}
                      </span>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Right Column: Focus Detail Display Card */}
        <div className="lg:col-span-1 lg:sticky lg:top-24">
          <AnimatePresence mode="wait">
            <motion.div
              key={selectedSkill.name}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className="glass-panel p-6 border-accent bg-[#00D084]/2 space-y-6 relative overflow-hidden"
            >
              {/* Background ambient radial light inside card */}
              <div className="absolute -top-12 -right-12 w-24 h-24 bg-[#00D084]/10 rounded-full blur-xl pointer-events-none" />

              {/* Title & Stage */}
              <div className="flex items-center justify-between border-b border-white/5 pb-4">
                <h3 className="text-xl font-bold text-white tracking-tight">{selectedSkill.name}</h3>
                <span className="px-2.5 py-0.5 rounded-full text-[9px] font-mono font-bold bg-accent/20 text-accent border border-accent/30 tracking-widest uppercase">
                  {selectedSkill.stage}
                </span>
              </div>

              {/* Stats / Details */}
              <div className="space-y-4 text-xs leading-relaxed">
                
                {/* Description */}
                <div className="space-y-1.5">
                  <span className="font-mono text-[10px] text-accent/80 font-bold block uppercase tracking-wider flex items-center gap-1">
                    <Compass className="w-3.5 h-3.5" />
                    Focus Area
                  </span>
                  <p className="text-gray-300">{selectedSkill.desc}</p>
                </div>

                {/* Related Projects */}
                <div className="space-y-1.5">
                  <span className="font-mono text-[10px] text-accent/80 font-bold block uppercase tracking-wider flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5" />
                    Applied In
                  </span>
                  <p className="text-gray-300 font-mono text-[10px] italic">{selectedSkill.projects}</p>
                </div>

                {/* Future Roadmap */}
                <div className="space-y-1.5 pt-2 border-t border-white/5">
                  <span className="font-mono text-[10px] text-accent/80 font-bold block uppercase tracking-wider flex items-center gap-1">
                    <Map className="w-3.5 h-3.5 animate-pulse" />
                    Roadmap Goal
                  </span>
                  <p className="text-gray-400 flex items-start gap-1.5">
                    <ArrowRight className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
                    <span>{selectedSkill.roadmap}</span>
                  </p>
                </div>

              </div>

            </motion.div>
          </AnimatePresence>
        </div>

      </div>

    </div>
  );
}
