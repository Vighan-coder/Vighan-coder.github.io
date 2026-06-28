"use client";

import React, { useState, useRef } from "react";
import { useAudio } from "@/context/AudioContext";
import { 
  ZoomIn, 
  ZoomOut, 
  Maximize2, 
  Download, 
  Printer, 
  Briefcase, 
  GraduationCap, 
  Code2, 
  Award,
  Globe
} from "lucide-react";

export default function ResumePage() {
  const { playClick, playHover } = useAudio();
  const viewerRef = useRef<HTMLDivElement>(null);
  
  const [zoom, setZoom] = useState(1.0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleZoomIn = () => {
    playClick();
    setZoom((z) => Math.min(z + 0.1, 1.4));
  };

  const handleZoomOut = () => {
    playClick();
    setZoom((z) => Math.max(z - 0.1, 0.8));
  };

  const handleFullscreen = () => {
    playClick();
    if (!viewerRef.current) return;

    if (!document.fullscreenElement) {
      viewerRef.current.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch((err) => {
        console.error("Fullscreen error", err);
      });
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handlePrint = () => {
    playClick();
    window.print();
  };

  const handleDownload = async (e: React.MouseEvent<HTMLAnchorElement>) => {
    playClick();
    try {
      const response = await fetch("/Vighan_Raj_Verma_Resume.pdf", { method: "HEAD" });
      if (!response.ok) {
        e.preventDefault();
        // Dynamic fallback to native print utility
        alert("Physical PDF file not found in public folder. Redirecting to native browser print-to-PDF...");
        window.print();
      }
    } catch (err) {
      e.preventDefault();
      window.print();
    }
  };

  return (
    <div className="max-w-5xl mx-auto w-full space-y-8 relative z-10 pt-8 print:p-0 print:pt-0">
      
      {/* Top Toolbar (Hidden during print) */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white/5 border border-white/10 rounded-2xl p-4 print:hidden">
        <div className="flex items-center space-x-2">
          <GraduationCap className="w-5 h-5 text-accent" />
          <h1 className="text-sm font-mono font-bold text-white tracking-widest uppercase">
            DIGITAL CURRICULUM VITAE
          </h1>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-2">
          
          {/* Zoom Out */}
          <button 
            onClick={handleZoomOut} 
            onMouseEnter={playHover}
            className="p-2 bg-white/5 border border-white/5 hover:border-accent/40 rounded-lg text-gray-400 hover:text-white transition-colors"
            aria-label="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          {/* Zoom Level Indicator */}
          <span className="text-[10px] font-mono text-gray-400 w-12 text-center select-none">
            {Math.round(zoom * 100)}%
          </span>

          {/* Zoom In */}
          <button 
            onClick={handleZoomIn} 
            onMouseEnter={playHover}
            className="p-2 bg-white/5 border border-white/5 hover:border-accent/40 rounded-lg text-gray-400 hover:text-white transition-colors"
            aria-label="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <span className="h-6 w-[1px] bg-white/10 mx-1" />

          {/* Fullscreen */}
          <button 
            onClick={handleFullscreen} 
            onMouseEnter={playHover}
            className="p-2 bg-white/5 border border-white/5 hover:border-accent/40 rounded-lg text-gray-400 hover:text-white transition-colors"
            aria-label="Fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>

          {/* Print */}
          <button 
            onClick={handlePrint} 
            onMouseEnter={playHover}
            className="p-2 bg-white/5 border border-white/5 hover:border-accent/40 rounded-lg text-gray-400 hover:text-white transition-colors"
            aria-label="Print Document"
          >
            <Printer className="w-4 h-4" />
          </button>

          {/* Download Mock File */}
          <a 
            href="/Vighan_Raj_Verma_Resume.pdf" 
            download
            onClick={handleDownload}
            onMouseEnter={playHover}
            className="flex items-center space-x-1.5 px-4 py-2 bg-accent text-black font-mono font-bold text-xs rounded-lg hover:bg-[#4AFFB8] transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">DOWNLOAD PDF</span>
          </a>

        </div>
      </div>

      {/* Main Document Viewer */}
      <div 
        ref={viewerRef} 
        className={`w-full overflow-auto flex justify-start md:justify-center p-4 md:p-6 border border-white/5 bg-black/40 rounded-2xl print:bg-white print:border-none print:p-0 ${
          isFullscreen ? "bg-[#050505] p-12 h-screen" : ""
        }`}
      >
        
        {/* Styled A4 sheet document */}
        <div 
          style={{ 
            transform: `scale(${zoom})`, 
            transformOrigin: typeof window !== "undefined" && window.innerWidth < 768 ? "top left" : "top center",
            transition: "transform 0.15s ease-out"
          }}
          className="w-[210mm] min-h-[297mm] bg-white text-gray-900 shadow-2xl p-12 space-y-8 print:transform-none print:shadow-none print:p-0 print:w-full print:text-black shrink-0 relative border border-gray-200 print:border-none"
        >
          {/* Header Contact Block */}
          <div className="border-b-2 border-accent pb-6 flex justify-between items-start">
            <div className="space-y-1">
              <h2 className="text-3xl font-sans font-black tracking-tight text-gray-900 leading-none">
                VIGHAN RAJ VERMA
              </h2>
              <p className="text-xs font-mono text-accent font-bold tracking-wider uppercase">
                Aspiring Data Scientist & Computer Vision Explorer
              </p>
            </div>
            <div className="text-right text-[10px] font-mono text-gray-500 space-y-0.5 print:text-black">
              <p>vighnrajverma00893@gmail.com</p>
              <p>+91 (Bhopal, MP, India)</p>
              <p>github.com/Vighan-coder</p>
              <p>linkedin.com/in/vighan-raj-verma-4992b2317</p>
            </div>
          </div>

          {/* Academic Block */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-gray-800 border-b border-gray-200 pb-1">
              <GraduationCap className="w-4 h-4 text-accent" />
              <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-gray-900">Education</h3>
            </div>
            
            <div className="space-y-4">
              <div className="flex justify-between items-start text-xs">
                <div>
                  <h4 className="font-bold text-gray-900">Truba Institute of Engineering & Information Technology</h4>
                  <p className="text-gray-600 font-mono text-[11px]">B.Tech in Computer Science & Engineering</p>
                  <p className="text-gray-500 text-[10px] mt-1">Coursework: Data Structures, Discrete Mathematics, DBMS, Probability & Linear Algebra</p>
                </div>
                <div className="text-right font-mono text-gray-500 print:text-black shrink-0">
                  <p className="font-bold">Bhopal, MP, India</p>
                  <p>Expected Graduation: 2028</p>
                </div>
              </div>

              <div className="flex justify-between items-start text-xs">
                <div>
                  <h4 className="font-bold text-gray-900">Model Higher Secondary School</h4>
                  <p className="text-gray-600 font-mono text-[11px]">High School Certification (12th Grade - PCM)</p>
                  <p className="text-gray-500 text-[10px] mt-1">Achievements: Passed with First Division</p>
                </div>
                <div className="text-right font-mono text-gray-500 print:text-black shrink-0">
                  <p className="font-bold">Bhopal, MP, India</p>
                  <p>Completed: 2024</p>
                </div>
              </div>

              <div className="flex justify-between items-start text-xs">
                <div>
                  <h4 className="font-bold text-gray-900">Model Higher Secondary School</h4>
                  <p className="text-gray-600 font-mono text-[11px]">Secondary Certification (10th Grade)</p>
                  <p className="text-gray-500 text-[10px] mt-1">Achievements: Passed with First Division, chose PCM stream</p>
                </div>
                <div className="text-right font-mono text-gray-500 print:text-black shrink-0">
                  <p className="font-bold">Bhopal, MP, India</p>
                  <p>Completed: 2022</p>
                </div>
              </div>
            </div>
          </div>

          {/* Internships Block */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-gray-800 border-b border-gray-200 pb-1">
              <Briefcase className="w-4 h-4 text-accent" />
              <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-gray-900">Experience</h3>
            </div>

            <div className="flex justify-between items-start text-xs">
              <div>
                <h4 className="font-bold text-gray-900">Research & Development Intern</h4>
                <p className="text-gray-600 font-mono text-[11px]">Indian Institute of Science Education and Research (IISER)</p>
                <ul className="list-disc list-inside text-gray-600 text-[11px] mt-2 space-y-1">
                  <li>Contributing to a research paper on fusing independently trained aerial and street-view Gaussian splats.</li>
                  <li>Edited raw data to correct bit depth, sampled video frames, and ran COLMAP structure-from-motion.</li>
                  <li>Trained 3DGS models, solved coordinate scaling, and fused multi-view reconstructions.</li>
                  <li>Leveraged Linux/Bash terminal commands and Git/GitHub version control workflows.</li>
                </ul>
              </div>
              <div className="text-right font-mono text-gray-500 print:text-black shrink-0">
                <p className="font-bold">Bhopal, MP, India</p>
                <p>24 May - 21 July</p>
              </div>
            </div>
          </div>

          {/* Selected Projects Block */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-gray-800 border-b border-gray-200 pb-1">
              <Code2 className="w-4 h-4 text-accent" />
              <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-gray-900">Projects</h3>
            </div>

            <div className="space-y-4">
              {/* Project 1 */}
              <div className="text-xs">
                <div className="flex justify-between font-bold">
                  <h4 className="text-gray-900">Real-Time Object Detection Pipeline</h4>
                  <span className="font-mono text-gray-500 print:text-black">Python, OpenCV, PyTorch</span>
                </div>
                <p className="text-gray-600 mt-1 text-[11px]">
                  Configured image preprocessing normalization filters in OpenCV, fine-tuned YOLO layers, and compiled weights to ONNX format. Achieved 45 FPS on edge nodes with 91.8% mAP accuracy.
                </p>
              </div>

              {/* Project 2 */}
              <div className="text-xs">
                <div className="flex justify-between font-bold">
                  <h4 className="text-gray-900">High-Dimensional Vectorization Engine</h4>
                  <span className="font-mono text-gray-500 print:text-black">Python, NumPy, Pandas</span>
                </div>
                <p className="text-gray-600 mt-1 text-[11px]">
                  Vectorized analytical routines using broad linear algebra expressions, bypassing native python nested iterator structures. Accelerated data processing cycles by 78%.
                </p>
              </div>
            </div>
          </div>

          {/* Skills Block */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-gray-800 border-b border-gray-200 pb-1">
              <Globe className="w-4 h-4 text-accent" />
              <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-gray-900">Skills</h3>
            </div>

            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <p className="font-bold text-gray-900">Programming Languages:</p>
                <p className="text-gray-600 font-mono text-[11px]">Python, C++, SQL, JavaScript</p>
              </div>
              <div>
                <p className="font-bold text-gray-900">Libraries & Data Science:</p>
                <p className="text-gray-600 font-mono text-[11px]">NumPy, Pandas, Scikit-learn, OpenCV, PyTorch</p>
              </div>
              <div>
                <p className="font-bold text-gray-900">Frontend Web:</p>
                <p className="text-gray-600 font-mono text-[11px]">React, Next.js, Three.js, Tailwind CSS</p>
              </div>
              <div>
                <p className="font-bold text-gray-900">Tools & Operating Systems:</p>
                <p className="text-gray-600 font-mono text-[11px]">Git, GitHub, Linux, Jupyter, Blender</p>
              </div>
            </div>
          </div>

          {/* Achievements Block */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-gray-800 border-b border-gray-200 pb-1">
              <Award className="w-4 h-4 text-accent" />
              <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-gray-900">Academic Trajectory</h3>
            </div>
            <ul className="list-disc list-inside text-xs text-gray-600 space-y-1 text-[11px]">
              <li>Appointed Research & Development Intern at IISER Bhopal during 2nd year of CSE undergraduate studies.</li>
              <li>Consistently maintaining high standing grades in core calculus and discrete structures coursework.</li>
              <li>Actively developing visual systems and 3D coordinate mapping configurations in personal time.</li>
            </ul>
          </div>

        </div>

      </div>

      {/* Adjust container spacer to handle scale layout clipping overflow */}
      <div className="h-20 print:hidden" />

    </div>
  );
}
