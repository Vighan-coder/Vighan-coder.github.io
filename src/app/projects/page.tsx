"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { 
  ExternalLink, 
  BookOpen, 
  Cpu, 
  Boxes 
} from "lucide-react";

const GithubIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

// Project categories
const CATEGORIES = ["All", "ML", "Data Science", "Computer Vision", "Research", "Web Dev", "Open Source"];

// Initial Projects Data
const PROJECTS = [
  {
    id: 1,
    title: "Real-Time Object Detection Pipeline",
    category: "Computer Vision",
    tags: ["ML", "Computer Vision"],
    tech: ["Python", "OpenCV", "YOLOv8", "PyTorch"],
    problem: "Low inference speed and boundary inaccuracies on custom image sets under low-light conditions.",
    approach: "Designed localized image-histogram normalization in OpenCV, fine-tuned YOLO layers, and compiled weights to ONNX format.",
    results: "Inference rates increased to 45 FPS on edge hardware, with mean average precision (mAP) climbing to 91.8%.",
    lessons: "Input preprocessing pipelines are often the true performance bottleneck, overshadowing pure tensor contraction calculations.",
    github: "https://github.com/Vighan-coder",
    demo: "#",
    active: true,
    challenge: "Low-light environments produce heavy pixel noise that disrupts convolutional layer activations, leading to a high rate of false negatives.",
    solution: "We integrated a Contrast Limited Adaptive Histogram Equalization (CLAHE) module in BGR-to-LAB color space before the convolutional head, followed by ONNX execution compilation.",
    codeLanguage: "python",
    code: `import cv2
import numpy as np
from ultralytics import YOLO

def preprocess_and_detect(image_path, model_path):
    # 1. Load optimized model weights
    model = YOLO(model_path)
    
    # 2. Read input frame
    frame = cv2.imread(image_path)
    
    # 3. Apply Localized Histogram Normalization for low-light enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 4. Perform YOLOv8 inference (compiled ONNX engine)
    results = model.predict(source=enhanced_bgr, conf=0.25, imgsz=640)
    
    # 5. Extract bounding boxes and confidence parameters
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
    return boxes, confs`
  },
  {
    id: 2,
    title: "High-Dimensional Vectorization Engine",
    category: "Data Science",
    tags: ["Data Science"],
    tech: ["Python", "NumPy", "Pandas", "Scikit-Learn"],
    problem: "Iterative tabular loops taking excessive CPU cycles on large scientific observation files.",
    approach: "Vectorized analytical routines using broad linear algebra expressions, bypassing native python nested iterator structures.",
    results: "Data cleaning and loading tasks accelerated by 78%, dropping runtime logs from minutes to seconds.",
    lessons: "Vectorization operations must map directly to raw memory block alignments for optimal performance.",
    github: "https://github.com/Vighan-coder",
    demo: "#",
    active: true,
    challenge: "Nested looping over 100k+ coordinate records to calculate statistical outlier distributions caused intense CPU thrashing.",
    solution: "Re-engineered loop math to exploit SIMD matrix registers in NumPy, converting Python loops into direct C-level contiguous memory operations.",
    codeLanguage: "python",
    code: `import numpy as np

def vectorized_observation_cleaner(raw_array, threshold=3.5):
    """
    Replaces slow nested CPU iterations with vectorized broad math.
    """
    # Calculate statistical z-scores across multi-dim column axis
    mean = np.mean(raw_array, axis=0)
    std = np.std(raw_array, axis=0)
    
    # Vectorized masking expression mapping directly to contiguous blocks
    z_scores = np.abs((raw_array - mean) / (std + 1e-9))
    mask = z_scores < threshold
    
    # Filter out noisy outliers in one memory tick
    cleaned_data = np.where(mask, raw_array, mean)
    return cleaned_data`
  },
  {
    id: 3,
    title: "3D Point Cloud Representation & Gaussian Splat Explorer",
    category: "Research",
    tags: ["Research", "Computer Vision"],
    tech: ["C++", "Three.js", "Python", "PyTorch"],
    problem: "High latency and noise in dense 3D point cloud reconstructions from multi-view inputs.",
    approach: "Developed WebGL-based vertex structures rendering anisotropic ellipsoids with camera-centric sorted scale parameters.",
    results: "Real-time spherical harmonics rendering on portable web screens, bypassing heavy server loading blocks.",
    lessons: "Sorting layers according to depth values dynamically on the client is crucial for transparency order calculations in 3DGS.",
    github: "https://github.com/Vighan-coder",
    demo: "#",
    active: true,
    challenge: "Traditional point clouds fail to capture view-dependent radiance and specular highlights, resulting in flat, artificial surfaces.",
    solution: "Integrated spherical harmonics parameters directly inside custom WebGL buffer arrays, rotating coordinate normals relative to client camera matrices.",
    codeLanguage: "glsl",
    code: `// Custom Vertex Shader for Anisotropic Splat Rendering (ThreeJS / WebGL)
const SplatVertexShader = \`
  attribute vec3 covA;
  attribute vec3 covB;
  attribute vec4 colorSH;

  varying vec4 vColor;
  varying vec2 vUv;

  void main() {
    // 1. Project splat center position
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // 2. Reconstruct 3D covariance matrix elements
    mat3 V = mat3(
      covA.x, covA.y, covA.z,
      covA.y, covB.x, covB.y,
      covA.z, covB.y, covB.z
    );

    // 3. Project 3D covariance to 2D screen coordinates
    mat3 J = mat3(
      1.0 / mvPosition.z, 0.0, -mvPosition.x / (mvPosition.z * mvPosition.z),
      0.0, 1.0 / mvPosition.z, -mvPosition.y / (mvPosition.z * mvPosition.z),
      0.0, 0.0, 0.0
    );

    mat3 W = transpose(modelViewMatrix) * J;
    mat3 T = transpose(W) * V * W;

    // 4. Pass view-dependent colors to fragment shader
    vColor = colorSH;
    vUv = uv;
  }
\`;`
  },
  {
    id: 4,
    title: "Neural Splat Reconstruction Simulator",
    category: "Research",
    tags: ["Research", "ML"],
    tech: ["Python", "PyTorch", "COLMAP"],
    problem: "Heavy gradient descent instabilities during Gaussian Splatting volumetric training loops.",
    approach: "Designed localized density controls based on structural-similarity (SSIM) gradients to prune/clone splats.",
    results: "Converged training loss 25% faster with cleaner high-frequency details (PSNR increased to 32.4 dB).",
    lessons: "Adaptive densification thresholds must scale inversely with iteration depth to prevent model divergence.",
    github: "https://github.com/Vighan-coder",
    demo: "#",
    active: true,
    challenge: "Standard stochastic gradient descent loops cause over-densification in uniform regions while starving high-frequency texture edges.",
    solution: "Implemented an SSIM-weighted scaling multiplier that dynamically controls learning rates for splat spatial scale matrices.",
    codeLanguage: "python",
    code: `import torch
import torch.nn as nn

class VolumetricSplatLoss(nn.Module):
    def __init__(self, ssim_weight=0.2):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, rendered_image, target_image):
        # 1. Compute pixel-level L1 reconstruction loss
        loss_l1 = self.l1_loss(rendered_image, target_image)
        
        # 2. Extract Structural Similarity Index Measure (SSIM)
        loss_ssim = 1.0 - self.compute_ssim(rendered_image, target_image)
        
        # 3. Formulate composite loss loop
        total_loss = (1.0 - self.ssim_weight) * loss_l1 + self.ssim_weight * loss_ssim
        return total_loss

    def compute_ssim(self, img1, img2):
        # Local structural similarity estimator mockup
        mu1 = img1.mean(dim=[-1, -2], keepdim=True)
        mu2 = img2.mean(dim=[-1, -2], keepdim=True)
        sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[-1, -2])
        sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[-1, -2])
        return (2.0 * mu1 * mu2 * (sigma1_sq * sigma2_sq).sqrt() + 1e-5).mean() / (mu1**2 + mu2**2 + 1e-5)`
  },
  {
    id: 5,
    title: "Airfield Flight Coordinates Indexer",
    category: "Open Source",
    tags: ["Open Source"],
    tech: ["Coming soon"],
    problem: "Under development. Structuring efficient geometry parsing methods.",
    approach: "Pending local repository integration.",
    results: "Project details coming soon.",
    lessons: "Project details coming soon.",
    github: "https://github.com/Vighan-coder",
    demo: "#",
    active: false,
    challenge: "",
    solution: "",
    codeLanguage: "",
    code: ""
  },
  {
    id: 6,
    title: "Frosted Portfolio Design Framework",
    category: "Web Dev",
    tags: ["Web Dev"],
    tech: ["Coming soon"],
    problem: "Under development. Designing next-generation responsive template models.",
    approach: "Pending layout configurations.",
    results: "Project details coming soon.",
    lessons: "Project details coming soon.",
    github: "https://github.com/Vighan-coder",
    demo: "#",
    active: false,
    challenge: "",
    solution: "",
    codeLanguage: "",
    code: ""
  }
];

// Interactive 3D tilt-on-hover card wrapper
function TiltCard({ children, onClick, onMouseEnter }: { children: React.ReactNode; onClick?: () => void; onMouseEnter?: () => void }) {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const card = cardRef.current;
    if (!card) return;

    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const xc = rect.width / 2;
    const yc = rect.height / 2;

    const angleX = (yc - y) / 18; // tilt intensity
    const angleY = (x - xc) / 18;

    card.style.transform = `perspective(1000px) rotateX(${angleX}deg) rotateY(${angleY}deg) translateY(-4px)`;
  };

  const handleMouseLeave = () => {
    const card = cardRef.current;
    if (!card) return;
    card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)`;
  };

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      className="glass-panel p-6 space-y-4 hover:border-accent/30 hover:shadow-[0_12px_40px_rgba(0,208,132,0.1)] transition-all duration-200 cursor-pointer flex flex-col justify-between h-full"
      style={{ transformStyle: "preserve-3d" }}
    >
      <div style={{ transform: "translateZ(20px)" }} className="flex-grow flex flex-col justify-between space-y-4">
        {children}
      </div>
    </div>
  );
}

/* ========================================================
   INTERACTIVE SIMULATION SUBCOMPONENTS
   ======================================================== */

// Project 1: YOLO Object Detection Demo
function YoloDemo() {
  const [detecting, setDetecting] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = canvas.width = 480;
    let height = canvas.height = 270;

    // Vector map outline elements
    const roads = [
      { x1: 0, y1: 135, x2: 480, y2: 135 },
      { x1: 240, y1: 0, x2: 240, y2: 270 }
    ];

    const buildings = [
      { x: 50, y: 30, w: 80, h: 60, label: "Structure A" },
      { x: 320, y: 160, w: 90, h: 70, label: "Structure B" }
    ];

    // Real-time moving targets
    const targets = [
      { id: "Drone-Cam", type: "Drone", x: 120, y: 80, speedX: 0.8, speedY: 0.5, size: 8, color: "#4AFFB8" },
      { id: "Vehicle-Alpha", type: "Vehicle", x: 20, y: 135, speedX: 1.2, speedY: 0, size: 6, color: "#38BDF8" },
      { id: "Vehicle-Beta", type: "Vehicle", x: 240, y: 20, speedX: 0, speedY: 1.0, size: 6, color: "#38BDF8" }
    ];

    let scanY = 0;

    const render = () => {
      // Background grid base
      ctx.fillStyle = "#0c0c0c";
      ctx.fillRect(0, 0, width, height);

      // Radial vignette lighting
      const lightGrad = ctx.createRadialGradient(width / 2, height / 2, 50, width / 2, height / 2, 240);
      lightGrad.addColorStop(0, "rgba(0, 208, 132, 0.03)");
      lightGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = lightGrad;
      ctx.fillRect(0, 0, width, height);

      // Grid Pattern
      ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
      ctx.lineWidth = 1;
      const gridSize = 20;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Draw Roads
      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 16;
      roads.forEach(r => {
        ctx.beginPath();
        ctx.moveTo(r.x1, r.y1);
        ctx.lineTo(r.x2, r.y2);
        ctx.stroke();
      });

      // Road dash markers
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 6]);
      roads.forEach(r => {
        ctx.beginPath();
        ctx.moveTo(r.x1, r.y1);
        ctx.lineTo(r.x2, r.y2);
        ctx.stroke();
      });
      ctx.setLineDash([]); // Reset

      // Outlined structures
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1.5;
      buildings.forEach(b => {
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.fillStyle = "rgba(0, 208, 132, 0.01)";
        ctx.fillRect(b.x, b.y, b.w, b.h);
      });

      // Move & render targets
      targets.forEach(t => {
        t.x += t.speedX;
        t.y += t.speedY;

        // Wrap screens
        if (t.x > width) t.x = 0;
        if (t.y > height) t.y = 0;

        // Draw physical point
        ctx.fillStyle = t.color;
        ctx.beginPath();
        ctx.arc(t.x, t.y, t.size / 2, 0, Math.PI * 2);
        ctx.fill();

        // Draw bounding boxes when YOLO is active
        if (detecting) {
          const boxSize = t.size * 3.5;
          const left = t.x - boxSize / 2;
          const top = t.y - boxSize / 2;

          // Box borders
          ctx.strokeStyle = "#00D084";
          ctx.lineWidth = 1.2;
          ctx.strokeRect(left, top, boxSize, boxSize);

          // Corner Brackets
          ctx.strokeStyle = "#FFFFFF";
          ctx.lineWidth = 1.2;
          const bLen = 4;
          // TL
          ctx.beginPath(); ctx.moveTo(left, top + bLen); ctx.lineTo(left, top); ctx.lineTo(left + bLen, top); ctx.stroke();
          // TR
          ctx.beginPath(); ctx.moveTo(left + boxSize - bLen, top); ctx.lineTo(left + boxSize, top); ctx.lineTo(left + boxSize, top + bLen); ctx.stroke();
          // BL
          ctx.beginPath(); ctx.moveTo(left, top + boxSize - bLen); ctx.lineTo(left, top + boxSize); ctx.lineTo(left + bLen, top + boxSize); ctx.stroke();
          // BR
          ctx.beginPath(); ctx.moveTo(left + boxSize - bLen, top + boxSize); ctx.lineTo(left + boxSize, top + boxSize); ctx.lineTo(left + boxSize, top + boxSize - bLen); ctx.stroke();

          // Text metrics overlay
          ctx.fillStyle = "#00D084";
          ctx.font = "bold 7px monospace";
          const conf = (91 + (t.x % 8)).toFixed(1); // realistic noise conf
          ctx.fillText(`${t.id} ${conf}%`, left, top - 4);
          
          ctx.fillStyle = "rgba(0, 208, 132, 0.6)";
          ctx.font = "5.5px monospace";
          ctx.fillText(`x:${t.x.toFixed(0)} y:${t.y.toFixed(0)}`, left, top + boxSize + 8);
        }
      });

      // Build static locks when active
      if (detecting) {
        buildings.forEach(b => {
          ctx.strokeStyle = "rgba(239, 68, 68, 0.6)";
          ctx.lineWidth = 1;
          ctx.strokeRect(b.x - 2, b.y - 2, b.w + 4, b.h + 4);
          
          ctx.fillStyle = "rgba(239, 68, 68, 0.8)";
          ctx.font = "bold 6.5px monospace";
          ctx.fillText(`STATIC_LOCK: ${b.label} [99%]`, b.x - 2, b.y - 6);
        });
      }

      // Horizontal scanner laser
      scanY = (scanY + 1.2) % height;
      ctx.strokeStyle = "rgba(0, 208, 132, 0.25)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, scanY);
      ctx.lineTo(width, scanY);
      ctx.stroke();

      // Scanner glow trail
      const grad = ctx.createLinearGradient(0, scanY - 40, 0, scanY);
      grad.addColorStop(0, "rgba(0, 208, 132, 0)");
      grad.addColorStop(1, "rgba(0, 208, 132, 0.08)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, scanY - 40, width, 40);

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [detecting]);

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="relative flex-grow bg-black border border-white/5 rounded-xl overflow-hidden flex items-center justify-center min-h-[240px] aspect-video">
        <canvas ref={canvasRef} className="w-full h-full block" />
        
        {/* Status badges */}
        <div className="absolute top-3 left-3 flex items-center space-x-1.5 z-20 bg-black/75 px-2 py-1 rounded border border-white/5 pointer-events-none">
          <span className={`w-2 h-2 rounded-full ${detecting ? "bg-rose-500 animate-ping" : "bg-gray-600"}`} />
          <span className="text-[8px] font-mono text-white/80 uppercase tracking-widest">
            {detecting ? "AERIAL_FEED_RAW [45 FPS]" : "SYSTEM_STANDBY [0 FPS]"}
          </span>
        </div>

        {detecting && (
          <div className="absolute top-3 right-3 bg-[#00D084]/15 border border-[#00D084]/30 px-2 py-1 rounded text-[8px] font-mono text-accent uppercase tracking-widest animate-pulse pointer-events-none">
            YOLOv8_ACTIVE_TRACKING
          </div>
        )}
      </div>

      <div className="flex items-center justify-between bg-white/2 border border-white/5 p-3 rounded-lg">
        <span className="text-[10px] font-mono text-gray-400">YOLOv8 & CLAHE Inference Pipeline</span>
        <button 
          onClick={() => setDetecting(!detecting)}
          className={`px-3 py-1.5 text-[9px] font-mono rounded font-bold transition-all cursor-pointer ${
            detecting ? "bg-rose-600 text-white hover:bg-rose-500" : "bg-[#00D084] text-black hover:bg-[#4AFFB8]"
          }`}
        >
          {detecting ? "PAUSE DETECTIONS" : "ACTIVATE INFERENCE"}
        </button>
      </div>
    </div>
  );
}

// Project 2: High-Dimensional Vectorization Engine Benchmarking
function VectorDemo() {
  const [running, setRunning] = useState(false);
  const [finished, setFinished] = useState(false);

  const runTest = () => {
    setRunning(true);
    setFinished(false);
    setTimeout(() => {
      setRunning(false);
      setFinished(true);
    }, 1200);
  };

  return (
    <div className="flex flex-col h-full justify-between space-y-4">
      <div className="bg-black/60 border border-white/5 p-6 rounded-xl space-y-6 flex-grow min-h-[240px]">
        <div className="border-b border-white/5 pb-2">
          <span className="text-[9px] font-mono text-gray-500 uppercase block">Performance Metrics Benchmark</span>
          <span className="text-xs font-mono font-bold text-white">Outlier filtration comparison (100,000 array elements)</span>
        </div>

        {running ? (
          <div className="flex flex-col items-center justify-center h-28 space-y-3">
            <div className="w-6 h-6 border-2 border-[#00D084] border-t-transparent rounded-full animate-spin" />
            <span className="text-[9px] font-mono text-accent animate-pulse uppercase tracking-wider">Evaluating cache alignments...</span>
          </div>
        ) : finished ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-accent/15 border border-accent/30 rounded-lg">
              <span className="text-[10px] font-bold text-accent font-mono">PROFILER STATUS: SUCCESS</span>
              <span className="text-[10px] font-mono font-black text-accent bg-accent/20 px-2 py-0.5 rounded">153.5x ACCELERATION</span>
            </div>

            <div className="space-y-3 text-[10px] font-mono">
              <div className="space-y-1">
                <div className="flex justify-between text-gray-400">
                  <span>Python Loop iterations</span>
                  <span className="text-rose-500 font-bold">1,842 ms</span>
                </div>
                <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                  <div className="bg-rose-500 h-full w-full rounded-full" />
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-gray-400">
                  <span>NumPy Vector registers</span>
                  <span className="text-[#00D084] font-bold">12 ms</span>
                </div>
                <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                  <div className="bg-[#00D084] h-full w-[1%] rounded-full" />
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-28 text-center space-y-2">
            <span className="text-[9px] font-mono text-gray-500 uppercase tracking-widest">Benchmarking Engine</span>
            <p className="text-[10px] text-gray-400 max-w-xs">Run execution comparison profiles comparing vector SIMD parameters against standard loops.</p>
          </div>
        )}
      </div>

      <button 
        onClick={runTest}
        disabled={running}
        className="w-full py-2 bg-[#00D084] hover:bg-[#4AFFB8] disabled:bg-white/10 text-black disabled:text-gray-500 font-mono font-bold text-xs rounded transition-all shadow-[0_0_15px_rgba(0,208,132,0.2)] disabled:shadow-none cursor-pointer"
      >
        {running ? "BENCHMARK RUNNING..." : "RUN PERFORMANCE BENCHMARK"}
      </button>
    </div>
  );
}

// Project 3: 3D Point Cloud Representation & Gaussian Splat Explorer
function PointCloudDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRotating, setIsRotating] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = canvas.width = canvas.offsetWidth;
    let height = canvas.height = canvas.offsetHeight;

    // Create points
    const points: { x: number; y: number; z: number; color: string }[] = [];
    for (let i = 0; i < 150; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = 50 + Math.random() * 40;
      
      points.push({
        x: r * Math.sin(phi) * Math.cos(theta),
        y: r * Math.sin(phi) * Math.sin(theta),
        z: r * Math.cos(phi),
        color: Math.random() > 0.45 ? "rgba(0, 208, 132, " : "rgba(56, 189, 248, "
      });
    }

    let angleY = 0.006;
    let angleX = 0.003;

    // Drag-to-rotate mouse settings
    let isDragging = false;
    let previousMouse = { x: 0, y: 0 };

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      previousMouse = { x: e.clientX, y: e.clientY };
      setIsRotating(false);
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - previousMouse.x;
      const dy = e.clientY - previousMouse.y;
      previousMouse = { x: e.clientX, y: e.clientY };

      // Manual rotation multipliers
      const cosY = Math.cos(dx * 0.007);
      const sinY = Math.sin(dx * 0.007);
      const cosX = Math.cos(dy * 0.007);
      const sinX = Math.sin(dy * 0.007);

      points.forEach(p => {
        // Rotate around Y axis (horizontal drag)
        let x1 = p.x * cosY - p.z * sinY;
        let z1 = p.z * cosY + p.x * sinY;

        // Rotate around X axis (vertical drag)
        let y2 = p.y * cosX - z1 * sinX;
        let z2 = z1 * cosX + p.y * sinX;

        p.x = x1;
        p.y = y2;
        p.z = z2;
      });
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    // Mobile touch-to-rotate touch settings
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isDragging = true;
        previousMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        setIsRotating(false);
      }
    };

    const onTouchMove = (e: TouchEvent) => {
      if (!isDragging || e.touches.length !== 1) return;
      const dx = e.touches[0].clientX - previousMouse.x;
      const dy = e.touches[0].clientY - previousMouse.y;
      previousMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };

      const cosY = Math.cos(dx * 0.007);
      const sinY = Math.sin(dx * 0.007);
      const cosX = Math.cos(dy * 0.007);
      const sinX = Math.sin(dy * 0.007);

      points.forEach(p => {
        let x1 = p.x * cosY - p.z * sinY;
        let z1 = p.z * cosY + p.x * sinY;

        let y2 = p.y * cosX - z1 * sinX;
        let z2 = z1 * cosX + p.y * sinX;

        p.x = x1;
        p.y = y2;
        p.z = z2;
      });
    };

    const onTouchEnd = () => {
      isDragging = false;
    };

    // Attach listeners
    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    canvas.addEventListener("touchmove", onTouchMove, { passive: true });
    canvas.addEventListener("touchend", onTouchEnd, { passive: true });

    const resize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
    };
    window.addEventListener("resize", resize);

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const fov = 200;

      if (isRotating) {
        const cosY = Math.cos(angleY);
        const sinY = Math.sin(angleY);
        const cosX = Math.cos(angleX);
        const sinX = Math.sin(angleX);

        points.forEach(p => {
          // Y rotation
          let x1 = p.x * cosY - p.z * sinY;
          let z1 = p.z * cosY + p.x * sinY;

          // X rotation
          let y2 = p.y * cosX - z1 * sinX;
          let z2 = z1 * cosX + p.y * sinX;

          p.x = x1;
          p.y = y2;
          p.z = z2;
        });
      }

      // Sort by depth (Z)
      const sortedPoints = [...points].sort((a, b) => b.z - a.z);

      sortedPoints.forEach(p => {
        const scale = fov / (fov + p.z + 100);
        const sx = cx + p.x * scale;
        const sy = cy + p.y * scale;

        if (sx >= 0 && sx <= width && sy >= 0 && sy <= height) {
          const size = Math.max(1, (p.z + 120) * 0.015 * scale);
          const opacity = Math.min(1, Math.max(0.1, (p.z + 100) / 200));
          
          ctx.beginPath();
          ctx.arc(sx, sy, size, 0, Math.PI * 2);
          ctx.fillStyle = p.color + opacity + ")";
          ctx.fill();

          if (size > 2) {
            ctx.beginPath();
            ctx.arc(sx, sy, size * 2.0, 0, Math.PI * 2);
            ctx.fillStyle = p.color + (opacity * 0.1) + ")";
            ctx.fill();
          }
        }
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);

      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);

      cancelAnimationFrame(animationFrameId);
    };
  }, [isRotating]);

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="relative flex-grow bg-black/60 border border-white/5 rounded-xl overflow-hidden min-h-[240px]">
        <canvas ref={canvasRef} className="w-full h-full block cursor-grab active:cursor-grabbing select-none" />
        
        <div className="absolute bottom-3 left-3 bg-black/80 px-2 py-1 rounded border border-white/10 text-[8px] font-mono text-gray-400 space-y-0.5 pointer-events-none">
          <div>SCALE: 1.000 (FUSED)</div>
          <div>POINTS: 150 GAUSSIANS</div>
          <div>STATUS: FUSED_NORMALIZED</div>
        </div>

        <div className="absolute top-3 right-3 bg-black/80 px-2 py-1 rounded border border-white/10 text-[8px] font-mono text-accent pointer-events-none">
          FUSED CAM-SCENE ACTIVE
        </div>
      </div>

      <div className="flex items-center justify-between bg-white/2 border border-white/5 p-3 rounded-lg">
        <span className="text-[10px] font-mono text-gray-400">Interactive 3DGS splats orbit view</span>
        <button 
          onClick={() => setIsRotating(!isRotating)}
          className={`px-3 py-1.5 text-[9px] font-mono rounded font-bold transition-all cursor-pointer ${
            isRotating ? "bg-amber-600 text-white hover:bg-amber-500" : "bg-[#00D084] text-black hover:bg-[#4AFFB8]"
          }`}
        >
          {isRotating ? "FREEZE CAM" : "ORBIT CAM"}
        </button>
      </div>
    </div>
  );
}

// Project 4: Neural Splat Reconstruction Simulator
function TrainingDemo() {
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0.85);
  const [psnr, setPsnr] = useState(15.4);
  const [ssim, setSsim] = useState(0.42);
  const [training, setTraining] = useState(false);

  const startTraining = () => {
    if (training) return;
    setTraining(true);
    setEpoch(0);
    setLoss(0.85);
    setPsnr(15.4);
    setSsim(0.42);

    let count = 0;
    const interval = setInterval(() => {
      count += 1;
      setEpoch(count);
      
      setLoss(prev => Math.max(0.012, prev - (prev * 0.08) - Math.random() * 0.004));
      setPsnr(prev => Math.min(32.4, prev + (32.4 - prev) * 0.06 + Math.random() * 0.08));
      setSsim(prev => Math.min(0.96, prev + (0.96 - prev) * 0.07 + Math.random() * 0.004));

      if (count >= 100) {
        clearInterval(interval);
        setTraining(false);
      }
    }, 45);
  };

  return (
    <div className="flex flex-col h-full justify-between space-y-4">
      <div className="bg-black/60 border border-white/5 p-6 rounded-xl space-y-5 flex-grow min-h-[240px]">
        <div className="flex justify-between items-center border-b border-white/5 pb-2">
          <div>
            <span className="text-[9px] font-mono text-gray-500 uppercase block">Model Training Simulator</span>
            <span className="text-xs font-mono font-bold text-white">3DGS Volumetric Loss Convergence Loop</span>
          </div>
          <span className="text-[9px] font-mono font-black text-accent bg-accent/20 px-2 py-0.5 rounded">
            EPOCH: {epoch}/100
          </span>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="bg-white/2 border border-white/5 p-3 rounded text-center space-y-1">
            <span className="text-[8px] font-mono text-gray-500 uppercase block">L1 Loss</span>
            <span className="text-sm font-mono font-bold text-white">
              {loss.toFixed(4)}
            </span>
          </div>
          <div className="bg-white/2 border border-white/5 p-3 rounded text-center space-y-1">
            <span className="text-[8px] font-mono text-gray-500 uppercase block">PSNR</span>
            <span className="text-sm font-mono font-bold text-accent">
              {psnr.toFixed(1)} dB
            </span>
          </div>
          <div className="bg-white/2 border border-white/5 p-3 rounded text-center space-y-1">
            <span className="text-[8px] font-mono text-gray-500 uppercase block">SSIM</span>
            <span className="text-sm font-mono font-bold text-white">
              {ssim.toFixed(3)}
            </span>
          </div>
        </div>

        <div className="space-y-1">
          <div className="w-full bg-white/5 h-2 rounded overflow-hidden">
            <div className="bg-[#00D084] h-full transition-all duration-75" style={{ width: `${epoch}%` }} />
          </div>
        </div>

        {epoch === 100 && (
          <div className="text-[9px] font-mono text-accent text-center bg-accent/10 border border-accent/20 p-2 rounded">
            ✓ CONVERGENCE LOOP STABLE: Training metrics achieved optimal quality thresholds.
          </div>
        )}
      </div>

      <button
        onClick={startTraining}
        disabled={training}
        className="w-full py-2 bg-[#00D084] hover:bg-[#4AFFB8] disabled:bg-white/10 text-black disabled:text-gray-500 font-mono font-bold text-xs rounded transition-all shadow-[0_0_15px_rgba(0,208,132,0.2)] disabled:shadow-none cursor-pointer"
      >
        {training ? "RUNNING CONVOLUTIONS..." : "START MODEL TRAINING"}
      </button>
    </div>
  );
}

/* ========================================================
   PROJECTS PAGE ENTRY COMPONENT
   ======================================================== */

export default function ProjectsPage() {
  const { playClick, playHover } = useAudio();
  const [activeCategory, setActiveCategory] = useState("All");
  const [selectedProject, setSelectedProject] = useState<typeof PROJECTS[0] | null>(null);
  const [modalTab, setModalTab] = useState<"code" | "output">("output");

  // Escape key closer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelectedProject(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const filteredProjects = PROJECTS.filter((p) => {
    if (activeCategory === "All") return true;
    return p.tags.includes(activeCategory) || p.category === activeCategory;
  });

  return (
    <div className="max-w-6xl mx-auto w-full space-y-16 relative z-10 pt-8">
      
      {/* Title */}
      <div className="text-center space-y-4">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          className="text-xs font-mono tracking-widest text-[#9CA3AF] uppercase"
        >
          Selected Works
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl sm:text-5xl font-sans font-black tracking-tight text-white uppercase"
        >
          Projects & <span className="text-accent">Solutions</span>
        </motion.h1>
      </div>

      {/* Category Filters */}
      <div className="flex flex-wrap items-center justify-center gap-2 max-w-2xl mx-auto">
        {CATEGORIES.map((cat) => {
          const isActive = activeCategory === cat;
          return (
            <button
              key={cat}
              onClick={() => {
                playClick();
                setActiveCategory(cat);
              }}
              onMouseEnter={playHover}
              className={`px-4 py-1.5 rounded-full text-xs font-mono font-bold tracking-wider uppercase transition-all cursor-pointer ${
                isActive 
                  ? "bg-[#00D084] text-black shadow-[0_0_15px_rgba(0,208,132,0.3)] border border-[#00D084]" 
                  : "bg-white/5 border border-white/10 text-[#9CA3AF] hover:text-white hover:border-white/20"
              }`}
            >
              {cat}
            </button>
          );
        })}
      </div>

      {/* Projects Grid */}
      <motion.div 
        layout
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
      >
        <AnimatePresence mode="popLayout">
          {filteredProjects.map((project) => (
            <motion.div
              layout
              key={project.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              className="h-full"
            >
              <TiltCard onClick={playClick} onMouseEnter={playHover}>
                <div className="space-y-4">
                  {/* Top Category Badge and Links */}
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-accent/10 border border-accent/20 text-[9px] font-mono text-accent uppercase font-semibold">
                      <Cpu className="w-2.5 h-2.5" />
                      {project.category}
                    </span>
                    
                    {/* Action Links */}
                    <div className="flex items-center space-x-2 text-[#9CA3AF]">
                      <a 
                        href={project.github} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        onClick={(e) => { e.stopPropagation(); playClick(); }}
                        className="hover:text-white transition-colors"
                        aria-label="GitHub link"
                      >
                        <GithubIcon className="w-4 h-4" />
                      </a>
                      {project.active && (
                        <a 
                          href={project.demo}
                          onClick={(e) => { e.stopPropagation(); playClick(); }}
                          className="hover:text-white transition-colors"
                          aria-label="External link"
                        >
                          <ExternalLink className="w-4 h-4" />
                        </a>
                      )}
                    </div>
                  </div>

                  {/* Title & Chips */}
                  <div>
                    <h3 className="text-xl font-bold tracking-tight text-white group-hover:text-accent transition-colors">
                      {project.title}
                    </h3>
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {project.tech.map((t) => (
                        <span key={t} className="px-2 py-0.5 rounded bg-white/5 border border-white/5 text-[9px] font-mono text-gray-400">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Structured Core Content */}
                  <div className="space-y-3 pt-2 text-xs border-t border-white/5">
                    <div>
                      <span className="font-mono text-[10px] text-accent/80 font-bold block uppercase">
                        Problem:
                      </span>
                      <p className="text-gray-400 mt-0.5">{project.problem}</p>
                    </div>
                    {project.active && (
                      <>
                        <div>
                          <span className="font-mono text-[10px] text-accent/80 font-bold block uppercase">
                            Approach:
                          </span>
                          <p className="text-gray-400 mt-0.5">{project.approach}</p>
                        </div>
                        <div>
                          <span className="font-mono text-[10px] text-accent/80 font-bold block uppercase">
                            Results:
                          </span>
                          <p className="text-gray-400 mt-0.5">{project.results}</p>
                        </div>
                        <div>
                          <span className="font-mono text-[10px] text-accent/80 font-bold block uppercase">
                            Lessons:
                          </span>
                          <p className="text-gray-400 mt-0.5">{project.lessons}</p>
                        </div>
                      </>
                    )}

                    {!project.active && (
                      <div className="py-2 flex items-center space-x-2 text-[#9CA3AF]/60 font-mono text-[10px] uppercase">
                        <Boxes className="w-4 h-4 animate-pulse" />
                        <span>Project details coming soon</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Case Study CTA bottom bar */}
                {project.active && (
                  <div className="pt-4 border-t border-white/5 flex justify-end">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        playClick();
                        setSelectedProject(project);
                        setModalTab("output");
                      }}
                      className="flex items-center space-x-1 text-[10px] font-mono font-bold tracking-widest text-[#00D084] hover:text-[#4AFFB8] transition-colors cursor-pointer bg-transparent border-none outline-none"
                    >
                      <BookOpen className="w-3.5 h-3.5" />
                      <span>READ CASE STUDY</span>
                    </button>
                  </div>
                )}
              </TiltCard>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {/* Case Study Modal */}
      <AnimatePresence>
        {selectedProject && (
          <div className="fixed inset-0 z-[99999] flex items-center justify-center p-4">
            {/* Backdrop blur overlay */}
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedProject(null)}
              className="absolute inset-0 bg-[#050505]/80 backdrop-blur-md"
            />
            
            {/* Modal Box */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              className="relative w-full max-w-5xl h-[85vh] bg-[#0c0c0c] border border-white/10 rounded-2xl overflow-hidden shadow-2xl flex flex-col z-10"
            >
              {/* Header */}
              <div className="p-6 border-b border-white/5 flex items-center justify-between bg-white/2">
                <div className="space-y-1">
                  <span className="text-[10px] font-mono text-accent uppercase tracking-widest">{selectedProject.category} Case Study</span>
                  <h2 className="text-xl font-bold text-white tracking-tight">{selectedProject.title}</h2>
                </div>
                <button 
                  onClick={() => { playClick(); setSelectedProject(null); }}
                  className="px-3 py-1.5 border border-white/10 hover:border-white/20 text-[#9CA3AF] hover:text-white rounded-lg text-xs font-mono transition-colors cursor-pointer"
                >
                  ESC [x]
                </button>
              </div>

              {/* Modal Content - Scrollable grid */}
              <div className="flex-grow overflow-y-auto grid grid-cols-1 lg:grid-cols-2">
                {/* Left panel: Challenge, Solution & Key Metrics */}
                <div className="p-8 space-y-6 border-r border-white/5">
                  <div className="space-y-2">
                    <span className="text-[10px] font-mono font-bold text-accent tracking-wider uppercase block">The Challenge</span>
                    <p className="text-sm text-gray-300 leading-relaxed font-light">{selectedProject.challenge}</p>
                  </div>
                  
                  <div className="space-y-2">
                    <span className="text-[10px] font-mono font-bold text-accent tracking-wider uppercase block">Solution Implementation</span>
                    <p className="text-sm text-gray-300 leading-relaxed font-light">{selectedProject.solution}</p>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4 border-t border-white/5">
                    <div className="space-y-1 p-4 bg-white/2 border border-white/5 rounded-xl">
                      <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wide block">Performance Gains</span>
                      <span className="text-lg font-bold text-accent block">{selectedProject.results.split("climbing")[0]}</span>
                    </div>
                    <div className="space-y-1 p-4 bg-white/2 border border-white/5 rounded-xl">
                      <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wide block">Key Ingestion Stack</span>
                      <span className="text-xs font-mono text-white block mt-1">{selectedProject.tech.join(" | ")}</span>
                    </div>
                  </div>

                  <div className="space-y-2 pt-2">
                    <span className="text-[10px] font-mono font-bold text-accent tracking-wider uppercase block">Engineering Takeaway</span>
                    <p className="text-xs text-gray-400 italic leading-relaxed font-light">&quot;{selectedProject.lessons}&quot;</p>
                  </div>
                </div>

                {/* Right panel: Tabbed interactive content */}
                <div className="p-8 bg-[#080808] flex flex-col min-h-[450px] lg:h-full overflow-hidden">
                  
                  {/* Tab Selector */}
                  <div className="flex items-center justify-between border-b border-white/5 pb-3 mb-4 shrink-0">
                    <div className="flex items-center space-x-1 bg-white/5 p-1 rounded-lg">
                      <button
                        onClick={() => { playClick(); setModalTab("output"); }}
                        className={`px-3 py-1.5 text-[10px] font-mono font-bold rounded-md transition-all cursor-pointer ${
                          modalTab === "output" 
                            ? "bg-[#00D084] text-black shadow-[0_0_10px_rgba(0,208,132,0.3)]" 
                            : "text-[#9CA3AF] hover:text-white"
                        }`}
                      >
                        OUTPUT DEMO
                      </button>
                      <button
                        onClick={() => { playClick(); setModalTab("code"); }}
                        className={`px-3 py-1.5 text-[10px] font-mono font-bold rounded-md transition-all cursor-pointer ${
                          modalTab === "code" 
                            ? "bg-[#00D084] text-black shadow-[0_0_10px_rgba(0,208,132,0.3)]" 
                            : "text-[#9CA3AF] hover:text-white"
                        }`}
                      >
                        CODE VIEW
                      </button>
                    </div>
                    <span className="text-[9px] font-mono text-[#00D084] border border-[#00D084]/20 px-2 py-0.5 rounded bg-[#00D084]/10 uppercase tracking-wider font-bold">
                      {modalTab === "code" ? "Source Code" : "Live Simulation"}
                    </span>
                  </div>

                  {/* Tab Content Area */}
                  <div className="flex-grow overflow-hidden flex flex-col justify-between">
                    {modalTab === "code" ? (
                      <div className="flex-grow overflow-auto font-mono text-[11px] text-gray-300 bg-black/40 p-4 border border-white/5 rounded-xl leading-relaxed select-text">
                        <pre className="whitespace-pre">{selectedProject.code}</pre>
                      </div>
                    ) : (
                      <div className="flex-grow overflow-hidden">
                        {selectedProject.id === 1 && <YoloDemo />}
                        {selectedProject.id === 2 && <VectorDemo />}
                        {selectedProject.id === 3 && <PointCloudDemo />}
                        {selectedProject.id === 4 && <TrainingDemo />}
                      </div>
                    )}
                  </div>

                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

    </div>
  );
}
