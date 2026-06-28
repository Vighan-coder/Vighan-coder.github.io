"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { 
  Sun, 
  Moon, 
  Volume2, 
  VolumeX, 
  Menu, 
  X, 
  FileText
} from "lucide-react";

const NAV_ITEMS = [
  { name: "Home", href: "/" },
  { name: "About", href: "/about" },
  { name: "Career", href: "/career" },
  { name: "Projects", href: "/projects" },
  { name: "Skills", href: "/skills" },
  { name: "Education", href: "/education" },
  { name: "Contact", href: "/contact" }
];

export default function Navbar() {
  const pathname = usePathname();
  const { isMuted, volume, toggleMute, setVolume, playClick, playHover } = useAudio();
  const { scrollY } = useScroll();

  const [hidden, setHidden] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showVolumeSlider, setShowVolumeSlider] = useState(false);
  
  const applyTheme = (t: "dark" | "light") => {
    if (typeof window === "undefined") return;
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    
    if (t === "light") {
      root.classList.add("light");
    } else {
      root.classList.add("dark");
    }
  };

  // Theme state: dark | light
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  // Load persisted theme after client mount to prevent hydration mismatch
  useEffect(() => {
    try {
      const savedTheme = localStorage.getItem("portfolio-theme");
      if (savedTheme === "dark" || savedTheme === "light") {
        const t = setTimeout(() => setTheme(savedTheme), 0);
        return () => clearTimeout(t);
      }
    } catch (e) {
      console.warn("localStorage get theme error:", e);
    }
  }, []);

  // Apply theme when state changes
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const cycleTheme = () => {
    playClick();
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    try {
      localStorage.setItem("portfolio-theme", nextTheme);
    } catch (e) {
      console.warn("localStorage set theme error:", e);
    }
  };

  // Hide Navbar when scrolling down, show when scrolling up
  useMotionValueEvent(scrollY, "change", (latest) => {
    const previous = scrollY.getPrevious() ?? 0;
    if (latest > previous && latest > 120) {
      setHidden(true);
    } else {
      setHidden(false);
    }
  });

  return (
    <>
      <motion.nav
        variants={{
          visible: { y: 0, opacity: 1 },
          hidden: { y: "-120%", opacity: 0 }
        }}
        animate={hidden ? "hidden" : "visible"}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="fixed top-4 left-1/2 -translate-x-1/2 w-[92%] max-w-7xl z-50 glass-panel border border-white/10 dark:border-white/5 py-3 px-6 flex items-center justify-between shadow-2xl"
      >
        {/* Brand Logo */}
        <Link 
          href="/" 
          onClick={playClick}
          onMouseEnter={playHover}
          className="flex items-center space-x-2 text-white font-sans text-lg font-bold tracking-wider hover:text-accent transition-colors"
        >
          <span className="w-8 h-8 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center font-extrabold text-accent text-sm logo-box nav-logo-initials">
            VR
          </span>
          <span className="hidden sm:inline bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent nav-name navbar-brand-text">Vighan Raj Verma</span>
        </Link>

        {/* Center Nav Links (Desktop) */}
        <div className="hidden lg:flex items-center space-x-1 bg-black/20 dark:bg-white/5 py-1 px-1.5 rounded-full border border-white/5">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                onClick={playClick}
                onMouseEnter={playHover}
                className={`relative px-4 py-1.5 rounded-full text-xs font-medium tracking-wide transition-all ${
                  isActive 
                    ? "text-[#050505] font-semibold" 
                    : "text-[#9CA3AF] hover:text-white"
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeNavBackground"
                    transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    className="absolute inset-0 bg-[#00D084] rounded-full shadow-[0_0_12px_rgba(0,208,132,0.4)]"
                  />
                )}
                <span className="relative z-10">{item.name}</span>
              </Link>
            );
          })}
        </div>

        {/* Right Menu Controls */}
        <div className="flex items-center space-x-3">
          {/* Sound Control Slider & Toggle */}
          <div 
            className="relative"
            onMouseEnter={() => setShowVolumeSlider(true)}
            onMouseLeave={() => setShowVolumeSlider(false)}
          >
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => {
                toggleMute();
                playClick();
              }}
              className="p-2 rounded-lg bg-white/5 border border-white/10 text-[#9CA3AF] hover:text-accent hover:border-accent/40 transition-colors"
              aria-label="Toggle Sound"
            >
              {isMuted ? <VolumeX className="w-4.5 h-4.5" /> : <Volume2 className="w-4.5 h-4.5 text-accent" />}
            </motion.button>

            {/* Slider Popup */}
            <AnimatePresence>
              {showVolumeSlider && !isMuted && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="absolute right-0 top-11 p-3 bg-[#0F1115] border border-white/10 rounded-xl flex items-center space-x-2 w-32 shadow-xl z-50"
                >
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={volume}
                    onChange={(e) => setVolume(parseFloat(e.target.value))}
                    className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Theme Toggle Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={cycleTheme}
            className="relative p-1 rounded-full bg-white/5 border border-white/10 flex items-center justify-between w-14 h-8 cursor-pointer focus:outline-none shrink-0"
            aria-label="Toggle Theme"
          >
            {/* Sliding indicator */}
            <motion.div 
              layout
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
              className="absolute w-5 h-5 rounded-full bg-[#00D084] shadow-[0_0_12px_rgba(0,208,132,0.5)]"
              style={{
                left: theme === "light" ? "4px" : "28px"
              }}
            />
            {/* Icons */}
            <span className={`relative z-10 flex items-center justify-center w-5 h-5 transition-colors duration-200 ${
              theme === "light" ? "text-black" : "text-amber-400/80"
            } ml-0.5`}>
              <Sun className="w-3.5 h-3.5" />
            </span>
            <span className={`relative z-10 flex items-center justify-center w-5 h-5 transition-colors duration-200 ${
              theme === "dark" ? "text-black" : "text-blue-300/80"
            } mr-0.5`}>
              <Moon className="w-3.5 h-3.5" />
            </span>
          </motion.button>

          {/* Resume Button */}
          <Link href="/resume" onClick={playClick} onMouseEnter={playHover} className="hidden sm:inline">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center space-x-1 px-4 py-2 text-xs font-mono font-bold tracking-wider text-black bg-[#00D084] rounded-lg hover:bg-[#4AFFB8] hover:shadow-[0_0_15px_rgba(74,255,184,0.4)] transition-all cursor-pointer"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>RESUME</span>
            </motion.button>
          </Link>

          {/* Mobile Menu Toggle Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              playClick();
              setMobileMenuOpen(!mobileMenuOpen);
            }}
            className="p-2 lg:hidden rounded-lg bg-white/5 border border-white/10 text-[#9CA3AF] hover:text-white"
            aria-label="Toggle Mobile Menu"
          >
            {mobileMenuOpen ? <X className="w-4.5 h-4.5" /> : <Menu className="w-4.5 h-4.5" />}
          </motion.button>
        </div>
      </motion.nav>

      {/* Mobile Menu Panel */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="fixed top-20 left-1/2 -translate-x-1/2 w-[92%] z-45 glass-panel border border-white/10 dark:border-white/5 p-6 flex flex-col space-y-4 shadow-2xl lg:hidden max-h-[80vh] overflow-y-auto"
          >
            <div className="flex flex-col space-y-3">
              {NAV_ITEMS.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    onClick={() => {
                      playClick();
                      setMobileMenuOpen(false);
                    }}
                    onMouseEnter={playHover}
                    className={`px-4 py-2.5 rounded-lg text-sm font-medium tracking-wide transition-all ${
                      isActive 
                        ? "text-[#00D084] bg-white/5 font-semibold" 
                        : "text-[#9CA3AF] hover:text-white hover:bg-white/5"
                    }`}
                  >
                    {item.name}
                  </Link>
                );
              })}
            </div>
            
            <hr className="border-white/10" />

            {/* Mobile Resume Link */}
            <Link 
              href="/resume" 
              onClick={() => {
                playClick();
                setMobileMenuOpen(false);
              }}
              onMouseEnter={playHover}
              className="w-full"
            >
              <button className="w-full flex items-center justify-center space-x-2 py-3 text-sm font-mono font-bold tracking-wider text-black bg-[#00D084] rounded-xl">
                <FileText className="w-4 h-4" />
                <span>RESUME</span>
              </button>
            </Link>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
