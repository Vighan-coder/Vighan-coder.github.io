"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAudio } from "@/context/AudioContext";
import { 
  Mail, 
  MapPin, 
  Send, 
  Check, 
  AlertCircle 
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

export default function ContactPage() {
  const { playClick, playHover } = useAudio();

  // Form Fields State
  const [form, setForm] = useState({ name: "", email: "", subject: "", message: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");

  // Floating label helpers
  const [focusedField, setFocusedField] = useState<string | null>(null);

  const validate = () => {
    const tempErrors: Record<string, string> = {};
    if (!form.name.trim()) tempErrors.name = "Name is required.";
    
    if (!form.email.trim()) {
      tempErrors.email = "Email is required.";
    } else if (!/\S+@\S+\.\S+/.test(form.email)) {
      tempErrors.email = "Please provide a valid email address.";
    }
    
    if (!form.subject.trim()) tempErrors.subject = "Subject is required.";
    if (!form.message.trim()) {
      tempErrors.message = "Message is required.";
    } else if (form.message.trim().length < 10) {
      tempErrors.message = "Message must be at least 10 characters.";
    }

    setErrors(tempErrors);
    return Object.keys(tempErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    playClick();
    if (!validate()) return;

    setStatus("submitting");

    try {
      const response = await fetch("https://api.web3forms.com/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          access_key: "c95c0dfc-5b4a-4479-9696-37ad94a33ff6",
          name: form.name,
          email: form.email,
          subject: form.subject,
          message: form.message,
          from_name: "Portfolio Form",
        }),
      });

      const result = await response.json();
      if (result.success) {
        setStatus("success");
        setForm({ name: "", email: "", subject: "", message: "" });
      } else {
        setStatus("error");
      }
    } catch (err) {
      console.error("Transmission error:", err);
      setStatus("error");
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    // Clear error for this field
    if (errors[field]) {
      setErrors((prev) => {
        const copy = { ...prev };
        delete copy[field];
        return copy;
      });
    }
  };

  return (
    <div className="max-w-5xl mx-auto w-full space-y-16 relative z-10 pt-8">
      
      {/* Title */}
      <div className="text-center space-y-4">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          className="text-xs font-mono tracking-widest text-[#9CA3AF] uppercase"
        >
          Get In Touch
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl sm:text-5xl font-sans font-black tracking-tight text-white uppercase"
        >
          Contact & <span className="text-accent">Connect</span>
        </motion.h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-8 items-stretch">
        
        {/* Left Column: Social Directories & Location Info */}
        <div className="md:col-span-2">
          <div className="glass-panel p-6 space-y-6 h-full flex flex-col justify-between">
            <h2 className="text-lg font-bold text-white tracking-tight uppercase border-b border-white/5 pb-3">
              Directories
            </h2>

            {/* Email Card */}
            <a 
              href="mailto:vighnrajverma00893@gmail.com" 
              onClick={playClick}
              onMouseEnter={playHover}
              className="flex items-center space-x-4 p-3 bg-white/2 hover:bg-white/5 border border-white/5 rounded-xl hover:border-accent/30 transition-all group"
            >
              <div className="w-10 h-10 rounded-lg bg-accent/15 border border-accent/25 flex items-center justify-center text-accent group-hover:bg-accent/20 transition-colors">
                <Mail className="w-4.5 h-4.5" />
              </div>
              <div className="space-y-0.5 overflow-hidden">
                <span className="text-[9px] font-mono text-gray-500 uppercase block tracking-wider">EMAIL DIRECTORY</span>
                <span className="text-xs font-mono text-white truncate block">vighnrajverma00893@gmail.com</span>
              </div>
            </a>

            {/* GitHub Card */}
            <a 
              href="https://github.com/Vighan-coder" 
              target="_blank" 
              rel="noopener noreferrer"
              onClick={playClick}
              onMouseEnter={playHover}
              className="flex items-center space-x-4 p-3 bg-white/2 hover:bg-white/5 border border-white/5 rounded-xl hover:border-accent/30 transition-all group"
            >
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-[#9CA3AF] group-hover:text-white transition-colors">
                <GithubIcon className="w-4.5 h-4.5" />
              </div>
              <div className="space-y-0.5">
                <span className="text-[9px] font-mono text-gray-500 uppercase block tracking-wider">REPOSITORY DIRECT</span>
                <span className="text-xs font-mono text-white block">github.com/Vighan-coder</span>
              </div>
            </a>

            {/* LinkedIn Card */}
            <a 
              href="https://www.linkedin.com/in/vighan-raj-verma-4992b2317" 
              target="_blank" 
              rel="noopener noreferrer"
              onClick={playClick}
              onMouseEnter={playHover}
              className="flex items-center space-x-4 p-3 bg-white/2 hover:bg-white/5 border border-white/5 rounded-xl hover:border-accent/30 transition-all group"
            >
              <div className="w-10 h-10 rounded-lg bg-[#0077b5]/10 border border-[#0077b5]/20 flex items-center justify-center text-[#0077b5] group-hover:bg-[#0077b5]/20 transition-colors">
                <LinkedinIcon className="w-4.5 h-4.5" />
              </div>
              <div className="space-y-0.5 overflow-hidden">
                <span className="text-[9px] font-mono text-gray-500 uppercase block tracking-wider">NETWORKING BOARD</span>
                <span className="text-xs font-mono text-white truncate block">Vighan Raj Verma</span>
              </div>
            </a>

            {/* Location Card */}
            <div className="flex items-center space-x-4 p-3 bg-white/2 border border-white/5 rounded-xl">
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-gray-400">
                <MapPin className="w-4.5 h-4.5" />
              </div>
              <div className="space-y-0.5">
                <span className="text-[9px] font-mono text-gray-500 uppercase block tracking-wider">LOCATION NODE</span>
                <span className="text-xs font-mono text-white block">Bhopal, Madhya Pradesh, India</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Contact Form */}
        <div className="md:col-span-3">
          <form onSubmit={handleSubmit} className="glass-panel p-6 sm:p-8 space-y-6 relative">
            <h2 className="text-lg font-bold text-white tracking-tight uppercase border-b border-white/5 pb-3">
              Transmission
            </h2>

            {/* Name Input */}
            <div className="relative">
              <label 
                className={`absolute left-3 transition-all duration-200 pointer-events-none text-xs font-mono ${
                  focusedField === "name" || form.name
                    ? "-top-2 px-1 text-accent bg-[#050505] scale-90"
                    : "top-3.5 text-gray-500 scale-100"
                }`}
              >
                Name
              </label>
              <input
                type="text"
                value={form.name}
                onFocus={() => setFocusedField("name")}
                onBlur={() => setFocusedField(null)}
                onChange={(e) => handleInputChange("name", e.target.value)}
                className={`w-full p-3.5 pt-4 text-xs bg-white/5 border rounded-xl font-mono text-white focus:outline-none transition-colors ${
                  errors.name 
                    ? "border-rose-500/50 focus:border-rose-500" 
                    : "border-white/10 focus:border-accent/60"
                }`}
              />
              {errors.name && (
                <p className="text-[10px] font-mono text-rose-500 mt-1 flex items-center gap-1">
                  <AlertCircle className="w-3.5 h-3.5" /> {errors.name}
                </p>
              )}
            </div>

            {/* Email Input */}
            <div className="relative">
              <label 
                className={`absolute left-3 transition-all duration-200 pointer-events-none text-xs font-mono ${
                  focusedField === "email" || form.email
                    ? "-top-2 px-1 text-accent bg-[#050505] scale-90"
                    : "top-3.5 text-gray-500 scale-100"
                }`}
              >
                Email Address
              </label>
              <input
                type="email"
                value={form.email}
                onFocus={() => setFocusedField("email")}
                onBlur={() => setFocusedField(null)}
                onChange={(e) => handleInputChange("email", e.target.value)}
                className={`w-full p-3.5 pt-4 text-xs bg-white/5 border rounded-xl font-mono text-white focus:outline-none transition-colors ${
                  errors.email 
                    ? "border-rose-500/50 focus:border-rose-500" 
                    : "border-white/10 focus:border-accent/60"
                }`}
              />
              {errors.email && (
                <p className="text-[10px] font-mono text-rose-500 mt-1 flex items-center gap-1">
                  <AlertCircle className="w-3.5 h-3.5" /> {errors.email}
                </p>
              )}
            </div>

            {/* Subject Input */}
            <div className="relative">
              <label 
                className={`absolute left-3 transition-all duration-200 pointer-events-none text-xs font-mono ${
                  focusedField === "subject" || form.subject
                    ? "-top-2 px-1 text-accent bg-[#050505] scale-90"
                    : "top-3.5 text-gray-500 scale-100"
                }`}
              >
                Subject
              </label>
              <input
                type="text"
                value={form.subject}
                onFocus={() => setFocusedField("subject")}
                onBlur={() => setFocusedField(null)}
                onChange={(e) => handleInputChange("subject", e.target.value)}
                className={`w-full p-3.5 pt-4 text-xs bg-white/5 border rounded-xl font-mono text-white focus:outline-none transition-colors ${
                  errors.subject 
                    ? "border-rose-500/50 focus:border-rose-500" 
                    : "border-white/10 focus:border-accent/60"
                }`}
              />
              {errors.subject && (
                <p className="text-[10px] font-mono text-rose-500 mt-1 flex items-center gap-1">
                  <AlertCircle className="w-3.5 h-3.5" /> {errors.subject}
                </p>
              )}
            </div>

            {/* Message Textarea */}
            <div className="relative">
              <label 
                className={`absolute left-3 transition-all duration-200 pointer-events-none text-xs font-mono ${
                  focusedField === "message" || form.message
                    ? "-top-2 px-1 text-accent bg-[#050505] scale-90"
                    : "top-3.5 text-gray-500 scale-100"
                }`}
              >
                Message Body
              </label>
              <textarea
                value={form.message}
                rows={5}
                onFocus={() => setFocusedField("message")}
                onBlur={() => setFocusedField(null)}
                onChange={(e) => handleInputChange("message", e.target.value)}
                className={`w-full p-3.5 pt-4 text-xs bg-white/5 border rounded-xl font-mono text-white focus:outline-none transition-colors resize-none ${
                  errors.message 
                    ? "border-rose-500/50 focus:border-rose-500" 
                    : "border-white/10 focus:border-accent/60"
                }`}
              />
              {errors.message && (
                <p className="text-[10px] font-mono text-rose-500 mt-1 flex items-center gap-1">
                  <AlertCircle className="w-3.5 h-3.5" /> {errors.message}
                </p>
              )}
            </div>

            {/* Submission Status Alert */}
            <AnimatePresence>
              {status === "success" && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="p-4 bg-accent/20 border border-accent/40 rounded-xl flex items-center gap-3 text-xs text-accent font-mono"
                >
                  <Check className="w-5 h-5 shrink-0" />
                  <span>Transmission successfully dispatched. Talk soon!</span>
                </motion.div>
              )}

              {status === "error" && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="p-4 bg-rose-500/20 border border-rose-500/40 rounded-xl flex items-center gap-3 text-xs text-rose-500 font-mono"
                >
                  <AlertCircle className="w-5 h-5 shrink-0" />
                  <span>Transmission failed to dispatch. Verify parameters and retry.</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={status === "submitting"}
              className="w-full py-4 text-xs font-mono font-bold tracking-widest text-black bg-[#00D084] hover:bg-[#4AFFB8] rounded-xl hover:shadow-[0_0_20px_rgba(0,208,132,0.4)] transition-all cursor-pointer flex items-center justify-center space-x-2"
            >
              {status === "submitting" ? (
                <>
                  <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
                  <span>DISPATCHING...</span>
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>DISPATCH TRANSMISSION</span>
                </>
              )}
            </button>

          </form>
        </div>

      </div>

    </div>
  );
}
