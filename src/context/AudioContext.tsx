"use client";

import React, { createContext, useContext, useEffect, useState, useRef } from "react";

interface AudioContextType {
  isMuted: boolean;
  volume: number;
  toggleMute: () => void;
  setVolume: (v: number) => void;
  playClick: () => void;
  playHover: () => void;
  playTransition: () => void;
}

const AudioContext = createContext<AudioContextType | undefined>(undefined);

interface ExtendedOscillatorNode extends OscillatorNode {
  _lfoInterval?: NodeJS.Timeout;
}

export const AudioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Initialize to server-safe defaults to prevent hydration mismatch
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [volume, setVolumeState] = useState<number>(0.5);

  // Load persisted audio settings after client mount
  useEffect(() => {
    try {
      const savedMute = localStorage.getItem("portfolio-audio-muted");
      const savedVolume = localStorage.getItem("portfolio-audio-volume");
      
      const t = setTimeout(() => {
        if (savedMute !== null) {
          setIsMuted(savedMute === "true");
        }
        if (savedVolume !== null) {
          setVolumeState(parseFloat(savedVolume));
        }
      }, 0);
      return () => clearTimeout(t);
    } catch (e) {
      console.warn("localStorage load audio settings error:", e);
    }
  }, []);
  
  // Web Audio API refs
  const audioCtxRef = useRef<AudioContext | null>(null);
  const droneNodesRef = useRef<{
    oscillators: OscillatorNode[];
    gains: GainNode[];
    filter: BiquadFilterNode;
    masterGain: GainNode;
  } | null>(null);
  const volumeRef = useRef(0.5);
  const isMutedRef = useRef(true);

  // Initialize Audio Context on user gesture
  const initAudioCtx = () => {
    if (!audioCtxRef.current) {
      const AudioCtxClass = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (AudioCtxClass) {
        audioCtxRef.current = new AudioCtxClass();
      }
    }
    if (audioCtxRef.current && audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume();
    }
  };

  // Generative Ambient Synth Drone Music: Multiple detuned oscillators running in parallel
  // Generative Ambient Synth Drone Music is disabled
  const startDrone = () => {};
  const stopDrone = () => {};

  // Synchronize state values with refs for async event callbacks
  useEffect(() => {
    volumeRef.current = volume;
  }, [volume]);

  useEffect(() => {
    isMutedRef.current = isMuted;
    if (!isMuted) {
      initAudioCtx();
      startDrone();
    } else {
      stopDrone();
    }
  }, [isMuted]);

  const toggleMute = () => {
    setIsMuted((prev) => {
      const newVal = !prev;
      try {
        localStorage.setItem("portfolio-audio-muted", String(newVal));
      } catch (e) {
        console.warn("localStorage set mute error:", e);
      }
      return newVal;
    });
  };

  const setVolume = (v: number) => {
    setVolumeState(v);
    try {
      localStorage.setItem("portfolio-audio-volume", String(v));
    } catch (e) {
      console.warn("localStorage set volume error:", e);
    }
    if (droneNodesRef.current) {
      // Set volume with a short ramp to avoid clicks
      const targetGain = isMutedRef.current ? 0 : v * 0.15; // drone volume scaling
      const ctx = audioCtxRef.current;
      if (ctx) {
        droneNodesRef.current.masterGain.gain.setValueAtTime(
          droneNodesRef.current.masterGain.gain.value,
          ctx.currentTime
        );
        droneNodesRef.current.masterGain.gain.linearRampToValueAtTime(
          targetGain,
          ctx.currentTime + 0.1
        );
      }
    }
  };

  // Synthesize UI Click Sound: Short decaying organic pluck
  const playClick = () => {
    try {
      initAudioCtx();
      const ctx = audioCtxRef.current;
      if (!ctx || isMutedRef.current) return;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      osc.type = "sine";
      osc.frequency.setValueAtTime(150, ctx.currentTime); // Low fundamental frequency
      osc.frequency.exponentialRampToValueAtTime(60, ctx.currentTime + 0.15); // Slide down for punch

      filter.type = "lowpass";
      filter.frequency.setValueAtTime(600, ctx.currentTime);
      filter.frequency.exponentialRampToValueAtTime(100, ctx.currentTime + 0.15);

      gain.gain.setValueAtTime(volumeRef.current * 0.4, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.16);
    } catch (e) {
      console.warn("Audio playClick error", e);
    }
  };

  // Synthesize UI Hover Sound: Soft, subtle sine sweep
  const playHover = () => {
    try {
      initAudioCtx();
      const ctx = audioCtxRef.current;
      if (!ctx || isMutedRef.current) return;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      // Subtle pitch rise
      osc.frequency.setValueAtTime(320, ctx.currentTime);
      osc.frequency.linearRampToValueAtTime(350, ctx.currentTime + 0.08);

      gain.gain.setValueAtTime(0.001, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(volumeRef.current * 0.06, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.09);
    } catch (e) {
      console.warn("Audio playHover error", e);
    }
  };

  // Synthesize Page Transition: Volumetric wind/sweep sound
  const playTransition = () => {
    try {
      initAudioCtx();
      const ctx = audioCtxRef.current;
      if (!ctx || isMutedRef.current) return;

      // Synthesize noise for wind sound
      const bufferSize = ctx.sampleRate * 0.8; // 0.8 seconds duration
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }

      const noiseNode = ctx.createBufferSource();
      noiseNode.buffer = buffer;

      const filter = ctx.createBiquadFilter();
      filter.type = "bandpass";
      // Sweep bandpass frequency from 100Hz to 1200Hz back to 200Hz
      filter.frequency.setValueAtTime(100, ctx.currentTime);
      filter.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.3);
      filter.frequency.exponentialRampToValueAtTime(200, ctx.currentTime + 0.8);
      filter.Q.setValueAtTime(5, ctx.currentTime);

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.001, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(volumeRef.current * 0.15, ctx.currentTime + 0.25);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.8);

      noiseNode.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      noiseNode.start(ctx.currentTime);
      noiseNode.stop(ctx.currentTime + 0.8);
    } catch (e) {
      console.warn("Audio playTransition error", e);
    }
  };

  return (
    <AudioContext.Provider
      value={{
        isMuted,
        volume,
        toggleMute,
        setVolume,
        playClick,
        playHover,
        playTransition,
      }}
    >
      {children}
    </AudioContext.Provider>
  );
};

export const useAudio = () => {
  const context = useContext(AudioContext);
  if (!context) {
    throw new Error("useAudio must be used within an AudioProvider");
  }
  return context;
};
