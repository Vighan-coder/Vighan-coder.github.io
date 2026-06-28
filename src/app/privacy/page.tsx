"use client";

import React from "react";
import { Shield } from "lucide-react";

export default function PrivacyPage() {
  return (
    <div className="max-w-2xl mx-auto w-full space-y-8 relative z-10 pt-16 flex-grow flex flex-col justify-center select-none">
      
      <div className="glass-panel p-8 space-y-6">
        <div className="flex items-center space-x-2 text-accent border-b border-white/5 pb-4">
          <Shield className="w-5 h-5" />
          <h1 className="text-sm font-mono font-bold tracking-widest uppercase text-white">
            MINIMALIST PRIVACY COMPLIANCE
          </h1>
        </div>

        <div className="text-xs text-gray-400 space-y-4 leading-relaxed font-mono">
          <p>
            This portfolio site utilizes Google Analytics 4 (GA4) and Microsoft Clarity to map visitor count, aggregate scroll bounds, and record download link activations. 
          </p>
          <p>
            No personal identity credentials, contact credentials, or private message structures are exposed or cataloged without explicit submission. All information entered in the Contact Form is processed directly in client memory loops (simulated) and is not persisted.
          </p>
          <p>
            Your browser audio permissions and theme toggle values are stored inside your device&apos;s local memory parameters (<code className="text-accent bg-white/5 px-1 rounded">localStorage</code>) to maintain continuous preferences across subsequent page visits.
          </p>
          <p className="text-[10px] text-gray-600">
            Last Updated: June 2026
          </p>
        </div>
      </div>

    </div>
  );
}
