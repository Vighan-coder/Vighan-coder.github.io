"use client";

/**
 * Dispatches tracking events to Google Analytics 4 (GA4) and Microsoft Clarity.
 * Bypasses tracking if the corresponding library script is not loaded.
 */
export const trackEvent = (
  action: string,
  category: string,
  label?: string,
  value?: number
) => {
  try {
    const win = typeof window !== "undefined" ? (window as typeof window & {
      gtag?: (command: string, action: string, params?: Record<string, unknown>) => void;
      clarity?: (command: string, action: string) => void;
    }) : null;

    // 1. Google Analytics tracking
    if (win && win.gtag) {
      win.gtag("event", action, {
        event_category: category,
        event_label: label,
        value: value,
      });
    }

    // 2. Microsoft Clarity tracking
    if (win && win.clarity) {
      win.clarity("event", action);
    }
    
    // Log to console in development mode
    if (process.env.NODE_ENV === "development") {
      console.log(`[Analytics Event] Action: ${action}, Category: ${category}, Label: ${label || "none"}`);
    }
  } catch (err) {
    console.warn("Analytics trackEvent error", err);
  }
};
