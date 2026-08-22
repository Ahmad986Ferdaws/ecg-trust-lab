"use client";

import { useEffect, useRef, useState } from "react";

export function useStoryMotion<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (reducedMotion) {
      node.style.setProperty("--story-progress", "0.5");
      const revealFrame = window.requestAnimationFrame(() => setIsVisible(true));
      return () => window.cancelAnimationFrame(revealFrame);
    }

    let frame = 0;
    const updateProgress = () => {
      frame = 0;
      const bounds = node.getBoundingClientRect();
      const travel = window.innerHeight + bounds.height;
      const progress = Math.min(
        1,
        Math.max(0, (window.innerHeight - bounds.top) / travel),
      );
      node.style.setProperty("--story-progress", progress.toFixed(4));
    };

    const onScroll = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(updateProgress);
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setIsVisible(true);
      },
      { rootMargin: "-8% 0px -8% 0px", threshold: 0.12 },
    );

    observer.observe(node);
    updateProgress();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);

    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  return { ref, isVisible };
}

export function useCountUp(value: number, active: boolean, duration = 1200) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (!active) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const revealFrame = window.requestAnimationFrame(() =>
        setDisplayValue(value),
      );
      return () => window.cancelAnimationFrame(revealFrame);
    }

    let frame = 0;
    const startedAt = performance.now();

    const tick = (now: number) => {
      const elapsed = Math.min(1, (now - startedAt) / duration);
      const eased = 1 - Math.pow(1 - elapsed, 4);
      setDisplayValue(Math.round(value * eased));
      if (elapsed < 1) frame = window.requestAnimationFrame(tick);
    };

    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [active, duration, value]);

  return displayValue;
}
