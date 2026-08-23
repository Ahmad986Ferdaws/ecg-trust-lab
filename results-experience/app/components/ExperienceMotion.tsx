"use client";

import { useSyncExternalStore } from "react";

type ExperienceMotionState = "paused" | "running";
type ExperienceMotionSource = "control" | "external" | "preference";

export type ExperienceMotionChangeDetail = {
  motion: ExperienceMotionState;
  paused: boolean;
  source: ExperienceMotionSource;
};

export const ECG_MOTION_CHANGE_EVENT = "ecg-motion-change";

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";
const listeners = new Set<() => void>();

let currentMotion: ExperienceMotionState = "paused";
let initialized = false;
let hasExplicitChoice = false;
let motionPreference: MediaQueryList | null = null;

function isMotionState(value: unknown): value is ExperienceMotionState {
  return value === "paused" || value === "running";
}

function notifyListeners() {
  listeners.forEach((listener) => listener());
}

function dispatchMotionChange(
  motion: ExperienceMotionState,
  source: ExperienceMotionSource,
) {
  if (typeof window === "undefined") return;

  const detail: ExperienceMotionChangeDetail = {
    motion,
    paused: motion === "paused",
    source,
  };

  window.dispatchEvent(
    new CustomEvent<ExperienceMotionChangeDetail>(ECG_MOTION_CHANGE_EVENT, {
      detail,
    }),
  );
}

function commitMotion(
  motion: ExperienceMotionState,
  source: ExperienceMotionSource,
  shouldDispatch = true,
) {
  const changed = currentMotion !== motion;
  currentMotion = motion;

  if (typeof document !== "undefined") {
    document.documentElement.dataset.motion = motion;
  }

  if (changed) notifyListeners();
  if (shouldDispatch) dispatchMotionChange(motion, source);
}

function getMotionFromEvent(event: Event): ExperienceMotionState | null {
  const detail = (event as CustomEvent<Partial<ExperienceMotionChangeDetail>>)
    .detail;

  if (isMotionState(detail?.motion)) return detail.motion;
  if (typeof detail?.paused === "boolean") {
    return detail.paused ? "paused" : "running";
  }

  const documentMotion =
    typeof document === "undefined"
      ? undefined
      : document.documentElement.dataset.motion;
  return isMotionState(documentMotion) ? documentMotion : null;
}

function handleMotionChange(event: Event) {
  const motion = getMotionFromEvent(event);
  if (!motion) return;

  const detail = (event as CustomEvent<Partial<ExperienceMotionChangeDetail>>)
    .detail;
  if (detail?.source !== "preference") hasExplicitChoice = true;

  commitMotion(motion, "external", false);
}

function handleMotionPreferenceChange(event: MediaQueryListEvent) {
  if (hasExplicitChoice) return;
  commitMotion(event.matches ? "paused" : "running", "preference");
}

function initializeMotion() {
  if (initialized || typeof window === "undefined") return;
  initialized = true;

  window.addEventListener(ECG_MOTION_CHANGE_EVENT, handleMotionChange);

  if (typeof window.matchMedia === "function") {
    motionPreference = window.matchMedia(REDUCED_MOTION_QUERY);
    motionPreference.addEventListener("change", handleMotionPreferenceChange);
  }

  const documentMotion = document.documentElement.dataset.motion;
  const initialMotion = motionPreference?.matches
    ? "paused"
    : isMotionState(documentMotion)
      ? documentMotion
      : "running";

  commitMotion(initialMotion, "preference");
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  initializeMotion();
  return () => listeners.delete(listener);
}

function getMotionSnapshot() {
  return currentMotion;
}

function getServerMotionSnapshot(): ExperienceMotionState {
  return "paused";
}

function toggleMotion() {
  hasExplicitChoice = true;
  commitMotion(currentMotion === "paused" ? "running" : "paused", "control");
}

export function useExperienceMotion(): { paused: boolean } {
  const motion = useSyncExternalStore(
    subscribe,
    getMotionSnapshot,
    getServerMotionSnapshot,
  );

  return { paused: motion === "paused" };
}

export function MotionControl() {
  const { paused } = useExperienceMotion();
  const label = paused ? "Play motion" : "Pause motion";

  return (
    <button
      type="button"
      className="motion-control"
      aria-label={label}
      onClick={toggleMotion}
    >
      {label}
    </button>
  );
}
