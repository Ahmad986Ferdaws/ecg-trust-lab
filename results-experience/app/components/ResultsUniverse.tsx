"use client";

import {
  Component,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
  type ErrorInfo,
  type ReactNode,
  type RefObject,
} from "react";
import { useExperienceMotion } from "./ExperienceMotion";
import { SchematicLeadFallback } from "./SchematicLeadFallback";
import styles from "./ResultsUniverse.module.css";
import { SCHEMATIC_LEADS as LEADS } from "../../lib/schematic-leads";

// Keep Three.js and R3F outside the initial client-reference graph.
const ResultsWebGL = lazy(() => import("./ResultsWebGL"));

type ResultsUniverseProps = {
  className?: string;
};

const PAPER = "#EEE7D8";
const INK = "#343B37";
const RESNET = "#006B70";
const TRANSFORMER = "#98483B";

class SceneErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    if (process.env.NODE_ENV !== "production") {
      console.warn("The ECG evidence canvas could not be rendered.", error, info);
    }
  }

  render() {
    if (this.state.hasError) return null;
    return this.props.children;
  }
}

function useElementInView(ref: RefObject<HTMLElement | null>) {
  const [inView, setInView] = useState(true);

  useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(
      ([entry]) => setInView(entry.isIntersecting),
      { rootMargin: "80px 0px", threshold: 0.01 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);

  return inView;
}

function subscribeToVisibility(onStoreChange: () => void) {
  document.addEventListener("visibilitychange", onStoreChange);
  return () => document.removeEventListener("visibilitychange", onStoreChange);
}

function getDocumentVisibility() {
  return document.visibilityState === "visible";
}

function getServerVisibility() {
  return true;
}

export function ResultsUniverse({ className }: ResultsUniverseProps) {
  const container = useRef<HTMLDivElement>(null);
  const [webglStatus, setWebglStatus] = useState<
    "waiting" | "available" | "unavailable"
  >("waiting");
  const [acquisitionComplete, setAcquisitionComplete] = useState(false);
  const [focusedLead, setFocusedLead] = useState<number | null>(null);
  const { paused } = useExperienceMotion();
  const inView = useElementInView(container);
  const documentVisible = useSyncExternalStore(
    subscribeToVisibility,
    getDocumentVisibility,
    getServerVisibility,
  );
  const onAcquisitionComplete = useCallback(
    () => setAcquisitionComplete(true),
    [],
  );
  useEffect(() => {
    // The SVG is already visible. Only enhance it when motion is wanted and
    // the scene is visible, leaving the initial hydration work to finish first.
    if (webglStatus !== "waiting" || paused || !inView || !documentVisible) return;

    const activateWebGL = () => {
      const probe = document.createElement("canvas");
      try {
        const context = probe.getContext("webgl2");
        if (context && !context.isContextLost()) {
          context.getExtension("WEBGL_lose_context")?.loseContext();
          setWebglStatus("available");
          return;
        }
      } catch {
        // A blocked context leaves the schematic intact without importing Three.js.
      }
      setWebglStatus("unavailable");
    };

    if (typeof window.requestIdleCallback === "function") {
      const request = window.requestIdleCallback(activateWebGL, { timeout: 1500 });
      return () => window.cancelIdleCallback(request);
    }
    const request = window.setTimeout(activateWebGL, 0);
    return () => window.clearTimeout(request);
  }, [documentVisible, inView, paused, webglStatus]);
  const onUnavailable = useCallback(() => {
    setWebglStatus("unavailable");
    setFocusedLead(null);
  }, []);
  const shouldAnimate =
    inView && documentVisible && !paused && !acquisitionComplete;

  return (
    <div
      ref={container}
      className={[styles.scene, className].filter(Boolean).join(" ")}
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        isolation: "isolate",
        cursor: focusedLead === null ? "default" : "crosshair",
        background: PAPER,
      }}
    >
      <SchematicLeadFallback />
      {webglStatus === "available" ? (
        <SceneErrorBoundary>
          <Suspense fallback={null}>
            <ResultsWebGL
              acquisitionComplete={acquisitionComplete}
              focusedLead={focusedLead}
              onAcquisitionComplete={onAcquisitionComplete}
              onFocusLead={setFocusedLead}
              onUnavailable={onUnavailable}
              paused={paused}
              shouldAnimate={shouldAnimate}
            />
          </Suspense>
        </SceneErrorBoundary>
      ) : null}

      <div
        aria-hidden="true"
        className={styles.lensLabels}
        style={{
          position: "absolute",
          top: "4.5%",
          left: "7%",
          right: "5%",
          color: INK,
          fontFamily: "var(--mono)",
          fontSize: "12px",
          fontWeight: 700,
          letterSpacing: ".14em",
          pointerEvents: "none",
        }}
      >
        <span style={{ color: RESNET }}>RESNET LENS</span>
        <span className={styles.sharedBaseline} style={{ opacity: 0.54 }}>SAME WAVEFORM · EQUAL BASELINE</span>
        <span className={styles.transformerLabel} style={{ color: TRANSFORMER }}>TRANSFORMER LENS</span>
      </div>

      {LEADS.map((lead, index) => (
        <span
          key={lead.name}
          aria-hidden="true"
          style={{
            position: "absolute",
            left: "2.2%",
            top: `${16.5 + index * 5.82}%`,
            color: focusedLead === index ? INK : "#65665E",
            fontFamily: "var(--mono)",
            fontSize: "12px",
            fontWeight: focusedLead === index ? 800 : 650,
            letterSpacing: ".05em",
            pointerEvents: "none",
            transition: paused ? "none" : "color 140ms ease",
          }}
        >
          {lead.name}
        </span>
      ))}
    </div>
  );
}

export default ResultsUniverse;
