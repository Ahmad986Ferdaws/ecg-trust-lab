/* eslint-disable react/no-unknown-property, react-hooks/immutability -- R3F and Line2 expose intentionally imperative render resources. */

import { Suspense, useCallback, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import * as THREE from "three";
import {
  SCHEMATIC_LEADS as LEADS,
  sampledLeadValue,
  type LeadDefinition,
} from "../../lib/schematic-leads";

type LeadResource = {
  geometry: LineGeometry;
  line: Line2;
  material: LineMaterial;
  segmentCount: number;
};

const PAPER = "#EEE7D8";
const PAPER_DARK = "#E3D7C4";
const INK = "#343B37";
const RESNET = "#006B70";
const TRANSFORMER = "#98483B";
const SAMPLE_COUNT = 520;
const ACQUISITION_SECONDS = 1.9;
const LEAD_SPACING = 0.53;
const TOP_LEAD_Y = 2.95;

function buildLeadResource(lead: LeadDefinition): LeadResource {
  const positions: number[] = [];
  const colors: number[] = [];
  const resnetColor = new THREE.Color(RESNET);
  const transformerColor = new THREE.Color(TRANSFORMER);

  for (let sample = 0; sample < SAMPLE_COUNT; sample += 1) {
    const progress = sample / (SAMPLE_COUNT - 1);
    const x = THREE.MathUtils.lerp(-1, 1, progress);
    const y = sampledLeadValue(progress, lead);
    const color = progress < 0.5 ? resnetColor : transformerColor;

    positions.push(x, y, 0);
    colors.push(color.r, color.g, color.b);
  }

  const geometry = new LineGeometry();
  geometry.setPositions(positions);
  geometry.setColors(colors);
  geometry.instanceCount = 0;

  const material = new LineMaterial({
    color: 0xffffff,
    linewidth: 1.35,
    vertexColors: true,
    worldUnits: false,
    alphaToCoverage: true,
    transparent: true,
    opacity: 0.82,
    depthTest: false,
    depthWrite: false,
  });

  const line = new Line2(geometry, material);
  line.computeLineDistances();
  line.renderOrder = 2;
  line.frustumCulled = false;

  return {
    geometry,
    line,
    material,
    segmentCount: SAMPLE_COUNT - 1,
  };
}

function PaperField() {
  const texture = useMemo(() => {
    const size = 128;
    const data = new Uint8Array(size * size * 4);
    const base = new THREE.Color(PAPER);
    const minor = new THREE.Color(PAPER_DARK);
    const major = new THREE.Color("#CDBAA2");

    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const index = (y * size + x) * 4;
        const isMajor = x % 32 === 0 || y % 32 === 0;
        const isMinor = x % 8 === 0 || y % 8 === 0;
        const color = isMajor ? major : isMinor ? minor : base;
        data[index] = Math.round(color.r * 255);
        data[index + 1] = Math.round(color.g * 255);
        data[index + 2] = Math.round(color.b * 255);
        data[index + 3] = 255;
      }
    }

    const paperTexture = new THREE.DataTexture(
      data,
      size,
      size,
      THREE.RGBAFormat,
    );
    paperTexture.wrapS = THREE.RepeatWrapping;
    paperTexture.wrapT = THREE.RepeatWrapping;
    paperTexture.repeat.set(8, 4);
    paperTexture.magFilter = THREE.NearestFilter;
    paperTexture.minFilter = THREE.NearestFilter;
    paperTexture.colorSpace = THREE.SRGBColorSpace;
    paperTexture.needsUpdate = true;
    return paperTexture;
  }, []);

  useEffect(() => () => texture.dispose(), [texture]);

  return (
    <mesh position={[0, 0, -0.5]}>
      <planeGeometry args={[24, 12]} />
      <meshBasicMaterial map={texture} toneMapped={false} />
    </mesh>
  );
}

function EvidenceField({
  acquisitionComplete,
  focusedLead,
  onAcquisitionComplete,
  onFocusLead,
  paused,
}: {
  acquisitionComplete: boolean;
  focusedLead: number | null;
  onAcquisitionComplete: () => void;
  onFocusLead: (index: number | null) => void;
  paused: boolean;
}) {
  const { invalidate, viewport } = useThree();
  const sweep = useRef<THREE.Mesh>(null);
  const progress = useRef(acquisitionComplete || paused ? 1 : 0);
  const previouslyPaused = useRef(paused);
  const traceHalfWidth = Math.max(2.25, Math.min(6.65, viewport.width / 2 - 0.65));
  const resources = useMemo(() => LEADS.map(buildLeadResource), []);

  useEffect(
    () => () => {
      for (const resource of resources) {
        resource.geometry.dispose();
        resource.material.dispose();
      }
    },
    [resources],
  );

  useEffect(() => {
    for (let index = 0; index < resources.length; index += 1) {
      const material = resources[index].material;
      const isFocused = focusedLead === index;
      const anotherLeadIsFocused = focusedLead !== null && !isFocused;
      material.opacity = anotherLeadIsFocused ? 0.2 : isFocused ? 1 : 0.82;
      material.linewidth = isFocused ? 2.25 : 1.35;
    }
    invalidate();
  }, [focusedLead, invalidate, resources]);

  useEffect(() => {
    if (!paused && !acquisitionComplete) {
      if (previouslyPaused.current) {
        progress.current = 0;
        for (const resource of resources) {
          resource.geometry.instanceCount = 0;
        }
        if (sweep.current) sweep.current.visible = true;
        invalidate();
      }
    } else {
      progress.current = 1;
      for (const resource of resources) {
        resource.geometry.instanceCount = resource.segmentCount;
      }
      if (sweep.current) sweep.current.visible = false;
      invalidate();
    }

    previouslyPaused.current = paused;
  }, [acquisitionComplete, invalidate, paused, resources]);

  useFrame((_, delta) => {
    if (paused || acquisitionComplete) return;
    if (progress.current >= 1) {
      onAcquisitionComplete();
      return;
    }

    progress.current = Math.min(
      1,
      progress.current + Math.min(delta, 0.05) / ACQUISITION_SECONDS,
    );

    for (const resource of resources) {
      resource.geometry.instanceCount = Math.max(
        1,
        Math.ceil(resource.segmentCount * progress.current),
      );
    }

    if (sweep.current) {
      sweep.current.position.x = THREE.MathUtils.lerp(
        -traceHalfWidth,
        traceHalfWidth,
        progress.current,
      );
      sweep.current.visible = progress.current < 1;
    }

    if (progress.current >= 1) onAcquisitionComplete();
  });

  const focusLeadAtPoint = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    const index = THREE.MathUtils.clamp(
      Math.round((TOP_LEAD_Y - event.point.y) / LEAD_SPACING),
      0,
      LEADS.length - 1,
    );
    onFocusLead(index);
  };

  return (
    <>
      <PaperField />

      <mesh position={[0, -0.01, -0.08]}>
        <planeGeometry args={[0.012, 6.45]} />
        <meshBasicMaterial
          color={INK}
          transparent
          opacity={0.16}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>

      {resources.map((resource, index) => {
        const y = TOP_LEAD_Y - index * LEAD_SPACING;
        return (
          <group key={LEADS[index].name} position={[0, y, 0]}>
            <primitive
              object={resource.line}
              scale={[traceHalfWidth, 0.235, 1]}
            />
          </group>
        );
      })}

      <mesh
        position={[0, 0.035, 0.12]}
        scale={[traceHalfWidth, 1, 1]}
        onPointerEnter={focusLeadAtPoint}
        onPointerMove={focusLeadAtPoint}
        onPointerLeave={() => onFocusLead(null)}
      >
        <planeGeometry args={[2, 6.29]} />
        <meshBasicMaterial
          transparent
          opacity={0}
          depthWrite={false}
          depthTest={false}
          colorWrite={false}
        />
      </mesh>

      <mesh
        ref={sweep}
        position={[-traceHalfWidth, 0.01, 0.18]}
        visible={!paused && !acquisitionComplete}
      >
        <planeGeometry args={[0.025, 6.45]} />
        <meshBasicMaterial
          color={INK}
          transparent
          opacity={0.38}
          depthWrite={false}
          depthTest={false}
          toneMapped={false}
        />
      </mesh>
    </>
  );
}

function ResultsScene({
  acquisitionComplete,
  focusedLead,
  onAcquisitionComplete,
  onFocusLead,
  paused,
}: {
  acquisitionComplete: boolean;
  focusedLead: number | null;
  onAcquisitionComplete: () => void;
  onFocusLead: (index: number | null) => void;
  paused: boolean;
}) {
  return (
    <>
      <color attach="background" args={[PAPER]} />
      <EvidenceField
        acquisitionComplete={acquisitionComplete}
        focusedLead={focusedLead}
        onAcquisitionComplete={onAcquisitionComplete}
        onFocusLead={onFocusLead}
        paused={paused}
      />
    </>
  );
}

type ResultsWebGLProps = {
  acquisitionComplete: boolean;
  focusedLead: number | null;
  onAcquisitionComplete: () => void;
  onFocusLead: (index: number | null) => void;
  onUnavailable: () => void;
  paused: boolean;
  shouldAnimate: boolean;
};

export default function ResultsWebGL({
  acquisitionComplete,
  focusedLead,
  onAcquisitionComplete,
  onFocusLead,
  onUnavailable,
  paused,
  shouldAnimate,
}: ResultsWebGLProps) {
  const onCanvasReady = useCallback((canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    canvas.addEventListener("webglcontextlost", onUnavailable);
    canvas.addEventListener("webglcontextcreationerror", onUnavailable);
    return () => {
      canvas.removeEventListener("webglcontextlost", onUnavailable);
      canvas.removeEventListener("webglcontextcreationerror", onUnavailable);
    };
  }, [onUnavailable]);

  return (
    <Canvas
      ref={onCanvasReady}
      aria-hidden="true"
      role="presentation"
      tabIndex={-1}
      dpr={1.25}
      frameloop={shouldAnimate ? "always" : "demand"}
      camera={{ position: [0, 0, 12], fov: 43, near: 0.1, far: 30 }}
      gl={{
        alpha: false,
        antialias: true,
        depth: true,
        stencil: false,
        powerPreference: "high-performance",
      }}
      onCreated={({ gl }) => {
        gl.outputColorSpace = THREE.SRGBColorSpace;
        gl.toneMapping = THREE.NoToneMapping;
      }}
      onPointerMissed={() => onFocusLead(null)}
      style={{ position: "absolute", inset: 0 }}
    >
      <Suspense fallback={null}>
        <ResultsScene
          acquisitionComplete={acquisitionComplete}
          focusedLead={focusedLead}
          onAcquisitionComplete={onAcquisitionComplete}
          onFocusLead={onFocusLead}
          paused={paused}
        />
      </Suspense>
    </Canvas>
  );
}
