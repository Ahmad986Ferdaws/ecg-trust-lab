"use client";

/* eslint-disable react/no-unknown-property -- React Three Fiber adds Three.js primitives to JSX. */

import {
  Component,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ErrorInfo,
  type ReactNode,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

type ResultsUniverseProps = {
  className?: string;
};

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";
const RESNET_AUROC = 0.921921;
const TRANSFORMER_AUROC = 0.89742;

function subscribeToReducedMotion(onStoreChange: () => void) {
  if (typeof window === "undefined") return () => undefined;

  const query = window.matchMedia(REDUCED_MOTION_QUERY);
  query.addEventListener("change", onStoreChange);
  return () => query.removeEventListener("change", onStoreChange);
}

function getReducedMotionSnapshot() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia(REDUCED_MOTION_QUERY).matches
  );
}

function useReducedMotionPreference() {
  return useSyncExternalStore(
    subscribeToReducedMotion,
    getReducedMotionSnapshot,
    () => false,
  );
}

function gaussian(x: number, center: number, width: number, amplitude: number) {
  return amplitude * Math.exp(-((x - center) ** 2) / (2 * width ** 2));
}

function ecgBeat(x: number, phase: number) {
  const period = 2.72;
  const t = ((x + phase) % period + period) % period;

  return (
    gaussian(t, 0.48, 0.12, 0.16) +
    gaussian(t, 1.04, 0.045, -0.18) +
    gaussian(t, 1.14, 0.028, 1.08) +
    gaussian(t, 1.24, 0.052, -0.42) +
    gaussian(t, 1.78, 0.2, 0.34)
  );
}

function buildEcgCurve(index: number, count: number) {
  const points: THREE.Vector3[] = [];
  const baseline = (index - (count - 1) / 2) * 0.42;
  const phase = index * 0.23;

  for (let step = 0; step <= 180; step += 1) {
    const progress = step / 180;
    const x = THREE.MathUtils.lerp(-7.2, 7.2, progress);
    const edgeFade = Math.sin(progress * Math.PI) ** 0.35;
    const signal = ecgBeat(x * 0.98, phase) * edgeFade;
    const y = baseline + signal * (0.46 + index * 0.015);
    const z =
      -0.65 +
      Math.sin(x * 0.42 + index * 0.82) * 0.16 +
      Math.cos(index * 1.08) * 0.17;

    points.push(new THREE.Vector3(x, y, z));
  }

  return new THREE.CatmullRomCurve3(points, false, "catmullrom", 0.25);
}

function EcgRibbon({
  curve,
  color,
  opacity,
}: {
  curve: THREE.CatmullRomCurve3;
  color: THREE.ColorRepresentation;
  opacity: number;
}) {
  const geometries = useMemo(
    () => ({
      core: new THREE.TubeGeometry(curve, 220, 0.012, 5, false),
      glow: new THREE.TubeGeometry(curve, 220, 0.048, 6, false),
    }),
    [curve],
  );

  useEffect(
    () => () => {
      geometries.core.dispose();
      geometries.glow.dispose();
    },
    [geometries],
  );

  return (
    <group>
      <mesh geometry={geometries.glow}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={opacity * 0.16}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      <mesh geometry={geometries.core}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={opacity}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

function PulseMarker({
  curve,
  color,
  offset,
  reducedMotion,
}: {
  curve: THREE.CatmullRomCurve3;
  color: THREE.ColorRepresentation;
  offset: number;
  reducedMotion: boolean;
}) {
  const marker = useRef<THREE.Group>(null);
  const initialPosition = useMemo(() => curve.getPointAt(offset), [curve, offset]);

  useFrame(({ clock }) => {
    if (!marker.current || reducedMotion) return;

    const progress = (offset + clock.getElapsedTime() * 0.072) % 1;
    marker.current.position.copy(curve.getPointAt(progress));
    const pulse = 0.82 + Math.sin(clock.getElapsedTime() * 7 + offset * 12) * 0.18;
    marker.current.scale.setScalar(pulse);
  });

  return (
    <group ref={marker} position={initialPosition}>
      <mesh>
        <sphereGeometry args={[0.055, 12, 12]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
      <mesh scale={3.6}>
        <sphereGeometry args={[0.055, 12, 12]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.12}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

function SignalField({ reducedMotion }: { reducedMotion: boolean }) {
  const group = useRef<THREE.Group>(null);
  const curves = useMemo(
    () => Array.from({ length: 8 }, (_, index) => buildEcgCurve(index, 8)),
    [],
  );
  const colors = useMemo(() => {
    const cyan = new THREE.Color("#48e8ff");
    const violet = new THREE.Color("#a76dff");

    return curves.map((_, index) =>
      cyan.clone().lerp(violet, index / Math.max(1, curves.length - 1)),
    );
  }, [curves]);

  useFrame(({ clock }) => {
    if (!group.current || reducedMotion) return;
    const elapsed = clock.getElapsedTime();
    group.current.rotation.z = Math.sin(elapsed * 0.13) * 0.018;
    group.current.rotation.y = Math.sin(elapsed * 0.09) * 0.028;
  });

  return (
    <group ref={group} rotation={[-0.12, -0.1, -0.035]} position={[0, 0.1, 0]}>
      {curves.map((curve, index) => (
        <EcgRibbon
          key={index}
          curve={curve}
          color={colors[index]}
          opacity={0.34 + index * 0.055}
        />
      ))}
      <PulseMarker
        curve={curves[2]}
        color={colors[2]}
        offset={0.12}
        reducedMotion={reducedMotion}
      />
      <PulseMarker
        curve={curves[5]}
        color={colors[5]}
        offset={0.58}
        reducedMotion={reducedMotion}
      />
    </group>
  );
}

function scoreToHeight(score: number) {
  return THREE.MathUtils.mapLinear(score, 0.85, 0.94, 1.25, 2.85);
}

function ParticleField({ reducedMotion }: { reducedMotion: boolean }) {
  const points = useRef<THREE.Points>(null);
  const geometry = useMemo(() => {
    const count = 140;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const cyan = new THREE.Color("#8fefff");
    const violet = new THREE.Color("#b592ff");
    let seed = 0x8f3d72a1;

    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    for (let index = 0; index < count; index += 1) {
      const stride = index * 3;
      const depth = random();
      positions[stride] = (random() - 0.5) * 13.5;
      positions[stride + 1] = (random() - 0.5) * 7.5;
      positions[stride + 2] = -4.4 + depth * 8.2;

      const color = cyan.clone().lerp(violet, random() * 0.72);
      colors[stride] = color.r;
      colors[stride + 1] = color.g;
      colors[stride + 2] = color.b;
    }

    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    );
    particleGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    particleGeometry.computeBoundingSphere();
    return particleGeometry;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  useFrame(({ clock }) => {
    if (!points.current || reducedMotion) return;
    const elapsed = clock.getElapsedTime();
    points.current.rotation.y = elapsed * 0.008;
    points.current.rotation.z = Math.sin(elapsed * 0.08) * 0.025;
    points.current.position.y = Math.sin(elapsed * 0.17) * 0.035;
  });

  return (
    <points ref={points} geometry={geometry} frustumCulled={false}>
      <pointsMaterial
        size={0.032}
        sizeAttenuation
        vertexColors
        transparent
        opacity={0.52}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        toneMapped={false}
      />
    </points>
  );
}

function DataMonolith({
  score,
  color,
  position,
  reducedMotion,
  offset,
}: {
  score: number;
  color: string;
  position: [number, number, number];
  reducedMotion: boolean;
  offset: number;
}) {
  const monolith = useRef<THREE.Group>(null);
  const ring = useRef<THREE.Group>(null);
  const height = scoreToHeight(score);
  const baseY = -2.2;

  useFrame(({ clock }, delta) => {
    if (!monolith.current) return;

    if (reducedMotion) {
      monolith.current.position.set(position[0], position[1], position[2]);
      monolith.current.rotation.set(0, 0, 0);
      return;
    }

    const elapsed = clock.getElapsedTime();
    monolith.current.position.set(
      position[0],
      position[1] + Math.sin(elapsed * (0.68 + offset * 0.06) + offset) * 0.055,
      position[2],
    );
    monolith.current.rotation.y = Math.sin(elapsed * 0.22 + offset) * 0.02;
    monolith.current.rotation.z = Math.sin(elapsed * 0.3 + offset) * 0.009;

    if (ring.current) {
      ring.current.rotation.z += delta * (0.16 + offset * 0.025);
      ring.current.rotation.x =
        Math.PI / 2.45 + Math.sin(elapsed * 0.32 + offset) * 0.1;
    }
  });

  return (
    <group ref={monolith} position={position}>
        <mesh position={[0, baseY + height / 2, 0]}>
          <cylinderGeometry args={[0.48, 0.67, height, 6, 1, false]} />
          <meshPhysicalMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.22}
            metalness={0.42}
            roughness={0.17}
            transmission={0.2}
            thickness={1.15}
            transparent
            opacity={0.78}
          />
        </mesh>
        <mesh position={[0, baseY + height / 2, 0]} scale={1.055}>
          <cylinderGeometry args={[0.48, 0.67, height, 6, 1, true]} />
          <meshBasicMaterial
            color={color}
            wireframe
            transparent
            opacity={0.18}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>

        <group ref={ring} position={[0, baseY + height + 0.12, 0]}>
          <mesh>
            <torusGeometry args={[0.77, 0.014, 8, 72]} />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.82}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
          <mesh scale={1.15}>
            <torusGeometry args={[0.77, 0.052, 8, 72]} />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.08}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
          <mesh position={[0.77, 0, 0]}>
            <sphereGeometry args={[0.045, 10, 10]} />
            <meshBasicMaterial color="#ffffff" toneMapped={false} />
          </mesh>
        </group>

        <mesh position={[0, baseY - 0.04, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.72, 1.1, 64]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.12}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.DoubleSide}
            toneMapped={false}
          />
        </mesh>
    </group>
  );
}

function Portal({ reducedMotion }: { reducedMotion: boolean }) {
  const portal = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (!portal.current || reducedMotion) return;
    portal.current.rotation.z -= delta * 0.024;
  });

  return (
    <group ref={portal} position={[0.25, 0, -3.25]} rotation={[0.08, -0.08, 0]}>
      {[2.55, 3.15, 3.9].map((radius, index) => (
        <mesh key={radius} rotation={[0, 0, index * 0.42]}>
          <torusGeometry args={[radius, index === 0 ? 0.018 : 0.01, 7, 128]} />
          <meshBasicMaterial
            color={index === 1 ? "#956fff" : "#3de5ff"}
            transparent
            opacity={0.18 - index * 0.035}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      ))}
      <mesh>
        <circleGeometry args={[4.1, 96]} />
        <meshBasicMaterial
          color="#10244d"
          transparent
          opacity={0.075}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

function CameraRig({ reducedMotion }: { reducedMotion: boolean }) {
  const { camera, pointer } = useThree();
  const targetPosition = useRef(new THREE.Vector3());
  const lookAt = useRef(new THREE.Vector3(0, -0.05, 0));

  useFrame((_, delta) => {
    if (reducedMotion) return;

    targetPosition.current.set(
      pointer.x * 0.42,
      0.26 + pointer.y * 0.2,
      9.35 + Math.abs(pointer.x) * 0.08,
    );
    const damping = 1 - Math.exp(-delta * 2.3);
    camera.position.lerp(targetPosition.current, damping);
    camera.lookAt(lookAt.current);
  });

  return null;
}

function ResultsScene({ reducedMotion }: { reducedMotion: boolean }) {
  return (
    <>
      <color attach="background" args={["#030610"]} />
      <fog attach="fog" args={["#030610", 7.5, 18]} />
      <ambientLight intensity={0.36} color="#7087ff" />
      <directionalLight position={[2.5, 5, 5]} intensity={2.1} color="#dfeaff" />
      <pointLight position={[-4.5, 1.5, 2]} intensity={18} distance={9} color="#35dfff" />
      <pointLight position={[4, 0.5, 1]} intensity={16} distance={8} color="#a167ff" />

      <Portal reducedMotion={reducedMotion} />
      <SignalField reducedMotion={reducedMotion} />
      <DataMonolith
        score={RESNET_AUROC}
        color="#35ddff"
        position={[-2.25, -0.1, 0.38]}
        reducedMotion={reducedMotion}
        offset={0}
      />
      <DataMonolith
        score={TRANSFORMER_AUROC}
        color="#a467ff"
        position={[2.25, -0.1, 0.15]}
        reducedMotion={reducedMotion}
        offset={1}
      />

      <ParticleField reducedMotion={reducedMotion} />

      <gridHelper
        args={[18, 36, "#345582", "#142340"]}
        position={[0, -2.28, -0.2]}
      />
      <CameraRig reducedMotion={reducedMotion} />
    </>
  );
}

function CanvasFallback() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        background:
          "radial-gradient(circle at 50% 46%, rgba(75, 88, 230, 0.2), transparent 34%), radial-gradient(circle at 18% 64%, rgba(53, 221, 255, 0.14), transparent 24%), #030610",
      }}
    >
      <div
        style={{
          position: "absolute",
          left: "-5%",
          right: "-5%",
          top: "48%",
          height: 2,
          opacity: 0.52,
          transform: "rotate(-2deg)",
          background:
            "linear-gradient(90deg, transparent, #35ddff 18%, #ffffff 50%, #a467ff 82%, transparent)",
          boxShadow: "0 0 28px rgba(53, 221, 255, 0.45)",
        }}
      />
      <div
        style={{
          position: "absolute",
          width: "min(54vw, 620px)",
          aspectRatio: "1",
          left: "50%",
          top: "48%",
          transform: "translate(-50%, -50%)",
          border: "1px solid rgba(121, 118, 255, 0.24)",
          borderRadius: "50%",
          boxShadow:
            "0 0 80px rgba(53, 221, 255, 0.07), inset 0 0 80px rgba(164, 103, 255, 0.06)",
        }}
      />
    </div>
  );
}

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
      console.warn("The WebGL results scene could not be rendered.", error, info);
    }
  }

  render() {
    if (this.state.hasError) return <CanvasFallback />;
    return this.props.children;
  }
}

export function ResultsUniverse({ className }: ResultsUniverseProps) {
  const reducedMotion = useReducedMotionPreference();

  return (
    <div
      className={className}
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        minHeight: 520,
        overflow: "hidden",
        isolation: "isolate",
        pointerEvents: "auto",
      }}
    >
      <SceneErrorBoundary>
        <Canvas
          aria-hidden="true"
          role="presentation"
          tabIndex={-1}
          fallback={<CanvasFallback />}
          dpr={1.25}
          frameloop={reducedMotion ? "demand" : "always"}
          camera={{ position: [0, 0.26, 9.35], fov: 36, near: 0.1, far: 30 }}
          gl={{
            alpha: false,
            antialias: true,
            depth: true,
            stencil: false,
            powerPreference: "high-performance",
          }}
          onCreated={({ gl }) => {
            gl.outputColorSpace = THREE.SRGBColorSpace;
            gl.toneMapping = THREE.ACESFilmicToneMapping;
            gl.toneMappingExposure = 1.08;
          }}
          style={{ position: "absolute", inset: 0 }}
        >
          <Suspense fallback={null}>
            <ResultsScene reducedMotion={reducedMotion} />
          </Suspense>
        </Canvas>
      </SceneErrorBoundary>
    </div>
  );
}

export default ResultsUniverse;
