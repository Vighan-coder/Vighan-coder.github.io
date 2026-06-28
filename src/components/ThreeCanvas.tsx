"use client";

import React, { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { MeshReflectorMaterial } from "@react-three/drei";
import * as THREE from "three";
import { usePathname } from "next/navigation";

// Camera controller for slow drifting, breathing, mouse parallax, and scene transition eases
function CameraRig() {
  const pathname = usePathname();
  const transitionProgress = useRef(0);
  const prevPathname = useRef(pathname);

  // Trigger transition motion if pathname changes
  useEffect(() => {
    if (pathname !== prevPathname.current) {
      transitionProgress.current = 1.0; // Start transition burst
      prevPathname.current = pathname;
    }
  }, [pathname]);

  useFrame((state) => {
    const { camera, pointer, clock } = state;
    const t = clock.getElapsedTime();
    
    // Normal slow breathing and mouse parallax
    const normalX = pointer.x * 0.8 + Math.sin(t * 0.3) * 0.2;
    const normalY = pointer.y * 0.5 + Math.cos(t * 0.2) * 0.15;
    
    // Transition drift: camera flies forward slightly when transition is active
    let transitionZOffset = 0;
    if (transitionProgress.current > 0.01) {
      // Zoom effect: push camera closer/further on click transitions
      transitionZOffset = Math.sin(transitionProgress.current * Math.PI) * -1.5;
      transitionProgress.current = THREE.MathUtils.lerp(transitionProgress.current, 0, 0.05);
    }

    const targetX = normalX;
    const targetY = normalY + 0.3; // Look slightly above ground
    const targetZ = 5 + transitionZOffset + Math.cos(t * 0.1) * 0.15; // Slow depth breathing

    // Lerp positions for smooth cinematic feel
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, targetX, 0.03);
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, targetY, 0.03);
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetZ, 0.03);

    // Dynamic camera rotation lookAt
    const targetLookAt = new THREE.Vector3(0, 0.2, 0);
    camera.lookAt(targetLookAt);
  });

  return null;
}

// Particle generation helper defined outside the render tree to ensure hook purity
function generateParticles(count: number) {
  const pos = new Float32Array(count * 3);
  const sw = new Float32Array(count * 3);
  const sc = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    // Cylindrical volumetric layout
    const theta = Math.random() * Math.PI * 2;
    const radius = 1.5 + Math.random() * 8.5;
    pos[i * 3] = Math.cos(theta) * radius;
    pos[i * 3 + 1] = (Math.random() - 0.5) * 8;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 12;

    sw[i * 3] = Math.random() * 0.3;     // sway rate x
    sw[i * 3 + 1] = Math.random() * 0.3; // sway rate y
    sw[i * 3 + 2] = Math.random() * 0.3; // sway rate z
    sc[i] = 0.5 + Math.random() * 1.5;
  }
  return [pos, sw, sc];
}

// Particle field that sways, drifts upward, and responds to mouse coordinates
function Particles({ count = 2500 }) {
  const pointsRef = useRef<THREE.Points>(null);
  const pathname = usePathname();
  const transitionBurst = useRef(0);
  const prevPathname = useRef(pathname);

  useEffect(() => {
    if (pathname !== prevPathname.current) {
      transitionBurst.current = 2.0; // Burst particle speeds
      prevPathname.current = pathname;
    }
  }, [pathname]);

  const [positions, sways] = useMemo(() => generateParticles(count), [count]);

  useFrame((state) => {
    if (!pointsRef.current) return;
    const { pointer, clock } = state;
    const t = clock.getElapsedTime();

    const geo = pointsRef.current.geometry;
    const posArr = geo.attributes.position.array as Float32Array;

    // Decay the transition speed burst back to normal
    if (transitionBurst.current > 0.01) {
      transitionBurst.current = THREE.MathUtils.lerp(transitionBurst.current, 0, 0.05);
    }

    const currentSpeed = 0.003 + transitionBurst.current * 0.02;

    for (let i = 0; i < count; i++) {
      const idx = i * 3;

      // Float particles slowly upwards
      posArr[idx + 1] += currentSpeed;
      if (posArr[idx + 1] > 4) {
        posArr[idx + 1] = -4; // Reset to bottom
      }

      // Parallax mouse sway
      const swayX = Math.sin(t * sways[idx] + i) * 0.001;
      posArr[idx] += swayX + (pointer.x * 0.001);

      // React to mouse proximity: subtle pushing
      const dx = posArr[idx] - pointer.x * 3;
      const dy = posArr[idx + 1] - pointer.y * 2;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 1.8) {
        const force = (1.8 - dist) * 0.004;
        posArr[idx] += (dx / dist) * force;
        posArr[idx + 1] += (dy / dist) * force;
      }
    }

    geo.attributes.position.needsUpdate = true;
    
    // Rotate the particle system slowly
    pointsRef.current.rotation.y = t * 0.015;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#00D084"
        transparent
        opacity={0.65}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// Volumetric lights and reflective ground studio environment
function StudioScene() {
  const pathname = usePathname();
  const spotlightRef = useRef<THREE.SpotLight>(null);
  const glowLightRef = useRef<THREE.PointLight>(null);
  
  const gsapAnimateLight = () => {
    // Basic JS animation fallback to keep it framework-independent inside standard R3F loops
    let count = 0;
    const interval = setInterval(() => {
      if (spotlightRef.current) {
        spotlightRef.current.intensity = THREE.MathUtils.lerp(spotlightRef.current.intensity, 1.8, 0.1);
      }
      count++;
      if (count > 30) clearInterval(interval);
    }, 30);
  };
  
  // Animate spotlight intensity/color on route transitions
  useEffect(() => {
    if (spotlightRef.current) {
      // Trigger a brief burst of light intensity on transition
      spotlightRef.current.intensity = 4.0;
      gsapAnimateLight();
    }
  }, [pathname]);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    // Soft volumetric light breathing effect
    if (glowLightRef.current) {
      glowLightRef.current.intensity = 1.0 + Math.sin(t * 0.8) * 0.3;
      // Drift light source coordinate slowly
      glowLightRef.current.position.x = Math.sin(t * 0.4) * 1.5;
    }
  });

  return (
    <>
      {/* Background black studio ambient */}
      <color attach="background" args={["#050505"]} />
      <fog attach="fog" args={["#050505", 4, 12]} />

      <ambientLight intensity={0.05} />
      
      {/* Soft emerald volumetric lighting sources */}
      <spotLight
        ref={spotlightRef}
        position={[0, 6, -3]}
        angle={0.7}
        penumbra={1}
        intensity={1.8}
        color="#00D084"
        castShadow
        shadow-bias={-0.0001}
      />

      <pointLight
        ref={glowLightRef}
        position={[0, -0.5, 1]}
        distance={8}
        intensity={1.2}
        color="#00D084"
      />

      {/* Reflective floor mirroring particles and foreground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.8, 0]}>
        <planeGeometry args={[60, 60]} />
        <MeshReflectorMaterial
          blur={[300, 100]}
          resolution={512}
          mixBlur={1}
          mixStrength={0.4}
          roughness={1}
          depthScale={1}
          minDepthThreshold={0.4}
          maxDepthThreshold={1.4}
          color="#0d0d0d"
          metalness={0.6}
          mirror={0.5}
        />
      </mesh>

      {/* Floating abstract AI sculpture placeholder on Home Page */}
      {pathname === "/" && <FloatingAvatarPlaceholder />}
    </>
  );
}

function FloatingAvatarPlaceholder() {
  const meshRef1 = useRef<THREE.Mesh>(null);
  const meshRef2 = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (meshRef1.current) {
      meshRef1.current.rotation.x = t * 0.15;
      meshRef1.current.rotation.y = t * 0.2;
      meshRef1.current.position.y = 0.3 + Math.sin(t * 1.2) * 0.08;
    }
    if (meshRef2.current) {
      meshRef2.current.rotation.x = -t * 0.2;
      meshRef2.current.rotation.y = -t * 0.15;
      meshRef2.current.position.y = 0.3 + Math.sin(t * 1.2) * 0.08;
    }
  });

  return (
    <group>
      {/* Outer 3D floating abstract geometric shape */}
      <mesh ref={meshRef1}>
        <icosahedronGeometry args={[0.9, 2]} />
        <meshPhysicalMaterial
          color="#00D084"
          roughness={0.1}
          metalness={0.15}
          transmission={0.85}
          thickness={1.8}
          ior={1.45}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Inner glowing core representing AI compute logic */}
      <mesh ref={meshRef2}>
        <icosahedronGeometry args={[0.55, 1]} />
        <meshBasicMaterial
          color="#4AFFB8"
          wireframe
          transparent
          opacity={0.35}
        />
      </mesh>

      {/* Volumetric orbit rings */}
      <mesh rotation={[Math.PI / 4, Math.PI / 4, 0]} position={[0, 0.3, 0]}>
        <torusGeometry args={[1.35, 0.015, 8, 64]} />
        <meshBasicMaterial color="#00D084" transparent opacity={0.15} />
      </mesh>
    </group>
  );
}

// Main exported Canvas component
export default function ThreeCanvas() {
  return (
    <div className="fixed inset-0 w-full h-full -z-10 bg-[#050505] pointer-events-none select-none">
      <Canvas
        camera={{ fov: 60, position: [0, 0, 5], near: 0.1, far: 20 }}
        gl={{
          antialias: true,
          powerPreference: "high-performance",
          alpha: false,
          stencil: false,
          depth: true,
        }}
      >
        <React.Suspense fallback={null}>
          <StudioScene />
          <Particles count={2200} />
          <CameraRig />
        </React.Suspense>
      </Canvas>
    </div>
  );
}
