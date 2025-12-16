"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Text, PerspectiveCamera } from "@react-three/drei";
import { useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import clsx from "clsx";
import { Eye, EyeOff, Layers, Grid } from "lucide-react";

// 🎨 구종별 색상표
const PITCH_COLORS: any = {
  FF: "#d22d49", // 4-Seam (Red)
  SI: "#fe9d00", // Sinker (Orange)
  SL: "#eee716", // Slider (Yellow)
  ST: "#fee716", // Sweeper (Gold)
  CH: "#1db05f", // Changeup (Green)
  CU: "#00d1ed", // Curveball (Cyan)
  FS: "#345fb5", // Splitter (Blue)
  FC: "#933f2c", // Cutter (Brick)
  KN: "#888888", // Knuckleball
  SV: "#ff00ff", // Slurve
};

// 📝 전체 이름 매핑
const PITCH_NAMES: Record<string, string> = {
  FF: "4-Seam Fastball", SI: "Sinker", SL: "Slider", ST: "Sweeper",
  CH: "Changeup", CU: "Curveball", FS: "Splitter", FC: "Cutter",
  KN: "Knuckleball", SV: "Slurve", KC: "Knuckle Curve", EP: "Ephus"
};

// 🔥 [수정됨] 3D 히트맵 컴포넌트 (디버깅 로그 + 투명도 렌더링 개선)
function Heatmap3D({ locations, activePitches }: { locations: any[], activePitches: string[] }) {
  // 1. 데이터 수신 확인 로그
  if (!locations || locations.length === 0) {
    console.warn("⚠️ [Heatmap] 데이터가 없습니다. api/app.py의 usecols에 plate_x, plate_z가 있는지 확인하세요.");
    return null;
  }

  const gridSize = 0.25; // 격자 크기 (0.25ft)
  
  const grid = useMemo(() => {
    const map = new Map();
    let maxCount = 0;
    
    // 2. 필터링 로직: 선택된 구종만 남김
    const filtered = locations.filter((loc: any) => {
      const type = loc.pitch_type;
      return activePitches.includes(type);
    });

    if (filtered.length === 0) return { map, maxCount: 0 };

    filtered.forEach((loc: any) => {
      // 좌표 유효성 검사
      if (typeof loc.plate_x !== 'number' || typeof loc.plate_z !== 'number') return;

      const xIndex = Math.floor((loc.plate_x + 1.75) / gridSize); 
      const zIndex = Math.floor((loc.plate_z - 1.0) / gridSize); // 바닥 1.0ft 기준

      const key = `${xIndex},${zIndex}`;
      const count = (map.get(key) || 0) + 1;
      map.set(key, count);
      if (count > maxCount) maxCount = count;
    });

    return { map, maxCount };
  }, [locations, activePitches]);

  const cells = [];
  const { map, maxCount } = grid;

  if (maxCount === 0) return null;

  for (const [key, count] of map.entries()) {
    const [xi, zi] = key.split(',').map(Number);
    
    // 📍 위치: Z = 0.05 (스트라이크 존 평면 바로 앞)
    const worldX = (xi * gridSize) - 1.75 + (gridSize/2);
    const worldZ = (zi * gridSize) + 1.0 + (gridSize/2); 

    const intensity = count / maxCount;
    // 색상: Blue(0.6) -> Red(0.0)
    const color = new THREE.Color().setHSL(0.6 - (intensity * 0.6), 1.0, 0.5); 

    cells.push(
      <mesh key={key} position={[worldX * -1, worldZ, 0.05]}>
        <boxGeometry args={[gridSize * 0.95, gridSize * 0.95, 0.05]} />
        {/* ✅ depthWrite={false} 추가: 투명한 격자가 겹칠 때 뒤가 안 보이는 문제 해결 */}
        <meshBasicMaterial 
          color={color} 
          transparent 
          opacity={0.3 + (intensity * 0.6)} 
          depthWrite={false} 
        />
      </mesh>
    );
  }

  // renderOrder를 사용하여 궤적보다 나중에 그려지거나 함께 섞이게 함
  return <group renderOrder={1}>{cells}</group>;
}

// 🧮 궤적 컴포넌트 (물리 엔진)
function Trajectory({ 
  type, speed, 
  pfx_x, pfx_z, 
  plate_x, plate_z, 
  release_pos_x, release_pos_z, release_extension, 
  hovered, setHovered,
  isVisible 
}: any) {
  if (!isVisible) return null;

  const color = PITCH_COLORS[type] || "#ffffff";
  const fullName = PITCH_NAMES[type] || type;
  const isHovered = hovered === type;
  const isDimmed = hovered && hovered !== type;

  const rX = Number.isFinite(release_pos_x) ? release_pos_x : -1.5;
  const rZ = Number.isFinite(release_pos_z) ? release_pos_z : 6.0;
  const ext = Number.isFinite(release_extension) ? release_extension : 6.0;
  const pX = Number.isFinite(plate_x) ? plate_x : 0;
  const pZ = Number.isFinite(plate_z) ? plate_z : 2.5;
  const v0_mph = Number.isFinite(speed) ? speed : 90;
  const movementX = Number.isFinite(pfx_x) ? pfx_x : 0;
  const movementZ = Number.isFinite(pfx_z) ? pfx_z : 0;

  const curve = useMemo(() => {
    const v0 = v0_mph * 1.467; 
    const startY = 60.5 - ext;
    const flightTime = startY / (v0 * 0.96); 
    const accX = (2 * ((movementX / 12) * -1)) / (flightTime * flightTime);
    
    const steps = 40;
    const points = [];
    
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * flightTime;
      const currentY = startY - (v0 * t);
      
      const startX = rX * -1;
      const endX = pX * -1;
      const vx0 = (endX - startX - (0.5 * accX * flightTime * flightTime)) / flightTime;
      const currentX = startX + (vx0 * t) + (0.5 * accX * t * t);

      const startZ = rZ;
      const endZ = pZ;
      const az = -32.174 + ((movementZ / 12) / (flightTime * flightTime) * 2);
      const vz0 = (endZ - startZ - (0.5 * az * flightTime * flightTime)) / flightTime;
      const currentZ = startZ + (vz0 * t) + (0.5 * az * t * t);

      points.push(new THREE.Vector3(currentX, currentZ, currentY));
    }
    return new THREE.CatmullRomCurve3(points);
  }, [rX, rZ, ext, pX, pZ, movementX, movementZ, v0_mph]);

  return (
    <group 
      onPointerOver={(e) => { e.stopPropagation(); setHovered(type); }} 
      onPointerOut={() => setHovered(null)}
    >
      <mesh>
        <tubeGeometry args={[curve, 64, isHovered ? 0.12 : 0.03, 8, false]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color}
          emissiveIntensity={isHovered ? 2.0 : 0.4}
          opacity={isDimmed ? 0.1 : 0.9} 
          transparent 
          metalness={0.1}
          roughness={0.1}
        />
      </mesh>
      
      <mesh position={curve.getPoint(1)}>
         <sphereGeometry args={[0.11, 16, 16]} />
         <meshStandardMaterial color="white" />
      </mesh>

      {isHovered && (
        <Text 
          position={[curve.getPoint(1).x, curve.getPoint(1).y + 0.8, curve.getPoint(1).z]} 
          fontSize={0.8} 
          color="white" 
          outlineWidth={0.05} 
          outlineColor="black" 
          anchorX="center" 
          anchorY="bottom" 
          billboard
        >
          {fullName} {Math.round(speed)}mph
        </Text>
      )}
    </group>
  );
}

// 🎥 카메라 컨트롤러
function CameraController({ view }: { view: string }) {
  const { camera } = useThree();
  useEffect(() => {
    const targetPos = {
      umpire: new THREE.Vector3(0, 5, -8),
      pitcher: new THREE.Vector3(0, 6, 65),
      batter: new THREE.Vector3(-3.5, 4, 2),
      side: new THREE.Vector3(-20, 5, 30),
      top: new THREE.Vector3(0, 50, 25)
    };
    const pos = targetPos[view as keyof typeof targetPos] || targetPos.umpire;
    camera.position.set(pos.x, pos.y, pos.z);
    camera.lookAt(0, 3, 25);
  }, [view, camera]);
  return null;
}

// 🏟️ 경기장 요소
function StadiumElements() {
  return (
    <group>
      <group position={[0, 2.5, 0]}>
        <mesh>
          <lineSegments>
            <edgesGeometry args={[new THREE.BoxGeometry(1.41, 1.8, 0.5)]} />
            <lineBasicMaterial color="#aaaaaa" linewidth={1} opacity={0.3} transparent />
          </lineSegments>
        </mesh>
      </group>
      <mesh position={[0, -0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
         <planeGeometry args={[1.41, 1.41]} />
         <meshStandardMaterial color="white" opacity={0.9} transparent side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, -0.1, 2]} rotation={[-Math.PI / 2, 0, 0]}>
         <circleGeometry args={[10, 32]} />
         <meshStandardMaterial color="#3e2723" roughness={1} /> 
      </mesh>
      <group position={[0, -0.1, 60.5]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
           <circleGeometry args={[9, 32]} />
           <meshStandardMaterial color="#5d4037" roughness={1} /> 
        </mesh>
        <mesh position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
           <planeGeometry args={[2, 0.5]} />
           <meshStandardMaterial color="white" /> 
        </mesh>
      </group>
      <gridHelper args={[150, 75, 0x333333, 0x111111]} position={[0, -0.2, 30]} />
    </group>
  );
}

// 🚀 메인 컴포넌트
export default function Pitch3D({ arsenal, locations }: { arsenal: any, locations: any[] }) {
  const [view, setView] = useState("umpire");
  const [hovered, setHovered] = useState<string | null>(null);
  const [activePitches, setActivePitches] = useState<string[]>([]);
  
  // 🔥 히트맵 토글 상태
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    if (arsenal) setActivePitches(Object.keys(arsenal));
  }, [arsenal]);

  const togglePitch = (type: string) => {
    setActivePitches(prev => prev.includes(type) ? prev.filter(p => p !== type) : [...prev, type]);
  };
  const toggleAll = () => {
    setActivePitches(activePitches.length === Object.keys(arsenal).length ? [] : Object.keys(arsenal));
  };

  return (
    <div className="w-full h-[600px] bg-[#1a1a1a] rounded-xl overflow-hidden border border-slate-700 relative shadow-2xl flex">
      
      {/* 1. 3D 뷰포트 (좌측) */}
      <div className="relative flex-1 h-full">
        {/* 헤더 */}
        <div className="absolute top-0 left-0 w-full p-4 z-10 pointer-events-none">
           <h3 className="text-white font-bold text-lg tracking-tight drop-shadow-md">
             PITCH VISUALIZATION <span className="text-xs font-normal text-slate-400 ml-2">PHYSICS V3 + HEATMAP</span>
           </h3>
        </div>

        {/* 🕹️ 히트맵 버튼 */}
        <div className="absolute top-16 left-4 z-20">
           <button 
             onClick={() => setShowHeatmap(!showHeatmap)}
             className={clsx(
               "flex items-center gap-2 px-4 py-2 rounded-lg font-bold text-xs transition-all border",
               showHeatmap 
                 ? "bg-red-500/20 border-red-500 text-red-400 shadow-[0_0_15px_rgba(239,68,68,0.3)]" 
                 : "bg-black/60 border-white/10 text-slate-400 hover:text-white"
             )}
           >
             <Grid className="w-4 h-4" /> 
             {showHeatmap ? "HEATMAP ON" : "HEATMAP OFF"}
           </button>
        </div>

        {/* 하단 뷰 컨트롤 */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 bg-black/80 p-1 rounded-full border border-white/20 backdrop-blur-md flex gap-1">
          {['umpire', 'pitcher', 'batter', 'side', 'top'].map((v) => (
            <button key={v} onClick={() => setView(v)} className={clsx("px-4 py-1.5 text-[10px] font-bold uppercase rounded-full transition-all", view === v ? "bg-white text-black" : "text-slate-400 hover:text-white")}>{v}</button>
          ))}
        </div>

        <Canvas shadows dpr={[1, 2]}>
          <PerspectiveCamera makeDefault fov={45} />
          <color attach="background" args={['#1e1e1e']} />
          <fog attach="fog" args={['#1e1e1e', 30, 150]} />
          
          <ambientLight intensity={0.4} />
          <spotLight position={[10, 80, 20]} angle={0.5} penumbra={1} intensity={1.5} castShadow />
          <pointLight position={[-10, 10, 0]} intensity={0.8} color="#ccccff" />

          <StadiumElements />
          <CameraController view={view} />
          <OrbitControls makeDefault enableZoom={true} enablePan={true} target={[0, 3, 25]} maxPolarAngle={Math.PI / 1.9} />

          {/* 🔥 히트맵 렌더링 (활성화 시, Canvas 내 마지막에 위치) */}
          {showHeatmap && locations && (
             <Heatmap3D locations={locations} activePitches={activePitches} />
          )}

          <group>
            {Object.entries(arsenal).map(([type, stats]: any) => (
              <Trajectory 
                key={type} 
                type={type} 
                speed={stats.release_speed} 
                pfx_x={stats.pfx_x} 
                pfx_z={stats.pfx_z}
                plate_x={stats.plate_x} 
                plate_z={stats.plate_z}
                release_pos_x={stats.release_pos_x} 
                release_pos_z={stats.release_pos_z} 
                release_extension={stats.release_extension}
                hovered={hovered}      
                setHovered={setHovered}
                isVisible={activePitches.includes(type)}
              />
            ))}
          </group>
        </Canvas>
      </div>

      {/* 2. 우측 컨트롤 패널 */}
      <div className="w-56 bg-black/90 border-l border-white/10 p-4 flex flex-col gap-4 overflow-y-auto z-20">
        <div className="flex justify-between items-center pb-2 border-b border-white/10">
          <span className="text-xs font-bold text-slate-400 flex items-center gap-2"><Layers className="w-4 h-4" /> FILTERS</span>
          <button onClick={toggleAll} className="text-[10px] text-blue-400 hover:text-blue-300 font-bold uppercase">{activePitches.length === Object.keys(arsenal).length ? "Hide All" : "Show All"}</button>
        </div>

        <div className="flex flex-col gap-2">
          {Object.entries(arsenal).map(([type, stats]: any) => {
            const isActive = activePitches.includes(type);
            const color = PITCH_COLORS[type] || "#ffffff";
            const fullName = PITCH_NAMES[type] || type; 
            
            return (
              <button
                key={type}
                onClick={() => togglePitch(type)}
                onMouseEnter={() => setHovered(type)}
                onMouseLeave={() => setHovered(null)}
                className={clsx(
                  "flex items-center gap-3 p-2 rounded-lg transition-all border text-left",
                  isActive ? "bg-white/10 border-white/20 opacity-100" : "bg-transparent border-transparent opacity-40 hover:opacity-70"
                )}
              >
                <div className={clsx("w-3 h-3 min-w-[12px] rounded-full shadow-sm transition-transform", isActive ? "scale-100" : "scale-75 grayscale")} style={{ backgroundColor: color }} />
                <div className="flex flex-col overflow-hidden">
                  <span className={clsx("text-xs font-bold truncate", isActive ? "text-white" : "text-slate-500")}>{fullName}</span>
                  <span className="text-[10px] text-slate-500 font-mono">{Math.round(stats.release_speed)} mph</span>
                </div>
                <div className="ml-auto text-slate-500 pl-2">
                  {isActive ? <Eye className="w-3 h-3 text-blue-400" /> : <EyeOff className="w-3 h-3" />}
                </div>
              </button>
            );
          })}
        </div>
      </div>

    </div>
  );
}