"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Text, PerspectiveCamera, Line } from "@react-three/drei";
import { useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import clsx from "clsx";
import { Eye, EyeOff, Layers, Grid, Zap } from "lucide-react";

// 🎨 구종별 색상표
const PITCH_COLORS: any = {
  FF: "#d22d49", SI: "#fe9d00", SL: "#eee716", ST: "#fee716",
  CH: "#1db05f", CU: "#00d1ed", FS: "#345fb5", FC: "#933f2c",
  KN: "#888888", SV: "#ff00ff"
};

const PITCH_NAMES: Record<string, string> = {
  FF: "4-Seam", SI: "Sinker", SL: "Slider", ST: "Sweeper",
  CH: "Changeup", CU: "Curve", FS: "Splitter", FC: "Cutter",
  KN: "Knuckle", SV: "Slurve"
};

// 🏟️ [Visual Upgrade] 경기장 요소
function StadiumElements() {
  // 홈플레이트 모양 (오각형) - 뾰족한 끝이 (0,0)
  const homePlateShape = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0, 0); 
    shape.lineTo(0.71, 0.71);  
    shape.lineTo(0.71, 1.42);  
    shape.lineTo(-0.71, 1.42); 
    shape.lineTo(-0.71, 0.71); 
    shape.lineTo(0, 0);        
    return shape;
  }, []);

  return (
    <group>
      {/* 1. 홈플레이트 (평면 오각형, 흰색) */}
      {/* rotation: X축 -90도(눕힘), Z축 180도(투수 방향 바라봄) */}
      <group position={[0, -0.09, 0]} rotation={[-Math.PI / 2, 0, Math.PI]}>
         <mesh>
            <shapeGeometry args={[homePlateShape]} />
            <meshBasicMaterial color="white" side={THREE.DoubleSide} />
         </mesh>
         <lineSegments position={[0, 0, 0.01]}>
             <edgesGeometry args={[new THREE.ShapeGeometry(homePlateShape)]} />
             <lineBasicMaterial color="#333" linewidth={2} />
         </lineSegments>
      </group>

      {/* 2. 배터 박스 (규격: 4ft x 6ft) */}
      <group position={[0, -0.08, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        {/* 우타자 박스 */}
        <group position={[3, 0, 0]}>
           <lineSegments>
              <edgesGeometry args={[new THREE.PlaneGeometry(4, 6)]} />
              <lineBasicMaterial color="white" linewidth={2} opacity={0.8} transparent />
           </lineSegments>
        </group>
        {/* 좌타자 박스 */}
        <group position={[-3, 0, 0]}>
           <lineSegments>
              <edgesGeometry args={[new THREE.PlaneGeometry(4, 6)]} />
              <lineBasicMaterial color="white" linewidth={2} opacity={0.8} transparent />
           </lineSegments>
        </group>
      </group>

      {/* 3. 캐쳐 박스: 요청에 따라 삭제됨 */}

      {/* 4. 파울 라인 (배터 박스 내부 제거) */}
      {/* 배터 박스 앞쪽(Z=3)부터 시작되도록 조정 */}
      <group position={[0, -0.08, 0]}>
        {/* 1루 라인 (45도) */}
        <Line 
          points={[[3, 0, 3], [70, 0, 70]]} 
          color="white" lineWidth={3} 
        />
        {/* 3루 라인 (-45도) */}
        <Line 
          points={[[-3, 0, 3], [-70, 0, 70]]} 
          color="white" lineWidth={3} 
        />
      </group>

      {/* 5. 스트라이크 존 가이드 (흰색) */}
      <group position={[0, 2.5, 0]}>
        <mesh>
          <lineSegments>
            <edgesGeometry args={[new THREE.BoxGeometry(1.41, 1.8, 0.5)]} />
            <lineBasicMaterial color="white" linewidth={1} opacity={0.6} transparent />
          </lineSegments>
        </mesh>
      </group>

      {/* 6. 마운드 */}
      <group position={[0, -0.1, 60.5]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
           <circleGeometry args={[9, 64]} />
           <meshStandardMaterial color="#5d4037" roughness={1} /> 
        </mesh>
        <mesh position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
           <planeGeometry args={[2, 0.5]} />
           <meshStandardMaterial color="white" /> 
        </mesh>
      </group>
      
      {/* 잔디 그리드 */}
      <gridHelper args={[200, 100, 0x1a472a, 0x0f2f1a]} position={[0, -0.2, 50]} />
    </group>
  );
}

// 🎥 카메라 컨트롤러 (구버전 시점 복원)
function CameraController({ view, isTunnelMode }: { view: string, isTunnelMode: boolean }) {
  const { camera } = useThree();
  
  useEffect(() => {
    // 1. 터널 모드 (투구 궤적 추적용 특수 뷰)
    if (isTunnelMode) {
      camera.position.set(0, 3.5, -2);
      camera.lookAt(0, 5, 55);
      return;
    } 

    // 2. 구버전 시점 좌표 복원 (Umpire, Pitcher, Batter, Side, Top)
    const views: any = {
      umpire: { pos: new THREE.Vector3(0, 5, -8), lookAt: new THREE.Vector3(0, 2, 25) },
      pitcher: { pos: new THREE.Vector3(0, 6, 65), lookAt: new THREE.Vector3(0, 1, 0) },
      batter: { pos: new THREE.Vector3(-3.5, 4, 2), lookAt: new THREE.Vector3(0, 3, 40) },
      side: { pos: new THREE.Vector3(-20, 5, 30), lookAt: new THREE.Vector3(0, 3, 25) },
      top: { pos: new THREE.Vector3(0, 50, 25), lookAt: new THREE.Vector3(0, 0, 25) }
    };
    
    // 기본값은 Umpire
    const v = views[view] || views.umpire;
    
    // 부드러운 전환 (lerp는 useFrame에서 써야하지만 여기선 set으로 즉시 이동 후 lookAt)
    // useEffect 내이므로 즉시 이동이 더 안정적임
    camera.position.copy(v.pos);
    camera.lookAt(v.lookAt);

  }, [view, isTunnelMode, camera]);

  return null;
}

// 🔥 3D 히트맵 (변경 없음)
function Heatmap3D({ locations, activePitches }: { locations: any[], activePitches: string[] }) {
    if (!locations || locations.length === 0) return null;
    const gridSize = 0.25;
    const grid = useMemo(() => {
        const map = new Map();
        let maxCount = 0;
        const filtered = locations.filter((loc: any) => activePitches.includes(loc.pitch_type));
        if (filtered.length === 0) return { map, maxCount: 0 };
        filtered.forEach((loc: any) => {
            if (typeof loc.plate_x !== 'number' || typeof loc.plate_z !== 'number') return;
            const xIndex = Math.floor((loc.plate_x + 1.75) / gridSize); 
            const zIndex = Math.floor((loc.plate_z - 1.0) / gridSize);
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
        const worldX = (xi * gridSize) - 1.75 + (gridSize/2);
        const worldZ = (zi * gridSize) + 1.0 + (gridSize/2); 
        const intensity = count / maxCount;
        const color = new THREE.Color().setHSL(0.6 - (intensity * 0.6), 1.0, 0.5); 
        cells.push(
            <mesh key={key} position={[worldX * -1, worldZ, 0.05]}>
                <boxGeometry args={[gridSize * 0.95, gridSize * 0.95, 0.05]} />
                <meshBasicMaterial color={color} transparent opacity={0.3 + (intensity * 0.6)} depthWrite={false} />
            </mesh>
        );
    }
    return <group renderOrder={1}>{cells}</group>;
}

// 🧮 궤적 컴포넌트
function SplitTrajectory({ 
  type, speed, pfx_x, pfx_z, plate_x, plate_z, 
  release_pos_x, release_pos_z, release_extension, 
  isTunnelMode, hovered, setHovered, isVisible
}: any) {
  
  if (!isVisible) return null;

  const rX = release_pos_x ?? -1.5;
  const rZ = release_pos_z ?? 6.0;
  const ext = release_extension ?? 6.0;
  const pX = plate_x ?? 0; 
  const pZ = plate_z ?? 2.5;
  const v0_mph = speed ?? 90;
  const movementX = pfx_x ?? 0; 
  const movementZ = pfx_z ?? 0;

  const color = PITCH_COLORS[type] || "#ffffff";
  const fullName = PITCH_NAMES[type] || type;

  const fullPoints = useMemo(() => {
    const v0 = v0_mph * 1.467; 
    const startY = 60.5 - ext;
    const flightTime = startY / (v0 * 0.96); 
    const accX = (2 * ((movementX / 12) * -1)) / (flightTime * flightTime);
    const az = -32.174 + ((movementZ / 12) / (flightTime * flightTime) * 2);
    
    const points = [];
    const steps = 40;
    
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * flightTime;
      const currentY = startY - (v0 * t);
      
      const startX = rX * -1;
      const endX = pX * -1;
      const vx0 = (endX - startX - (0.5 * accX * flightTime * flightTime)) / flightTime;
      const currentX = startX + (vx0 * t) + (0.5 * accX * t * t);

      const startZ = rZ;
      const endZ = pZ;
      const vz0 = (endZ - startZ - (0.5 * az * flightTime * flightTime)) / flightTime;
      const currentZ = startZ + (vz0 * t) + (0.5 * az * t * t);

      points.push(new THREE.Vector3(currentX, currentZ, currentY));
    }
    return points;
  }, [rX, rZ, ext, pX, pZ, movementX, movementZ, v0_mph]);

  const splitIndex = Math.floor(fullPoints.length * 0.5); 
  const tunnelPoints = fullPoints.slice(0, splitIndex + 1);
  const breakPoints = fullPoints.slice(splitIndex);

  const tunnelCurve = useMemo(() => new THREE.CatmullRomCurve3(tunnelPoints), [tunnelPoints]);
  const breakCurve = useMemo(() => new THREE.CatmullRomCurve3(breakPoints), [breakPoints]);

  const isHovered = hovered === type;
  const isDimmed = hovered && hovered !== type;

  return (
    <group 
      onPointerOver={(e) => { e.stopPropagation(); setHovered(type); }} 
      onPointerOut={() => setHovered(null)}
    >
      <mesh>
        <tubeGeometry args={[tunnelCurve, 20, isHovered ? 0.12 : 0.06, 8, false]} />
        <meshStandardMaterial 
          color={isTunnelMode ? "#cbd5e1" : color} 
          emissive={isTunnelMode ? "#94a3b8" : color}
          emissiveIntensity={isTunnelMode ? 0.3 : 0.5}
          opacity={isDimmed ? 0.1 : 0.8}
          transparent
        />
      </mesh>
      <mesh>
        <tubeGeometry args={[breakCurve, 20, isHovered ? 0.12 : 0.06, 8, false]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color}
          emissiveIntensity={isHovered ? 2.0 : 0.8}
          opacity={isDimmed ? 0.1 : 1}
          transparent
        />
      </mesh>
      <mesh position={fullPoints[fullPoints.length - 1]}>
         <sphereGeometry args={[0.11, 16, 16]} />
         <meshStandardMaterial color={color} />
      </mesh>

      {isHovered && (
        <Text 
          position={[fullPoints[fullPoints.length - 1].x, fullPoints[fullPoints.length - 1].y + 0.8, fullPoints[fullPoints.length - 1].z]} 
          fontSize={0.8} color="white" outlineWidth={0.05} outlineColor="black" anchorX="center" anchorY="bottom" billboard
        >
          {fullName} {Math.round(speed)}mph
        </Text>
      )}
    </group>
  );
}

// 🚀 Pitch3D 메인 컴포넌트
export default function Pitch3D({ arsenal, locations }: { arsenal: any, locations: any[] }) {
  // 기본 시점을 'umpire'로 설정 (구버전 복원)
  const [view, setView] = useState("umpire");
  const [hovered, setHovered] = useState<string | null>(null);
  const [activePitches, setActivePitches] = useState<string[]>([]);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showTunnel, setShowTunnel] = useState(false);

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
      <div className="relative flex-1 h-full">
        {/* 헤더 */}
        <div className="absolute top-0 left-0 w-full p-4 z-10 pointer-events-none">
           <h3 className="text-white font-bold text-lg tracking-tight drop-shadow-md flex items-center gap-2">
             PITCH VISUALIZATION 
             {showTunnel && <span className="bg-indigo-600 text-[10px] px-2 py-0.5 rounded text-white animate-pulse">TUNNEL MODE</span>}
           </h3>
        </div>

        {/* 컨트롤 버튼 */}
        <div className="absolute top-16 left-4 z-20 flex flex-col gap-2">
           <button onClick={() => setShowHeatmap(!showHeatmap)} className={clsx("flex items-center gap-2 px-4 py-2 rounded-lg font-bold text-xs border transition-all", showHeatmap ? "bg-red-500/20 border-red-500 text-red-400" : "bg-black/60 border-white/10 text-slate-400")}>
             <Grid className="w-4 h-4" /> {showHeatmap ? "HEATMAP ON" : "HEATMAP OFF"}
           </button>
           <button onClick={() => setShowTunnel(!showTunnel)} className={clsx("flex items-center gap-2 px-4 py-2 rounded-lg font-bold text-xs border transition-all", showTunnel ? "bg-indigo-500/20 border-indigo-500 text-indigo-400" : "bg-black/60 border-white/10 text-slate-400")}>
             <Zap className="w-4 h-4" /> {showTunnel ? "TUNNELING ON" : "TUNNELING OFF"}
           </button>
        </div>

        {/* 뷰 포인트 버튼 (복구됨: umpire, pitcher, batter, side, top) */}
        {!showTunnel && (
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 bg-black/80 p-1 rounded-full border border-white/20 backdrop-blur-md flex gap-1">
            {['umpire', 'pitcher', 'batter', 'side', 'top'].map((v) => (
              <button key={v} onClick={() => setView(v)} className={clsx("px-4 py-1.5 text-[10px] font-bold uppercase rounded-full transition-all", view === v ? "bg-white text-black" : "text-slate-400 hover:text-white")}>{v}</button>
            ))}
          </div>
        )}

        {/* 3D 캔버스 */}
        <Canvas shadows dpr={[1, 2]}>
          <PerspectiveCamera makeDefault fov={45} />
          <color attach="background" args={['#1e1e1e']} />
          <fog attach="fog" args={['#1e1e1e', 30, 150]} />
          
          <ambientLight intensity={0.4} />
          <spotLight position={[10, 80, 20]} angle={0.5} penumbra={1} intensity={1.5} castShadow />
          <pointLight position={[-10, 10, 0]} intensity={0.8} color="#ccccff" />

          {/* 🏟️ 리얼한 경기장 (수정됨) */}
          <StadiumElements />
          
          <CameraController view={view} isTunnelMode={showTunnel} />
          <OrbitControls makeDefault enableZoom={true} enablePan={true} target={[0, 2, 25]} maxPolarAngle={Math.PI / 1.9} />

          {/* 히트맵 */}
          {showHeatmap && locations && <Heatmap3D locations={locations} activePitches={activePitches} />}

          {/* ⚾️ 궤적 */}
          <group>
            {Object.entries(arsenal).map(([type, stats]: any) => (
              <SplitTrajectory 
                key={type} 
                type={type} 
                speed={stats.release_speed} 
                pfx_x={stats.pfx_x} pfx_z={stats.pfx_z}
                plate_x={stats.plate_x} plate_z={stats.plate_z}
                release_pos_x={stats.release_pos_x} release_pos_z={stats.release_pos_z} 
                release_extension={stats.release_extension}
                hovered={hovered} setHovered={setHovered}
                isTunnelMode={showTunnel}
                isVisible={activePitches.includes(type)}
              />
            ))}
          </group>
        </Canvas>
        
        {showTunnel && (
            <div className="absolute bottom-10 w-full text-center pointer-events-none">
                <span className="bg-black/50 text-white px-4 py-2 rounded-full text-sm font-bold backdrop-blur-sm border border-white/20">
                   👀 Batter's Eye View (Color Deception Active)
                </span>
            </div>
        )}
      </div>

      <div className="w-56 bg-black/90 border-l border-white/10 p-4 flex flex-col gap-4 overflow-y-auto z-20">
        <div className="flex justify-between items-center pb-2 border-b border-white/10">
          <span className="text-xs font-bold text-slate-400 flex items-center gap-2"><Layers className="w-4 h-4" /> FILTERS</span>
          <button onClick={toggleAll} className="text-[10px] text-blue-400 hover:text-blue-300 font-bold uppercase">{activePitches.length === Object.keys(arsenal).length ? "Hide All" : "Show All"}</button>
        </div>
        <div className="flex flex-col gap-2">
          {Object.entries(arsenal).map(([type, stats]: any) => {
            const isActive = activePitches.includes(type);
            const color = PITCH_COLORS[type] || "#ffffff";
            return (
              <button
                key={type}
                onClick={() => togglePitch(type)}
                onMouseEnter={() => setHovered(type)}
                onMouseLeave={() => setHovered(null)}
                className={clsx("flex items-center gap-3 p-2 rounded-lg transition-all border text-left", isActive ? "bg-white/10 border-white/20 opacity-100" : "bg-transparent border-transparent opacity-40 hover:opacity-70")}
              >
                <div className={clsx("w-3 h-3 min-w-[12px] rounded-full shadow-sm", isActive ? "scale-100" : "scale-75 grayscale")} style={{ backgroundColor: color }} />
                <div className="flex flex-col overflow-hidden">
                  <span className={clsx("text-xs font-bold truncate", isActive ? "text-white" : "text-slate-500")}>{PITCH_NAMES[type] || type}</span>
                  <span className="text-[10px] text-slate-500 font-mono">{Math.round(stats.release_speed || 0)} mph</span>
                </div>
                <div className="ml-auto text-slate-500 pl-2">{isActive ? <Eye className="w-3 h-3 text-blue-400" /> : <EyeOff className="w-3 h-3" />}</div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}