"use client";

import { useState, useMemo } from "react";
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  BarChart, Bar, ReferenceLine 
} from "recharts";
import { Activity, Move, Zap } from "lucide-react";
import clsx from "clsx";

// 🎨 구종별 색상표
const PITCH_COLORS: any = {
  FF: "#d22d49", SI: "#fe9d00", SL: "#eee716", ST: "#fee716",
  CH: "#1db05f", CU: "#00d1ed", FS: "#345fb5", FC: "#933f2c",
  KN: "#888888", SV: "#ff00ff"
};

// 툴팁 커스텀 (마우스 올렸을 때 정보 표시)
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-black/90 border border-slate-700 p-3 rounded-lg shadow-xl text-xs z-50">
        <p className="font-bold text-white mb-1">{data.pitch_type}</p>
        <p className="text-slate-300">Speed: <span className="text-white">{data.release_speed} mph</span></p>
        {data.pfx_x !== undefined && (
          <>
            <p className="text-slate-300">H-Break: <span className="text-white">{data.pfx_x} in</span></p>
            <p className="text-slate-300">V-Break: <span className="text-white">{data.pfx_z} in</span></p>
          </>
        )}
      </div>
    );
  }
  return null;
};

export default function AnalyticsCharts({ locations }: { locations: any[] }) {
  const [tab, setTab] = useState<"movement" | "velocity">("movement");

  // 1. 무브먼트 데이터 처리 (투수 시점으로 좌우 반전)
  const movementData = useMemo(() => {
    // 🛡️ [수정됨] 데이터 방어 로직 추가
    // pfx_x나 pfx_z가 숫자가 아닌 경우(null, undefined) 미리 걸러냅니다.
    return locations
      .filter(loc => typeof loc.pfx_x === 'number' && typeof loc.pfx_z === 'number') 
      .map(loc => ({
        ...loc,
        pfx_x: Number((loc.pfx_x * -1).toFixed(1)), 
        pfx_z: Number(loc.pfx_z.toFixed(1))
      }));
  }, [locations]);

  // 2. 구속 분포 데이터 처리 (히스토그램)
  const velocityData = useMemo(() => {
    const bins: Record<string, any> = {};
    
    locations.forEach(loc => {
      const speed = Math.round(loc.release_speed); // 정수로 반올림
      const type = loc.pitch_type;
      
      if (!bins[speed]) bins[speed] = { speed, total: 0 };
      if (!bins[speed][type]) bins[speed][type] = 0;
      
      bins[speed][type]++;
      bins[speed].total++;
    });

    return Object.values(bins).sort((a: any, b: any) => a.speed - b.speed);
  }, [locations]);

  // 존재하는 구종 목록 추출 (범례용)
  const pitchTypes = Array.from(new Set(locations.map(l => l.pitch_type)));

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 shadow-xl h-[400px] flex flex-col">
      
      {/* 탭 헤더 */}
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-white font-bold flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-400" /> PITCH ANALYTICS
        </h3>
        <div className="flex bg-slate-800 rounded-lg p-1">
          <button
            onClick={() => setTab("movement")}
            className={clsx(
              "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-bold transition-all",
              tab === "movement" ? "bg-blue-600 text-white" : "text-slate-400 hover:text-white"
            )}
          >
            <Move className="w-3 h-3" /> MOVEMENT
          </button>
          <button
            onClick={() => setTab("velocity")}
            className={clsx(
              "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-bold transition-all",
              tab === "velocity" ? "bg-blue-600 text-white" : "text-slate-400 hover:text-white"
            )}
          >
            <Zap className="w-3 h-3" /> VELOCITY
          </button>
        </div>
      </div>

      {/* 차트 영역 */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          {tab === "movement" ? (
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                type="number" dataKey="pfx_x" name="Horizontal" unit="in" 
                stroke="#94a3b8" label={{ value: 'Horizontal Break (in)', position: 'bottom', fill: '#94a3b8', fontSize: 12 }} 
              />
              <YAxis 
                type="number" dataKey="pfx_z" name="Vertical" unit="in" 
                stroke="#94a3b8" label={{ value: 'Vertical Break (in)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }} 
              />
              <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
              <ReferenceLine x={0} stroke="#cbd5e1" strokeOpacity={0.5} />
              <ReferenceLine y={0} stroke="#cbd5e1" strokeOpacity={0.5} />
              
              {/* 구종별로 점 찍기 */}
              {pitchTypes.map((type: any) => (
                <Scatter 
                  key={type} 
                  name={type} 
                  data={movementData.filter(d => d.pitch_type === type)} 
                  fill={PITCH_COLORS[type] || "#fff"} 
                  shape="circle"
                />
              ))}
            </ScatterChart>
          ) : (
            <BarChart data={velocityData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
              <XAxis dataKey="speed" stroke="#94a3b8" label={{ value: 'Velocity (mph)', position: 'bottom', fill: '#94a3b8', fontSize: 12 }} />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#fff' }}
                cursor={{fill: '#ffffff10'}}
              />
              {pitchTypes.map((type: any) => (
                <Bar key={type} dataKey={type} stackId="a" fill={PITCH_COLORS[type] || "#fff"} />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}