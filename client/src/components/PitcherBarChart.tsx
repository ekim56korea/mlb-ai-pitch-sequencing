"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell
} from 'recharts';

// 점수에 따른 색상 결정 함수
const getBarColor = (value: number) => {
    if (value >= 120) return "#c084fc"; // Elite (Purple)
    if (value >= 110) return "#60a5fa"; // Great (Blue)
    if (value >= 105) return "#34d399"; // Good (Green)
    if (value >= 95)  return "#94a3b8"; // Average (Gray)
    if (value >= 90)  return "#facc15"; // Below Avg (Yellow)
    return "#f87171";                   // Poor (Red)
};

export default function PitcherBarChart({ stats }: { stats: any }) {
  // RV/100을 점수화 (0 -> 100, -1 -> 120, +1 -> 80)
  // RV가 낮을수록(음수) 좋으므로, 반대로 변환
  const rvScore = 100 - (stats.run_value_per_100 * 20);

  const data = [
    { name: 'Stuff+ (구위)', value: stats.stuff_plus || 100 },
    { name: 'Loc+ (제구)', value: stats.location_plus || 100 },
    { name: 'Pitching+ (종합)', value: stats.pitching_plus || 100 },
    { name: 'Results (RV)', value: rvScore }, 
  ];

  return (
    <div className="w-full h-64 bg-slate-900/50 rounded-xl border border-slate-700 p-4 relative flex flex-col justify-center">
      <h4 className="absolute top-2 left-4 text-xs font-bold text-slate-400 uppercase">Metric Performance (Avg = 100)</h4>
      
      <ResponsiveContainer width="100%" height="90%">
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 20, right: 30, left: 40, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#334155" />
          <XAxis type="number" domain={[50, 150]} hide />
          <YAxis 
            dataKey="name" 
            type="category" 
            tick={{ fill: '#cbd5e1', fontSize: 11, fontWeight: 'bold' }} 
            width={100}
          />
          <Tooltip 
            cursor={{fill: 'transparent'}}
            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
            formatter={(value: number, name: string) => {
                if (name === 'Results (RV)') return [stats.run_value_per_100, "Actual RV/100"];
                return [value, "Score"];
            }}
          />
          {/* 기준선: 리그 평균(100) */}
          <ReferenceLine x={100} stroke="#94a3b8" strokeDasharray="3 3" label={{ position: 'top', value: 'AVG', fill: '#94a3b8', fontSize: 10 }} />
          {/* 기준선: 엘리트(120) */}
          <ReferenceLine x={120} stroke="#c084fc" strokeDasharray="3 3" label={{ position: 'top', value: 'ELITE', fill: '#c084fc', fontSize: 10 }} />
          
          <Bar dataKey="value" barSize={20} radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.value)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}