import React from 'react';
import dynamic from 'next/dynamic';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// Plotly 로딩 (SSR 비활성화)
const Plot = dynamic(() => import("react-plotly.js"), { 
  ssr: false,
  loading: () => <div className="h-full flex items-center justify-center text-slate-500 font-mono">Loading Pitch Commander v7.0 Engine...</div>
});

interface TrajectoryChartProps {
  // 이제 단일 궤적이 아니라 '여러 개의 궤적'을 받습니다.
  trajectories: { 
    pitchType: string; 
    data: { x: number; y: number; z: number }[]; 
    metrics: any 
  }[];
}

const PITCH_COLORS: Record<string, string> = {
  FF: '#ef4444', // 포심 (Red)
  SL: '#eab308', // 슬라이더 (Yellow)
  CH: '#22c55e', // 체인지업 (Green)
  CU: '#06b6d4', // 커브 (Cyan)
  SI: '#f97316', // 싱커 (Orange)
  FC: '#b45309', // 커터 (Brown)
  ST: '#ec4899', // 스위퍼 (Pink)
  SV: '#ec4899',
  SPL: '#6366f1', // 스플리터
};

export default function TrajectoryChart({ trajectories }: TrajectoryChartProps) {
  // 데이터가 없어도 경기장은 보여주기 위해 null 체크를 렌더링 내부로 미룸
  
  // Plotly 데이터 배열 생성
  const plotData: any[] = [];

  // 1. 경기장 바닥 (잔디)
  plotData.push({
    type: 'mesh3d',
    x: [-15, 15, 15, -15],
    y: [-5, -5, 70, 70], // 포수 뒤 ~ 마운드 뒤
    z: [-0.1, -0.1, -0.1, -0.1],
    color: '#1a2e1a', // 짙은 잔디색
    opacity: 0.8,
    name: 'Field',
    hoverinfo: 'none'
  });

  // 2. 마운드 (Mound) - 솟아오른 흙
  // 마운드 중심: (0, 60.5), 반경 약 9ft
  const theta = Array.from({length: 30}, (_, i) => (i * 2 * Math.PI) / 29);
  const r = 9; // 마운드 반경
  const moundX = theta.map(t => r * Math.cos(t));
  const moundY = theta.map(t => 60.5 + r * Math.sin(t));
  const moundZ = theta.map(() => 0.05); // 약간 위로
  
  // 마운드 흙 (원형 메쉬)
  plotData.push({
    type: 'mesh3d',
    x: [...moundX, 0], 
    y: [...moundY, 60.5], 
    z: [...moundZ, 0.8], // 중앙이 10인치(0.8ft) 높음
    color: '#5c4033', // 흙색
    opacity: 1,
    alphahull: 0,
    name: 'Mound',
    hoverinfo: 'none'
  });

  // 3. 투수판 (Rubber)
  plotData.push({
    type: 'mesh3d',
    x: [-1, 1, 1, -1],
    y: [60.5, 60.5, 61, 61],
    z: [0.85, 0.85, 0.85, 0.85],
    color: 'white',
    name: 'Rubber'
  });

  // 4. 홈플레이트 (수정됨: 뾰족한 부분이 뒤로 가도록)
  // 투수가 볼 때: 평평한 면이 앞(60.5쪽), 뾰족한 면이 뒤(0쪽)
  // 실제 좌표: y=1.417(앞) ~ y=0(뒤, 뾰족끝)
  plotData.push({
    type: 'mesh3d',
    x: [0, 0.71, 0.71, -0.71, -0.71],
    y: [0, 0.5, 1.417, 1.417, 0.5], // y=0이 뾰족한 끝(포수쪽)
    z: [0.05, 0.05, 0.05, 0.05, 0.05],
    color: 'white',
    name: 'Home Plate'
  });

  // 5. 배터 박스 (라인)
  const boxLineColor = 'rgba(255, 255, 255, 0.8)';
  plotData.push(
    {
      type: 'scatter3d', mode: 'lines',
      x: [-3.5, -3.5, -0.5, -0.5, -3.5], // 좌타석
      y: [0, 4, 4, 0, 0],
      z: [0.05, 0.05, 0.05, 0.05, 0.05],
      line: { color: boxLineColor, width: 5 }, hoverinfo: 'none'
    },
    {
      type: 'scatter3d', mode: 'lines',
      x: [3.5, 3.5, 0.5, 0.5, 3.5], // 우타석
      y: [0, 4, 4, 0, 0],
      z: [0.05, 0.05, 0.05, 0.05, 0.05],
      line: { color: boxLineColor, width: 5 }, hoverinfo: 'none'
    }
  );

  // 6. 3D 스트라이크 존 (입체 박스)
  // 규정: 홈플레이트 너비 17인치(0.71ft) + 공 반개 여유
  plotData.push({
    type: 'mesh3d',
    // 8개 꼭짓점
    x: [-0.8, 0.8, 0.8, -0.8, -0.8, 0.8, 0.8, -0.8],
    y: [0, 0, 0, 0, 1.4, 1.4, 1.4, 1.4], // 홈플레이트 깊이만큼
    z: [1.6, 1.6, 3.5, 3.5, 1.6, 1.6, 3.5, 3.5],
    i: [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    j: [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    k: [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
    opacity: 0.1, // 투명하게
    color: '#ff0000', // 붉은 핫존 느낌
    name: 'Strike Zone'
  });

  // 7. [핵심] 투구 궤적들 (Trajectories)
  if (trajectories && trajectories.length > 0) {
    trajectories.forEach((traj, idx) => {
      // 좌표 변환: x는 좌우 반전(포수시점), y는 그대로, z는 그대로
      const x = traj.data.map(p => p.x * -1); 
      const y = traj.data.map(p => p.y);
      const z = traj.data.map(p => p.z);
      
      const color = PITCH_COLORS[traj.pitchType] || '#ffffff';
      const isPrimary = idx === 0; // 첫 번째 궤적(1순위)은 더 진하게

      plotData.push({
        type: 'scatter3d',
        mode: 'lines',
        x: x, y: y, z: z,
        line: { width: isPrimary ? 8 : 4, color: color },
        opacity: isPrimary ? 1.0 : 0.4, // 순위 낮은건 흐리게
        name: `${traj.pitchType} (${idx+1})`
      });

      // 공 위치 (마지막 지점)
      plotData.push({
        type: 'scatter3d',
        mode: 'markers',
        x: [x[x.length-1]], y: [y[y.length-1]], z: [z[z.length-1]],
        marker: { size: 5, color: 'white', line: {color: 'black', width: 1} },
        name: 'Ball',
        showlegend: false
      });
    });
  }

  return (
    <Card className="bg-slate-950 border-slate-800 shadow-2xl h-full overflow-hidden">
      <CardHeader className="pb-2 border-b border-slate-800 bg-slate-900/50 absolute top-0 left-0 w-full z-10">
        <CardTitle className="text-sm font-bold text-white uppercase flex justify-between items-center px-4">
          <span className="flex items-center gap-2 drop-shadow-md">
             🏟️ Pitch Commander Pro v7.0 (Multi-Arsenal)
          </span>
          {trajectories.length > 0 && (
            <div className="flex gap-2">
                {trajectories.map((t, i) => (
                    <span key={i} className="text-[10px] font-mono px-2 py-1 rounded bg-slate-800 text-white border border-slate-700">
                        <span style={{color: PITCH_COLORS[t.pitchType]}}>●</span> {t.pitchType}
                    </span>
                ))}
            </div>
          )}
        </CardTitle>
      </CardHeader>
      
      <CardContent className="h-full p-0 relative">
        <Plot
          data={plotData}
          layout={{
            autosize: true,
            paper_bgcolor: '#0f172a', // Slate-900
            plot_bgcolor: '#0f172a',
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: true,
            legend: { x: 0, y: 1, font: {color: 'white'} },
            scene: {
              // 축 범위 고정 (공이 중간에 짤리지 않게)
              xaxis: { title: '', range: [-5, 5], visible: false },
              yaxis: { title: '', range: [-2, 65], visible: false }, // -2까지 늘려서 포수 뒤까지 보이게
              zaxis: { title: '', range: [0, 10], visible: false },
              camera: {
                eye: { x: -1.5, y: 0.1, z: 0.5 }, // 포수 어깨 너머 시점 (박진감)
                center: { x: 0, y: 0, z: -0.1 }
              },
              aspectmode: 'manual',
              aspectratio: { x: 1, y: 4, z: 1 }
            }
          }}
          useResizeHandler={true}
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false }}
        />
      </CardContent>
    </Card>
  );
}