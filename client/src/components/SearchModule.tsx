"use client";

import { useState } from "react";
import axios from "axios";
import { Search, BrainCircuit, Activity, Target, Zap, ShieldAlert } from "lucide-react"; // 아이콘 추가
import Pitch3D from "./Pitch3D";
import AnalyticsCharts from "./AnalyticsCharts";
import clsx from "clsx";

// 구종 이름 매핑
const PITCH_NAMES: Record<string, string> = {
  FF: "4-Seam Fastball", SI: "Sinker", SL: "Slider", ST: "Sweeper",
  CH: "Changeup", CU: "Curveball", FS: "Splitter", FC: "Cutter",
  KN: "Knuckleball", SV: "Slurve", KC: "Knuckle Curve"
};

const PITCH_COLORS: any = {
  FF: "#d22d49", SI: "#fe9d00", SL: "#eee716", ST: "#fee716",
  CH: "#1db05f", CU: "#00d1ed", FS: "#345fb5", FC: "#933f2c",
};

export default function SearchModule() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);

  // 🎮 게임 상황 컨트롤
  const [balls, setBalls] = useState(0);
  const [strikes, setStrikes] = useState(0);
  const [outs, setOuts] = useState(0);
  const [batterStand, setBatterStand] = useState("R"); 
  
  // 🤖 AI 예측 결과
  const [predictions, setPredictions] = useState<any[]>([]);
  const [aiLoading, setAiLoading] = useState(false);

  // 🎲 [New] 시뮬레이션 결과 상태
  const [simResult, setSimResult] = useState<any>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;
    setLoading(true);
    setData(null);
    setPredictions([]);
    setSimResult(null); // 초기화
    
    try {
      const res = await axios.post("http://127.0.0.1:8000/load/matchup", null, {
        params: { pitcher_name: query, batter_name: "Ohtani" } 
      });
      setData(res.data);
    } catch (err) {
      alert("데이터 로딩 실패 (백엔드 확인 필요)");
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async () => {
    if (!data) return;
    setAiLoading(true);
    setSimResult(null);
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", {
        pitcher_name: data.pitcher.name,
        batter_stand: batterStand,
        balls: balls,
        strikes: strikes,
        outs: outs,
        inning: 1
      });
      
      if (res.data.status === "success") {
        setPredictions(res.data.predictions);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setAiLoading(false);
    }
  };

  // 🎲 [New] 시뮬레이션 실행 함수
  const runSimulation = async (pitchType: string) => {
    try {
      const res = await axios.post("http://127.0.0.1:8000/simulate_outcome", {
        pitcher_name: data.pitcher.name,
        pitch_type: pitchType,
        batter_stand: batterStand
      });
      if (res.data.status === "success") {
        setSimResult(res.data);
      }
    } catch (err) {
      console.error(err);
      alert("시뮬레이션 데이터가 부족합니다.");
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto space-y-6 animate-in fade-in duration-700 pb-20">
      
      {/* 🔍 검색창 */}
      <form onSubmit={handleSearch} className="relative group max-w-2xl mx-auto">
        <Search className="absolute left-4 top-4 w-6 h-6 text-slate-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter Pitcher Name (e.g., Skenes, Cole)..."
          className="w-full bg-slate-800/50 border border-slate-700 text-white text-xl rounded-2xl py-4 pl-14 pr-32 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button 
          type="submit" 
          disabled={loading}
          className="absolute right-2 top-2 px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold transition-all"
        >
          {loading ? "Scanning..." : "Analyze"}
        </button>
      </form>

      {/* 📊 메인 컨텐츠 */}
      {data && data.status === "success" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* 🎮 좌측: 컨트롤 & AI 패널 (4칸) */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* 1. 게임 상황 컨트롤러 */}
            <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 shadow-xl">
              <h3 className="text-white font-bold flex items-center gap-2 mb-6">
                <Activity className="w-5 h-5 text-emerald-400" /> GAME CONTEXT
              </h3>
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 font-mono text-sm">BATTER</span>
                  <div className="flex bg-slate-800 rounded-lg p-1">
                    {['L', 'R'].map((side) => (
                      <button
                        key={side}
                        onClick={() => setBatterStand(side)}
                        className={`px-4 py-1 rounded-md font-bold text-sm transition-all ${batterStand === side ? 'bg-white text-black' : 'text-slate-500'}`}
                      >
                        {side === 'L' ? 'Left' : 'Right'}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 font-mono text-sm">BALLS</span>
                    <div className="flex gap-2">
                      {[0, 1, 2, 3].map(b => (
                        <button key={b} onClick={() => setBalls(b)} className={`w-8 h-8 rounded-full font-bold text-sm transition-all ${balls >= b + 1 ? 'bg-green-500 text-black shadow-[0_0_10px_#22c55e]' : 'bg-slate-800 text-slate-600'}`}>
                          {b + 1}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 font-mono text-sm">STRIKES</span>
                    <div className="flex gap-2">
                      {[0, 1, 2].map(s => (
                        <button key={s} onClick={() => setStrikes(s)} className={`w-8 h-8 rounded-full font-bold text-sm transition-all ${strikes >= s + 1 ? 'bg-red-500 text-white shadow-[0_0_10px_#ef4444]' : 'bg-slate-800 text-slate-600'}`}>
                          {s + 1}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <button 
                  onClick={runPrediction}
                  disabled={aiLoading}
                  className="w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white rounded-xl font-bold text-lg shadow-lg flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-50"
                >
                  {aiLoading ? <span className="animate-pulse">Thinking...</span> : <><BrainCircuit className="w-5 h-5" /> PREDICT NEXT PITCH</>}
                </button>
              </div>
            </div>

            {/* 2. AI 예측 결과 (클릭 가능!) */}
            {predictions.length > 0 && (
              <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 shadow-xl animate-in slide-in-from-bottom-4">
                <h3 className="text-white font-bold mb-4 flex justify-between">
                  AI PREDICTION
                  <span className="text-[10px] font-normal text-slate-400 bg-slate-800 px-2 py-1 rounded border border-slate-700">Click to Simulate</span>
                </h3>
                <div className="space-y-3">
                  {predictions.map((pred: any, idx: number) => {
                    const color = PITCH_COLORS[pred.type] || "#ffffff";
                    const fullName = PITCH_NAMES[pred.type] || pred.type;
                    const isSelected = simResult && simResult.scenario.pitch === pred.type;
                    
                    return (
                      <button 
                        key={idx} 
                        onClick={() => runSimulation(pred.type)}
                        className={clsx(
                          "w-full text-left space-y-1 p-3 rounded-xl transition-all border",
                          isSelected 
                            ? "bg-white/10 border-white/40 shadow-lg scale-[1.02]" 
                            : "bg-slate-800/50 border-transparent hover:bg-slate-800 hover:border-slate-600"
                        )}
                      >
                        <div className="flex justify-between text-sm">
                          <span className="font-bold text-white flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full" style={{backgroundColor: color}}></div>
                            {fullName}
                          </span>
                          <span className="font-mono text-blue-300 font-bold">{pred.probability}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden">
                          <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${pred.probability}%`, backgroundColor: color }}></div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 3. 🎲 시뮬레이션 결과 카드 (조건부 렌더링) */}
            {simResult && (
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-blue-500/30 rounded-2xl p-6 shadow-2xl animate-in zoom-in-95 duration-300">
                <div className="flex items-center gap-2 mb-4 pb-2 border-b border-white/10">
                  <Target className="w-5 h-5 text-blue-400" />
                  <h3 className="text-white font-bold text-lg">EXPECTED OUTCOME</h3>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-black/40 p-3 rounded-lg text-center">
                    <span className="text-slate-400 text-[10px] font-bold uppercase block mb-1">Whiff Rate</span>
                    <span className={`text-2xl font-black ${simResult.stats.whiff_rate > 30 ? 'text-red-400' : 'text-white'}`}>
                      {simResult.stats.whiff_rate}%
                    </span>
                    <span className="text-[10px] text-slate-500 block">Swinging Strike</span>
                  </div>
                  <div className="bg-black/40 p-3 rounded-lg text-center">
                    <span className="text-slate-400 text-[10px] font-bold uppercase block mb-1">Put Away %</span>
                    <span className={`text-2xl font-black ${simResult.stats.put_away_rate > 20 ? 'text-emerald-400' : 'text-white'}`}>
                      {simResult.stats.put_away_rate}%
                    </span>
                    <span className="text-[10px] text-slate-500 block">Strikeout Efficiency</span>
                  </div>
                  <div className="bg-black/40 p-3 rounded-lg text-center col-span-2 flex justify-between items-center px-6">
                    <div className="text-left">
                      <span className="text-slate-400 text-[10px] font-bold uppercase block">Avg Velocity</span>
                      <span className="text-xl font-bold text-white flex items-center gap-1">
                        <Zap className="w-4 h-4 text-yellow-400" /> {simResult.stats.avg_speed} mph
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-slate-400 text-[10px] font-bold uppercase block">Hard Hit %</span>
                      <span className="text-xl font-bold text-white flex items-center gap-1 justify-end">
                        <ShieldAlert className="w-4 h-4 text-orange-400" /> {simResult.stats.hard_hit_rate}%
                      </span>
                    </div>
                  </div>
                </div>
                <div className="mt-3 text-[10px] text-slate-500 text-center font-mono">
                  Based on {simResult.scenario.sample_size} historical pitches vs {simResult.scenario.batter}
                </div>
              </div>
            )}

          </div>

          {/* ⚾️ 우측: 3D 및 차트 */}
          <div className="lg:col-span-8 space-y-6">
             <Pitch3D arsenal={data.pitcher.arsenal} locations={data.pitcher.locations} />
             {data.pitcher.locations && <AnalyticsCharts locations={data.pitcher.locations} />}
          </div>

        </div>
      )}
    </div>
  );
}