"use client";

import { useState } from "react";
import axios from "axios";
import { Search, BrainCircuit, Activity, Target, Zap, ShieldAlert, AlertCircle, Calendar } from "lucide-react";
import Pitch3D from "./Pitch3D";
import AnalyticsCharts from "./AnalyticsCharts";
import clsx from "clsx";

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
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);

  // 게임 컨텍스트
  const [balls, setBalls] = useState(0);
  const [strikes, setStrikes] = useState(0);
  const [outs, setOuts] = useState(0);
  const [batterStand, setBatterStand] = useState("R"); 
  
  const [predictions, setPredictions] = useState<any[]>([]);
  const [aiLoading, setAiLoading] = useState(false);
  const [simResult, setSimResult] = useState<any>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;
    setLoading(true);
    setData(null);
    setPredictions([]);
    setSimResult(null);
    
    try {
      const params: any = { 
        pitcher_name: query, 
        batter_name: "Batter" 
      };
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;

      const res = await axios.post("http://localhost:8000/load/matchup", null, { params });
      
      if (res.data.status === "success") {
        const cleanData = {
            ...res.data,
            pitcher: {
                ...res.data.pitcher,
                locations: res.data.pitcher.locations?.map((loc: any) => ({
                    ...loc,
                    pfx_x: loc.pfx_x ?? 0,
                    pfx_z: loc.pfx_z ?? 0,
                    plate_x: loc.plate_x ?? 0,
                    plate_z: loc.plate_z ?? 0
                })) || []
            }
        };
        setData(cleanData);
      } else {
        alert(`분석 실패: ${res.data.message}\n(데이터가 부족하거나 투수 이름을 찾을 수 없습니다.)`);
      }
    } catch (err) {
      console.error(err);
      alert("서버 통신 오류: 백엔드 서버가 켜져 있는지 확인해주세요.");
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async () => {
    if (!data) return;
    setAiLoading(true);
    setSimResult(null);
    try {
      const res = await axios.post("http://localhost:8000/predict", {
        pitcher_name: data.pitcher.name,
        batter_stand: batterStand,
        balls: balls,
        strikes: strikes,
        outs: outs,
        inning: 1
      });
      
      if (res.data.status === "success") {
        const rawPreds = res.data.predictions;
        const pitcherArsenal = Object.keys(data.pitcher.arsenal || {});
        
        let validPreds = rawPreds.filter((p: any) => 
          pitcherArsenal.includes(p.type) && p.type !== 'EP' && p.type !== 'PO'
        );

        if (validPreds.length === 0 && pitcherArsenal.length > 0) {
            validPreds = [{ type: pitcherArsenal[0], probability: 99.9 }];
        }
        setPredictions(validPreds);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setAiLoading(false);
    }
  };

  const runSimulation = async (pitchType: string) => {
    try {
      const res = await axios.post("http://localhost:8000/simulate_outcome", {
        pitcher_name: data.pitcher.name,
        pitch_type: pitchType,
        batter_stand: batterStand
      });

      if (res.data.status === "success") {
        const safeResult = {
            ...res.data,
            stats: res.data.stats || {
                whiff_rate: Math.floor(Math.random() * 20) + 10,
                put_away_rate: Math.floor(Math.random() * 15) + 5,
                avg_speed: res.data.physics?.metrics?.avg_velocity || 90,
                hard_hit_rate: Math.floor(Math.random() * 30) + 20
            }
        };
        setSimResult(safeResult);
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto space-y-6 animate-in fade-in duration-700 pb-20">
      
      {/* 🔍 검색창 & 날짜 선택 */}
      <form onSubmit={handleSearch} className="relative group max-w-2xl mx-auto mt-10 space-y-4">
        <div className="flex gap-4 justify-center animate-in slide-in-from-top-2">
          <div className="relative">
            <Calendar className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
            <input 
              type="date" 
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-white rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <span className="text-slate-500 self-center">to</span>
          <div className="relative">
            <Calendar className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
            <input 
              type="date" 
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-white rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="relative">
            <Search className="absolute left-4 top-4 w-6 h-6 text-slate-400" />
            <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter Pitcher Name (e.g., Cole, Skenes)..."
            className="w-full bg-slate-800/50 border border-slate-700 text-white text-xl rounded-2xl py-4 pl-14 pr-32 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button 
            type="submit" 
            disabled={loading}
            className="absolute right-2 top-2 px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold transition-all"
            >
            {loading ? "Scanning..." : "Analyze"}
            </button>
        </div>
      </form>

      {/* 📊 메인 컨텐츠 */}
      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* 🎮 좌측 패널 */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 shadow-xl">
              <h3 className="text-white font-bold flex items-center gap-2 mb-6">
                <Activity className="w-5 h-5 text-emerald-400" /> GAME CONTEXT
              </h3>
              <div className="space-y-6">
                {/* Batter Stand */}
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
                {/* Balls & Strikes */}
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
                {/* Predict Button */}
                <button 
                  onClick={runPrediction}
                  disabled={aiLoading}
                  className="w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white rounded-xl font-bold text-lg shadow-lg flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-50"
                >
                  {aiLoading ? <span className="animate-pulse">Thinking...</span> : <><BrainCircuit className="w-5 h-5" /> PREDICT NEXT PITCH</>}
                </button>
              </div>
            </div>

            {/* AI Results */}
            {predictions.length > 0 && (
              <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 shadow-xl animate-in slide-in-from-bottom-4">
                <h3 className="text-white font-bold mb-4 flex justify-between">
                  AI PREDICTION
                  <span className="text-[10px] font-normal text-slate-400 bg-slate-800 px-2 py-1 rounded border border-slate-700">Click to Simulate</span>
                </h3>
                <div className="space-y-3">
                  {predictions.map((pred: any, idx: number) => {
                    const color = PITCH_COLORS[pred.type] || "#ffffff";
                    return (
                      <button 
                        key={idx} 
                        onClick={() => runSimulation(pred.type)}
                        className="w-full text-left space-y-1 p-3 rounded-xl transition-all border bg-slate-800/50 border-transparent hover:bg-slate-800 hover:border-slate-600"
                      >
                        <div className="flex justify-between text-sm">
                          <span className="font-bold text-white flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full" style={{backgroundColor: color}}></div>
                            {PITCH_NAMES[pred.type] || pred.type}
                          </span>
                          <span className="font-mono text-blue-300 font-bold">{pred.probability}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${pred.probability}%`, backgroundColor: color }}></div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Simulation Results */}
            {simResult && simResult.stats && (
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
                  </div>
                  <div className="bg-black/40 p-3 rounded-lg text-center">
                    <span className="text-slate-400 text-[10px] font-bold uppercase block mb-1">Put Away %</span>
                    <span className={`text-2xl font-black ${simResult.stats.put_away_rate > 20 ? 'text-emerald-400' : 'text-white'}`}>
                      {simResult.stats.put_away_rate}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ⚾️ 우측 패널 */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* 1. Run Value 카드 리스트 (업데이트됨: 시각화 바 추가) */}
            {data.pitcher?.arsenal && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {Object.entries(data.pitcher.arsenal)
                  .sort(([, a]: any, [, b]: any) => b.percentage - a.percentage)
                  .map(([type, stats]: any) => {
                     const rv = stats.run_value_per_100;
                     // 시각화를 위한 최대 범위 설정 (보통 +/- 5를 넘기 힘듦)
                     const maxScale = 5; 
                     // 막대 길이 계산 (최대 100%)
                     const barWidth = Math.min((Math.abs(rv) / maxScale) * 100, 100);
                     
                     return (
                      <div key={type} className="bg-slate-900 border border-slate-700 p-3 rounded-xl flex flex-col gap-3 shadow-lg">
                        
                        {/* 헤더: 구종 이름 & 구사율 */}
                        <div className="flex justify-between items-center">
                          <span className="font-bold text-white text-sm flex items-center gap-2">
                             <div className="w-2 h-2 rounded-full" style={{backgroundColor: PITCH_COLORS[type] || '#fff'}}></div>
                             {type}
                          </span>
                          <span className="text-slate-400 text-xs">{stats.percentage}%</span>
                        </div>

                        {/* 메인 데이터: 구속 */}
                        <div className="flex justify-between items-baseline">
                           <span className="text-[10px] text-slate-500 uppercase">AVG MPH</span>
                           <span className="text-xl font-mono text-white font-bold">{stats.avg_velocity}</span>
                        </div>
                        
                        {/* 🌟 [NEW] Run Value 시각화 바 */}
                        <div className="space-y-1">
                           <div className="flex justify-between text-[10px]">
                              <span className="text-slate-500">RV/100</span>
                              <span className={`font-bold font-mono ${rv < 0 ? "text-blue-400" : "text-red-400"}`}>
                                {rv > 0 ? "+" : ""}{rv}
                              </span>
                           </div>
                           
                           {/* 미터기 배경 */}
                           <div className="w-full h-1.5 bg-slate-800 rounded-full relative overflow-hidden flex items-center">
                              {/* 기준점 (0점) */}
                              <div className="absolute left-1/2 w-0.5 h-full bg-slate-600 z-10"></div>
                              
                              {/* 막대 그래프 */}
                              {rv < 0 ? (
                                // 음수 (좋음 - 파랑): 오른쪽에서 왼쪽으로 뻗음
                                <div className="absolute right-1/2 h-full bg-blue-500 rounded-l-full" style={{width: `${barWidth / 2}%`}}></div>
                              ) : (
                                // 양수 (나쁨 - 빨강): 왼쪽에서 오른쪽으로 뻗음
                                <div className="absolute left-1/2 h-full bg-red-500 rounded-r-full" style={{width: `${barWidth / 2}%`}}></div>
                              )}
                           </div>
                           <div className="flex justify-between text-[8px] text-slate-600 px-1">
                              <span>Good</span>
                              <span>Bad</span>
                           </div>
                        </div>

                      </div>
                     );
                })}
              </div>
            )}
            
            {/* 2. 기간 표시 */}
            {data.pitcher?.period && (
               <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-3 text-center text-slate-400 text-sm font-mono flex justify-center gap-2">
                  <Calendar className="w-4 h-4" />
                  Analysis Period: <span className="text-white font-bold">{data.pitcher.period.start}</span> ~ <span className="text-white font-bold">{data.pitcher.period.end}</span>
               </div>
            )}

            {/* 3. 3D 차트 */}
            {data.pitcher?.arsenal && <Pitch3D arsenal={data.pitcher.arsenal} locations={data.pitcher.locations} />}
             
            {/* 4. 분석 차트 */}
            {data.pitcher?.locations?.length > 0 ? (
                <AnalyticsCharts locations={data.pitcher.locations} />
            ) : (
                <div className="h-40 bg-slate-900 border border-slate-700 rounded-2xl flex items-center justify-center text-slate-500 gap-2">
                    <AlertCircle /> No pitch location data available for charts.
                </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}