import React from 'react';
import { useGameStore } from '@/store/gameStore';
import { MonitorPlay, FlaskConical } from 'lucide-react';

export default function ModeToggle() {
  const { mode, setMode } = useGameStore();

  return (
    <div className="flex bg-mlb-navy/50 p-1 rounded-lg border border-white/10">
      <button
        onClick={() => setMode('field')}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-bold transition-all ${mode === 'field' ? 'bg-white text-mlb-navy shadow-sm' : 'text-slate-300 hover:text-white'}`}
      >
        <MonitorPlay size={14} /> Field
      </button>
      <button
        onClick={() => setMode('lab')}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-bold transition-all ${mode === 'lab' ? 'bg-savant-blue text-mlb-navy shadow-sm' : 'text-slate-300 hover:text-white'}`}
      >
        <FlaskConical size={14} /> Lab
      </button>
    </div>
  );
}