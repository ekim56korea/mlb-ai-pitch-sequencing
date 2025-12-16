import { create } from 'zustand'

interface GameState {
  mode: 'field' | 'lab';
  pitcherName: string;
  batterStand: 'R' | 'L';
  balls: number;
  strikes: number;
  outs: number;
  env: {
    temp: number;
    altitude: number;
    humidity: number;
  };
  setMode: (mode: 'field' | 'lab') => void;
  setPitcher: (name: string) => void;
  setBatterStand: (stand: 'R' | 'L') => void;
  setBallCount: (type: 'balls' | 'strikes' | 'outs', val: number) => void;
  setEnv: (key: 'temp' | 'altitude' | 'humidity', val: number) => void;
}

export const useGameStore = create<GameState>((set) => ({
  mode: 'lab',
  pitcherName: '',
  batterStand: 'R',
  balls: 0,
  strikes: 0,
  outs: 0,
  env: {
    temp: 70,
    altitude: 0,
    humidity: 50
  },
  setMode: (mode) => set({ mode }),
  setPitcher: (name) => set({ pitcherName: name }),
  setBatterStand: (stand) => set({ batterStand: stand }),
  setBallCount: (type, val) => set((state) => ({ ...state, [type]: val })),
  setEnv: (key, val) => set((state) => ({ env: { ...state.env, [key]: val } }))
}))