import React from 'react';
import { useGameStore } from '@/store/gameStore';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Thermometer, Wind, Mountain } from 'lucide-react';

export default function EnvironmentControl() {
  const { env, setEnv } = useGameStore();

  return (
    <Card className="bg-slate-50 border-slate-200">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-bold uppercase text-slate-500 flex items-center gap-2">
          <Wind size={16} /> Physics Environment
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-medium">
            <span className="flex items-center gap-1"><Thermometer size={12}/> Temp</span>
            <span>{env.temp}°F</span>
          </div>
          <Slider min={30} max={100} step={1} value={[env.temp]} onValueChange={(v) => setEnv('temp', v[0])} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-medium">
            <span className="flex items-center gap-1"><Mountain size={12}/> Altitude</span>
            <span>{env.altitude} ft</span>
          </div>
          <Slider min={0} max={5280} step={100} value={[env.altitude]} onValueChange={(v) => setEnv('altitude', v[0])} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-medium">
            <span className="flex items-center gap-1"><Wind size={12}/> Humidity</span>
            <span>{env.humidity}%</span>
          </div>
          <Slider min={0} max={100} step={1} value={[env.humidity]} onValueChange={(v) => setEnv('humidity', v[0])} />
        </div>
      </CardContent>
    </Card>
  );
}