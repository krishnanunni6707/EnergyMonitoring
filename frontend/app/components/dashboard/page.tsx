'use client';
import React, { useState, useEffect, useCallback } from 'react';
import {
  Bell, Plug2, AlertCircle, PowerOff, Plus, Zap, Activity,
  Clock, LayoutGrid, BarChart3, Settings, Menu, Moon,
  ExternalLink, ChevronDown, TrendingUp, Sparkles
} from 'lucide-react';

// --- UI COMPONENTS ---

const StatCard = ({ icon: Icon, label, value, unit, colorClass, bgClass }: any) => (
  <div className="bg-white p-5 rounded-3xl shadow-sm border border-slate-100 flex items-center gap-4 flex-1 min-w-[200px]">
    <div className={`p-4 rounded-2xl ${bgClass} ${colorClass}`}>
      <Icon size={24} />
    </div>
    <div>
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold text-slate-800">{value}</span>
        <span className="text-sm font-semibold text-slate-400 uppercase">{unit}</span>
      </div>
    </div>
  </div>
);

const DeviceCard = ({ name, status, voltage, current, power, type = 'online', score }: any) => {
  const isWarning = type === 'warning';

  return (
    <div className={`p-6 rounded-[2.5rem] bg-white border shadow-sm transition-all duration-500 ${
      isWarning ? 'border-red-200 bg-red-50/30' : 'border-slate-50'
    }`}>
      <div className="flex justify-between items-start mb-6">
        <div className="flex items-center gap-4">
          <div className={`p-4 rounded-2xl ${
            isWarning ? 'bg-red-50 text-red-500' : 'bg-green-50 text-green-500'
          }`}>
            {isWarning ? <AlertCircle size={24} /> : <Plug2 size={24} />}
          </div>
          <div>
            <h4 className="font-bold text-slate-800">{name}</h4>
            <div className={`flex items-center gap-1.5 text-[10px] font-bold uppercase ${
              isWarning ? 'text-red-500' : 'text-green-500'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${isWarning ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
              {status}
            </div>
          </div>
        </div>
        {isWarning && (
          <div className="bg-red-500 text-white text-[10px] px-2 py-1 rounded-lg font-bold">
            RISK: {score}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-y-4 mb-6">
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400 font-bold uppercase">Voltage</span>
          <div className="text-sm font-bold text-slate-700">{voltage} V</div>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400 font-bold uppercase">Current</span>
          <div className="text-sm font-bold text-slate-700">{current} A</div>
        </div>
        <div className="col-span-2 pt-2 border-t border-slate-50">
          <span className="text-[10px] text-slate-400 font-bold uppercase">Load</span>
          <div className="text-lg font-black text-blue-600">{power} W</div>
        </div>
      </div>
      
      {isWarning ? (
        <p className="text-[11px] text-red-600 font-bold bg-red-100/50 p-2 rounded-lg">
          ⚠️ LSTM: Abnormal Pattern Detected
        </p>
      ) : (
        <p className="text-[11px] text-green-700 font-bold bg-green-100/60 p-2 rounded-lg">
          ✅ LSTM: Normal Pattern Detected
        </p>
      )}
    </div>
  );
};

// --- MAIN DASHBOARD ---

export default function Dashboard() {
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [isAIAnalyzing, setIsAIAnalyzing] = useState(false);

  // 1. STATE: history stores [voltage, current] pairs for the LSTM
  const [appliances, setAppliances] = useState<Array<{
    id: string;
    name: string;
    voltage: string;
    current: string;
    history: number[][];
    prediction: string;
    score: number;
  }>>([]);

  // 2. LIVE CSV DATA POLLING
  useEffect(() => {
    const fetchSensorData = async () => {
      try {
        const res = await fetch('/api/sensor'); // Calls the Python /sensor-data endpoint
        const json = await res.json();
        
        if (Array.isArray(json.data)) {
          setAppliances(prev => {
            const previousById = new Map(prev.map(app => [app.id, app]));
            const latestSensorRowById = new Map<string, any>();

            for (const sensorRow of json.data) {
              const applianceId = typeof sensorRow?.appliance_id === 'string' ? sensorRow.appliance_id.trim() : '';
              if (!applianceId) continue;
              latestSensorRowById.set(applianceId, sensorRow);
            }

            return Array.from(latestSensorRowById.values())
              .map((sensorRow: any) => {
                const applianceId = sensorRow.appliance_id.trim();

                const previous = previousById.get(applianceId);
                const voltage = Number(sensorRow?.voltage ?? 0);
                const current = Number(sensorRow?.current ?? 0);

                const newPair: number[] = [voltage, current];
                const updatedHistory = [...(previous?.history ?? []), newPair].slice(-10);

                return {
                  id: applianceId,
                  name: applianceId,
                  voltage: voltage.toFixed(1),
                  current: current.toFixed(2),
                  history: updatedHistory,
                  prediction: typeof sensorRow?.status === 'string' ? sensorRow.status : previous?.prediction ?? 'Normal',
                  score: Number.isFinite(Number(sensorRow?.usage_score)) ? Number(sensorRow.usage_score) : previous?.score ?? 0
                };
              });
          });
        }
      } catch (err) {
        console.error("Sensor fetch error:", err);
      }
    };

    const timer = setInterval(fetchSensorData, 2000);
    return () => clearInterval(timer);
  }, []);

  // 3. AI PREDICTION TRIGGER
  const runAIAudit = async () => {
    if (appliances.length === 0 || appliances.some(app => app.history.length < 5)) {
      alert("Gathering more sensor history... please wait.");
      return;
    }

    setIsAIAnalyzing(true);
    try {
      const payload = {
        appliances: appliances.map(app => ({
          appliance_id: app.id,
          sequence: app.history
        }))
      };

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      setAppliances(prev => prev.map(app => {
        const result = data.results?.find((r: any) => r.appliance_id === app.id);
        return result ? { ...app, prediction: result.status, score: result.usage_score } : app;
      }));
    } catch (err) {
      console.error("AI Audit error:", err);
    } finally {
      setIsAIAnalyzing(false);
    }
  };

  const totalLoad = appliances.reduce((sum, app) => sum + (parseFloat(app.voltage) * parseFloat(app.current)), 0) / 1000;

  return (
    <div className="min-h-screen bg-[#F8FAFC] flex font-sans">
      {/* Sidebar */}
      <aside className={`fixed lg:sticky top-0 left-0 h-screen w-64 bg-white border-r p-6 z-50 transition-transform lg:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center gap-3 mb-10">
          <div className="bg-blue-600 p-2 rounded-xl text-white"><Zap size={20} fill="currentColor" /></div>
          <span className="text-xl font-black text-blue-600 tracking-tighter">VoltFlow</span>
        </div>
        <nav className="space-y-1">
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-blue-50 text-blue-600 font-bold"><LayoutGrid size={20} /> Panel</button>
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 font-medium hover:bg-slate-50"><BarChart3 size={20} /> Usage</button>
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 font-medium hover:bg-slate-50"><Settings size={20} /> Config</button>
        </nav>
      </aside>

      <main className="flex-1">
        <header className="p-6 bg-white border-b flex justify-between items-center sticky top-0 z-40">
          <div className="flex items-center gap-4">
            <button className="lg:hidden" onClick={() => setSidebarOpen(true)}><Menu /></button>
            <h1 className="font-black text-slate-800 uppercase tracking-widest text-sm">System Live Feed</h1>
          </div>
          <div className="flex items-center gap-2 bg-green-50 text-green-600 px-3 py-1.5 rounded-full border border-green-100">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-[10px] font-black uppercase">CSV Stream Active</span>
          </div>
        </header>

        <div className="p-6 lg:p-10 space-y-10">
          <div className="flex flex-wrap gap-6">
            <StatCard icon={Zap} label="Total Load" value={totalLoad.toFixed(2)} unit="kW" colorClass="text-blue-500" bgClass="bg-blue-50" />
            <StatCard icon={Activity} label="Status" value={appliances.filter(a => a.prediction === 'Abnormal').length} unit="Alerts" colorClass="text-red-500" bgClass="bg-red-50" />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
            {/* Grid of Devices */}
            <div className="xl:col-span-8 space-y-6">
              <h2 className="text-sm font-black text-slate-400 uppercase tracking-[0.2em]">Monitoring Nodes</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {appliances.map(app => (
                  <DeviceCard 
                    key={app.id}
                    name={app.name}
                    voltage={app.voltage}
                    current={app.current}
                    power={(parseFloat(app.voltage) * parseFloat(app.current)).toFixed(1)}
                    status={app.prediction === 'Abnormal' ? 'Abnormal' : 'Normal'}
                    type={app.prediction === 'Abnormal' ? 'warning' : 'online'}
                    score={app.score}
                  />
                ))}
              </div>
            </div>

            {/* AI Control Sidebar */}
            <div className="xl:col-span-4">
              <div className="bg-slate-900 rounded-[3rem] p-8 text-white shadow-2xl relative overflow-hidden">
                <Sparkles className="absolute -right-6 -top-6 text-blue-500/20 w-32 h-32" />
                <h3 className="text-xl font-bold mb-4">LSTM AI Analysis</h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-8">
                  Click below to scan the last 10 snapshots of your CSV data for anomalies.
                </p>
                <button 
                  onClick={runAIAudit}
                  disabled={isAIAnalyzing}
                  className="w-full py-4 bg-blue-600 hover:bg-blue-500 rounded-2xl font-black transition-all active:scale-95 flex items-center justify-center gap-3 disabled:bg-slate-800"
                >
                  {isAIAnalyzing ? <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" /> : "RUN SYSTEM AUDIT"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}