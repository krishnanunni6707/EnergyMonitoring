'use client';
import React, { useState } from 'react';
import {
  Bell, Plug2, AlertCircle, PowerOff, Plus, Zap, Activity,
  Clock, LayoutGrid, BarChart3, Settings, Menu, X, Moon,
  ExternalLink, ChevronDown, TrendingUp, Sparkles
} from 'lucide-react';

// --- Sub-Components ---

const SidebarItem = ({ icon: Icon, label, active = false }: any) => (
  <button className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-medium mb-1 ${active ? 'bg-blue-50 text-blue-600' : 'text-slate-400 hover:bg-slate-50'
    }`}>
    <Icon size={20} />
    <span className="text-sm">{label}</span>
  </button>
);

const StatCard = ({ icon: Icon, label, value, unit, colorClass, bgClass }: any) => (
  <div className="bg-white p-5 rounded-3xl shadow-sm border border-slate-100 flex items-center gap-4 flex-1 min-w-[240px]">
    <div className={`p-4 rounded-2xl ${bgClass} ${colorClass}`}>
      <Icon size={24} />
    </div>
    <div>
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold text-slate-800">{value}</span>
        <span className="text-sm font-semibold text-slate-400 uppercase">{unit}</span>
      </div>
    </div>
  </div>
);

const DeviceCard = ({ name, status, voltage, current, power, timer, type = 'online' }: any) => {
  const isWarning = type === 'warning';
  const isOffline = type === 'offline';

  return (
    <div className={`p-6 rounded-[2.5rem] bg-white border shadow-sm transition-all ${isWarning ? 'border-red-100' : 'border-slate-50'
      } ${isOffline ? 'opacity-60' : ''}`}>
      <div className="flex justify-between items-start mb-6">
        <div className="flex items-center gap-4">
          <div className={`p-4 rounded-2xl ${isWarning ? 'bg-red-50 text-red-500' : isOffline ? 'bg-slate-100 text-slate-400' : 'bg-green-50 text-green-500'}`}>
            {isOffline ? <PowerOff size={24} /> : isWarning ? <AlertCircle size={24} /> : <Plug2 size={24} />}
          </div>
          <div>
            <h4 className="font-bold text-slate-800">{name}</h4>
            <div className={`flex items-center gap-1.5 text-[10px] font-bold uppercase ${isWarning ? 'text-red-500' : isOffline ? 'text-slate-400' : 'text-green-500'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${isWarning ? 'bg-red-500' : isOffline ? 'bg-slate-400' : 'bg-green-500'}`} />
              {status}
            </div>
          </div>
        </div>
        {!isOffline && (
          <div className={`w-12 h-6 rounded-full relative p-1 cursor-pointer ${!isWarning ? 'bg-blue-500' : 'bg-slate-200'}`}>
            <div className={`bg-white w-4 h-4 rounded-full transition-transform ${!isWarning ? 'translate-x-6' : 'translate-x-0'}`} />
          </div>
        )}
      </div>

      {!isOffline && (
        <div className="grid grid-cols-2 gap-y-4 mb-6">
          <div className="flex items-center gap-2 text-xs text-slate-500"><Zap size={14} className="text-slate-300" /> {voltage} V</div>
          <div className="flex items-center gap-2 text-xs text-slate-500"><Activity size={14} className="text-slate-300" /> {current} A</div>
          {power && <div className="flex items-center gap-2 text-xs font-bold text-slate-800"><Zap size={14} className="text-slate-300" /> {power} W</div>}
          {timer && <div className="flex items-center gap-2 text-xs font-bold text-slate-800"><Clock size={14} className="text-slate-300" /> {timer}</div>}
        </div>
      )}

      {isWarning && (
        <div className="bg-red-50/50 p-4 rounded-2xl border border-red-50">
          <p className="text-[11px] text-red-500 font-medium leading-relaxed">Over-voltage protection triggered.</p>
        </div>
      )}
      {isOffline && <p className="text-sm font-medium text-slate-400 italic mt-8">Last seen 3 hours ago</p>}
    </div>
  );
};

// --- Main Dashboard ---

export default function Dashboard() {
  const [isSidebarOpen, setSidebarOpen] = useState(false);

  // LSTM Logic States
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);

  const fetchLSTMForecasting = async () => {
    setLoadingPrediction(true);
    try {
      // Mocking the last 4 power readings to send to the LSTM
      const history = [2.1, 2.3, 2.2, 2.4];
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: history }),
      });
      const data = await res.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error("Inference failed", err);
    } finally {
      setLoadingPrediction(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] flex font-sans text-slate-900">

      {/* Mobile Backdrop */}
      {isSidebarOpen && (
        <div className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <aside className={`fixed lg:sticky top-0 left-0 h-screen w-64 bg-white border-r border-slate-100 p-6 z-50 transition-transform lg:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}>
        <div className="flex items-center gap-3 mb-10 px-2">
          <div className="bg-blue-500 p-2 rounded-xl text-white"><Zap size={20} fill="currentColor" /></div>
          <span className="text-xl font-black tracking-tight">VoltFlow</span>
        </div>

        <nav className="flex-1">
          <SidebarItem icon={LayoutGrid} label="Panel" active />
          <SidebarItem icon={BarChart3} label="Usage" />
          <SidebarItem icon={Clock} label="Schedule" />
          <SidebarItem icon={Settings} label="Settings" />
        </nav>

        <div className="mt-auto bg-slate-50 p-4 rounded-2xl">
          <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Support</p>
          <button className="flex items-center justify-between w-full text-sm font-bold text-slate-600">
            Help Center <ExternalLink size={14} />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-x-hidden">
        {/* Top Navbar */}
        <header className="p-6 lg:px-10 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <button className="lg:hidden p-2 text-slate-600" onClick={() => setSidebarOpen(true)}>
              <Menu size={24} />
            </button>
            <div>
              <h1 className="text-2xl font-extrabold text-slate-800">Smart Plug Control Panel</h1>
              <p className="text-xs font-medium text-slate-400">Live monitoring & energy management dashboard</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="p-3 bg-white rounded-full shadow-sm text-slate-500 border border-slate-50"><Moon size={18} /></button>
            <button className="p-3 bg-white rounded-full shadow-sm text-slate-500 border border-slate-50 relative">
              <Bell size={18} />
              <span className="absolute top-3 right-3 w-1.5 h-1.5 bg-red-500 rounded-full border border-white"></span>
            </button>
          </div>
        </header>

        <div className="p-6 lg:p-10 pt-0 grid grid-cols-1 xl:grid-cols-12 gap-8">

          {/* Left Column (Stats + Devices) */}
          <div className="xl:col-span-8">
            <div className="flex flex-wrap gap-4 mb-10">
              <StatCard icon={Zap} label="Total Power" value="2.4" unit="kW" colorClass="text-blue-500" bgClass="bg-blue-50" />
              <StatCard icon={Plug2} label="Active" value="8" unit="Devices" colorClass="text-green-500" bgClass="bg-green-50" />
              <StatCard icon={AlertCircle} label="Abnormal" value="1" unit="Alert" colorClass="text-red-500" bgClass="bg-red-50" />
            </div>

            <div className="flex justify-between items-end mb-6">
              <h2 className="text-xl font-bold">Devices</h2>
              <button className="text-blue-500 text-sm font-bold">View all</button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DeviceCard name="Plug 1" status="Online" voltage="232.9" current="0.45" power="104.8" timer="01:24:00" />
              <DeviceCard name="Plug 2" status="Warning" voltage="245.1" current="0.00" type="warning" />
              <DeviceCard name="Plug 3" status="Offline" type="offline" />
              <button className="min-h-[220px] border-2 border-dashed border-slate-200 rounded-[2.5rem] flex flex-col items-center justify-center text-slate-400 gap-3 hover:bg-slate-100/50 transition-colors">
                <div className="p-3 bg-white rounded-full shadow-sm border border-slate-100"><Plus size={24} /></div>
                <span className="font-bold text-sm">Add Device</span>
              </button>
            </div>
          </div>

          {/* Right Column (Controls & AI) */}
          <div className="xl:col-span-4 space-y-8">

            {/* LSTM Prediction Card */}
            <div className="bg-gradient-to-br from-indigo-600 to-blue-700 p-8 rounded-[2.5rem] text-white shadow-xl shadow-blue-100 overflow-hidden relative">
              <Sparkles className="absolute -top-2 -right-2 text-white/10 w-24 h-24" />
              <div className="relative z-10">
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h3 className="font-bold text-white/90">AI Energy Forecast</h3>
                    <p className="text-[10px] text-white/60 uppercase font-black tracking-widest">LSTM Prediction Model</p>
                  </div>
                  <TrendingUp size={24} className="text-white/40" />
                </div>

                <div className="mb-8">
                  <p className="text-xs text-white/70 mb-1">Expected usage for next hour:</p>
                  <div className="text-4xl font-black">
                    {loadingPrediction ? "..." : prediction ? `${prediction} kW` : "---"}
                  </div>
                </div>

                <button
                  onClick={fetchLSTMForecasting}
                  disabled={loadingPrediction}
                  className="w-full bg-white/10 hover:bg-white/20 backdrop-blur-md border border-white/20 text-white text-xs font-bold py-4 rounded-2xl transition-all active:scale-95 flex items-center justify-center gap-2"
                >
                  {loadingPrediction ? (
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <>Run Inference</>
                  )}
                </button>
              </div>
            </div>

            {/* Timer Card */}
            <div className="bg-blue-50/40 p-8 rounded-[2.5rem] border border-blue-100">
              <h3 className="text-center font-bold text-slate-700 mb-8">Set Quick Timer</h3>
              <div className="flex justify-center items-center gap-3 mb-10">
                {[{ v: 0, l: 'HRS' }, { v: 3, l: 'MIN' }, { v: 0, l: 'SEC' }].map((unit, i) => (
                  <React.Fragment key={unit.l}>
                    <div className="bg-white flex flex-col items-center justify-center w-20 h-20 rounded-2xl shadow-sm border border-blue-100">
                      <div className="flex items-center gap-1 font-bold text-xl">
                        {unit.v} <ChevronDown size={14} className="text-slate-300" />
                      </div>
                      <span className="text-[8px] font-black text-slate-400 tracking-tighter uppercase">{unit.l}</span>
                    </div>
                    {i < 2 && <span className="font-bold text-slate-300">:</span>}
                  </React.Fragment>
                ))}
              </div>
              <button className="w-full bg-blue-500 text-white font-bold py-5 rounded-[1.25rem] shadow-xl shadow-blue-200 hover:bg-blue-600 active:scale-95 transition-all">
                Start Countdown
              </button>
            </div>

            {/* Energy Target Card */}
            <div className="bg-white p-8 rounded-[2.5rem] border border-slate-100 shadow-sm">
              <div className="flex justify-between items-center mb-6">
                <h3 className="font-bold">Energy Target</h3>
                <span className="bg-green-100 text-green-600 text-[10px] font-black px-2 py-1 rounded-md uppercase tracking-tight">On Track</span>
              </div>
              <div className="h-3 w-full bg-slate-100 rounded-full mb-4 overflow-hidden">
                <div className="h-full bg-blue-500 w-[65%]" />
              </div>
              <div className="flex justify-between items-end">
                <div>
                  <span className="text-sm font-bold text-slate-700">6.5 kWh</span>
                  <span className="text-[10px] text-slate-400 font-bold ml-1 uppercase">Used</span>
                </div>
                <div className="text-right">
                  <span className="text-sm font-bold text-slate-700">10 kWh</span>
                  <span className="text-[10px] text-slate-400 font-bold ml-1 uppercase">Limit</span>
                </div>
              </div>
            </div>

          </div>
        </div>
      </main>
    </div>
  );
}