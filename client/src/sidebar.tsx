import React, { useEffect, useRef, useState } from 'react';
import { ClipboardList, User, Monitor, PlusCircle, BrainCircuit, Users, HelpCircle } from 'lucide-react';

const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

export type SidebarAction =
  | 'dashboard'
  | 'my-exams'
  | 'results'
  | 'profile'
  | 'help'
  | 'overview'
  | 'create-exam'
  | 'live-proctoring'
  | 'lecturer-live-exams';

export default function Sidebar({ user, onAction, onLogout }: { user: any; onAction: (action: SidebarAction) => void; onLogout?: () => void }) {
  // Live alerts for lecturers
  const [alerts, setAlerts] = useState<any[]>([]);
  const lastTsRef = useRef<string | null>(null);
  const RAW_API_URL = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:5000/api';
  const API_URL = (() => {
    const trimmed = String(RAW_API_URL).trim().replace(/\/+$/, '');
    return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
  })();
  useEffect(() => {
    if (user?.role !== 'lecturer') return;
    let timer: any;
    const tick = async () => {
      try {
        const qs = lastTsRef.current ? `?since=${encodeURIComponent(lastTsRef.current)}&limit=50` : '?limit=50';
        const res = await fetch(`${API_URL}/proctoring/recent-global${qs}`, { headers: { 'X-User-Id': user._id } });
        const data = await res.json();
        if (res.ok && data.events) {
          // newest-first provided by API; prepend and update lastTs
          const events = data.events as any[];
          if (events.length > 0) {
            lastTsRef.current = events[0].timestamp;
            setAlerts(prev => {
              // merge unique by _id
              const byId: Record<string, any> = {};
              [...events, ...prev].forEach(e => { byId[e._id] = e; });
              // Keep newest-first and clip to 30
              return Object.values(byId).sort((a: any, b: any) => (b.timestamp || '').localeCompare(a.timestamp || '')).slice(0, 30);
            });
          }
        }
      } catch {}
    };
    tick();
    timer = setInterval(tick, 30000); // Poll every 30 seconds (reasonable for live alerts)
    return () => clearInterval(timer);
  }, [user]);
  const studentItems = [
    { icon: <ClipboardList className="h-5 w-5" />, label: 'Dashboard', action: 'dashboard' as SidebarAction },
    { icon: <ClipboardList className="h-5 w-5" />, label: 'My Exams', action: 'my-exams' as SidebarAction },
    { icon: <BrainCircuit className="h-5 w-5" />, label: 'Results', action: 'results' as SidebarAction },
    { icon: <User className="h-5 w-5" />, label: 'Profile', action: 'profile' as SidebarAction },
    { icon: <HelpCircle className="h-5 w-5" />, label: 'Help', action: 'help' as SidebarAction },
  ];

  const lecturerItems = [
    { icon: <Monitor className="h-5 w-5" />, label: 'Overview', action: 'overview' as SidebarAction },
    { icon: <PlusCircle className="h-5 w-5" />, label: 'Create Exam', action: 'create-exam' as SidebarAction },
    { icon: <Users className="h-5 w-5" />, label: 'Live Proctoring', action: 'lecturer-live-exams' as SidebarAction },
    { icon: <User className="h-5 w-5" />, label: 'Profile', action: 'profile' as SidebarAction },
    { icon: <HelpCircle className="h-5 w-5" />, label: 'Help', action: 'help' as SidebarAction },
  ];

  const items = user?.role === 'student' ? studentItems : lecturerItems;

  return (
    <div className="w-64 bg-slate-900 p-4 flex flex-col border-r border-slate-800">
      <div className="flex items-center space-x-2 mb-10">
        <BrainCircuit className="h-8 w-8 text-indigo-400" />
        <span className="text-xl font-bold">Invigilo</span>
      </div>

      <nav className="flex-1 space-y-2">
        {items.map(it => (
          <button key={it.label} onClick={() => onAction(it.action)} className={cn(
            "flex items-center space-x-3 w-full text-left px-3 py-2 rounded-lg transition-colors duration-200",
            "text-slate-400 hover:bg-slate-800 hover:text-white"
          )}>
            {it.icon}
            <span>{it.label}</span>
          </button>
        ))}

        {user?.role === 'lecturer' && (
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-white">Live Alerts</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-600 text-white">LIVE</span>
            </div>
            <div className="bg-slate-800/50 border border-slate-700 rounded-md max-h-48 overflow-y-auto">
              {alerts.length === 0 ? (
                <div className="text-xs text-slate-400 p-2">No recent alerts</div>
              ) : (
                alerts.slice(0, 8).map(a => (
                  <div key={a._id} className="p-2 border-b border-slate-700/50 text-xs">
                    <div className="text-slate-200 font-medium">{(a.eventType || 'event').replace(/_/g, ' ')}</div>
                    <div className="text-slate-400">Exam: {a.examId} • Student: {a.userId}</div>
                    <div className="text-slate-500">{new Date(a.timestamp).toLocaleTimeString()}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </nav>

      <div className="mt-auto">
        <div className="flex items-center space-x-3 mb-4 p-2">
          <div className="w-10 h-10 rounded-full bg-indigo-500 flex items-center justify-center font-bold">{(user?.name || 'U').charAt(0)}</div>
          <div>
            <p className="font-semibold text-sm text-white">{user?.name}</p>
            <p className="text-xs text-slate-400">{user?.institution}</p>
          </div>
        </div>
        <div className="px-2 space-y-2">
          <button onClick={() => onAction('profile')} className="w-full inline-flex items-center justify-center rounded-md border border-white/10 px-4 py-2 text-sm text-white hover:bg-white/5">Profile</button>
          {onLogout && (
            <button onClick={onLogout} className="w-full inline-flex items-center justify-center rounded-md border border-white/10 px-4 py-2 text-sm text-white hover:bg-white/5">Logout</button>
          )}
        </div>
      </div>
    </div>
  );
}
