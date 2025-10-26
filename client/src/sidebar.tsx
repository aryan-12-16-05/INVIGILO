import React from 'react';
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
  | 'live-proctoring';

export default function Sidebar({ user, onAction, onLogout }: { user: any; onAction: (action: SidebarAction) => void; onLogout?: () => void }) {
  const studentItems = [
    { icon: <ClipboardList className="h-5 w-5" />, label: 'Dashboard', action: 'dashboard' as SidebarAction },
    { icon: <ClipboardList className="h-5 w-5" />, label: 'My Exams', action: 'my-exams' as SidebarAction },
    { icon: <Users className="h-5 w-5" />, label: 'Live Proctoring', action: 'live-proctoring' as SidebarAction },
    { icon: <User className="h-5 w-5" />, label: 'Profile', action: 'profile' as SidebarAction },
    { icon: <HelpCircle className="h-5 w-5" />, label: 'Help', action: 'help' as SidebarAction },
  ];

  const lecturerItems = [
    { icon: <Monitor className="h-5 w-5" />, label: 'Overview', action: 'overview' as SidebarAction },
    { icon: <PlusCircle className="h-5 w-5" />, label: 'Create Exam', action: 'create-exam' as SidebarAction },
    { icon: <Users className="h-5 w-5" />, label: 'Live Proctoring', action: 'live-proctoring' as SidebarAction },
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
