import React, { useState, useEffect, type FormEvent, type ChangeEvent, useRef, useCallback } from 'react';
import Sidebar from './sidebar';
import { motion, AnimatePresence } from 'framer-motion';
import {
    User, LogIn, ShieldCheck, Cpu, BrainCircuit,
    Timer, PlusCircle, Monitor, AlertTriangle, CheckCircle, XCircle,
    School, GraduationCap, ChevronLeft, Eye, EyeOff,
    Lock, Users, Wifi, Mic, Video, Globe, Trash2, Unlock, Edit, Save, Play
} from 'lucide-react';

// Import new proctoring components
import LecturerProctoringMonitor from './components/LecturerProctoringMonitor';
import LecturerExamReport from './components/LecturerExamReport';
import LecturerLiveExamsList from './components/LecturerLiveExamsList';

import { io, type Socket } from 'socket.io-client';

// Auth portal illustrations
import studentSidebarImg from '../../student_sidebar.png';
import lecturerSidebarImg from '../../lecturer_sidebar.png';

// Browser lockdown helper (fullscreen + tab/window/devtools restrictions)
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - browserLock.js is a plain CommonJS module
import BrowserLock from './browserLock';


// Note: This assumes you have a 'cn' utility function for class names, e.g., from 'clsx' and 'tailwind-merge'.
// If not, you can replace cn(...) with a simple string of class names.
// import { cn } from './lib/utils';
const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');


// --- API URL ---
// In production set VITE_API_URL to your deployed backend (e.g. https://<service>.onrender.com/api)
const RAW_API_URL = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:5000/api';
const API_URL = (() => {
    // Normalize common misconfigurations:
    // - trailing slash: https://host/ -> https://host
    // - missing /api: https://host -> https://host/api
    const trimmed = String(RAW_API_URL).trim().replace(/\/+$/, '');
    return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
})();

// --- MOCK INSTITUTION DATA ---
const INSTITUTIONS: { [key: string]: string[] } = {
  "Indian Institute of Technology Hyderabad (IIT Hyderabad)": [
    "Artificial Intelligence",
    "Biomedical Engineering",
    "Biotechnology",
    "Chemical Engineering",
    "Civil Engineering",
    "Computer Science and Engineering",
    "Electrical Engineering",
    "Engineering Science",
    "Materials Science and Metallurgical Engineering",
    "Mechanical and Aerospace Engineering",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Design"
  ],
  "International Institute of Information Technology, Hyderabad (IIIT Hyderabad)": [
    "Computer Science and Engineering",
    "Electronics and Communication Engineering",
    "Computational Linguistics",
    "Bioinformatics",
    "Building Science"
  ],
  "Jawaharlal Nehru Technological University Hyderabad (JNTU Hyderabad)": [
    "Civil Engineering",
    "Electrical and Electronics Engineering",
    "Mechanical Engineering",
    "Electronics and Communication Engineering",
    "Computer Science and Engineering",
    "Information Technology",
    "Chemical Engineering",
    "Metallurgical Engineering"
  ],
  "University College of Engineering, Osmania University": [
    "Civil Engineering",
    "Mechanical Engineering",
    "Electrical Engineering",
    "Electronics & Communication Engineering",
    "Bio-Medical Engineering",
    "Computer Science & Engineering",
    "Artificial Intelligence & Machine Learning",
    "Mining Engineering"
  ],
  "VNR Vignana Jyothi Institute of Engineering and Technology (VNR VJIET)": [
    "Artificial Intelligence & Data Science",
    "CSE (Artificial Intelligence & Machine Learning)",
    "CSE (Data Science)",
    "CSE (Cyber Security)",
    "CSE (Internet of Things)",
    "Computer Science and Business Systems",
    "Civil Engineering",
    "Electrical and Electronics Engineering",
    "Mechanical Engineering",
    "Electronics and Communication Engineering",
    "Computer Science and Engineering",
    "Electronics and Instrumentation Engineering",
    "Information Technology",
    "Automobile Engineering"
  ],
  "Mahatma Gandhi Institute of Technology (MGIT)": [
    "Mechanical Engineering (Mechatronics)",
    "Metallurgical and Materials Engineering",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Computer Science and Engineering",
    "Civil Engineering",
    "Mechanical Engineering",
    "Computer Science and Business System",
    "Computer Science and Engineering (Artificial Intelligence and Machine Learning)",
    "Computer Science and Engineering (Data Science)"
  ],
  "Sreenidhi Institute of Science & Technology (SNIST)": [
    "Civil Engineering",
    "Mechanical Engineering",
    "Computer Science and Engineering",
    "Information Technology",
    "Electrical and Electronics Engineering",
    "Electronics and Communication Engineering",
    "Electronics and Computer Engineering",
    "CSE (Cybersecurity)",
    "CSE (Artificial Intelligence & Machine Learning)",
    "CSE (Data Science)",
    "CSE (Internet of Things)"
  ],
  "Chaitanya Bharathi Institute of Technology (CBIT Hyderabad)": [
    "Civil Engineering",
    "Mechanical Engineering",
    "Electronics & Communication Engineering",
    "Computer Science and Engineering",
    "Electrical & Electronics Engineering",
    "Information Technology",
    "Chemical Engineering",
    "Biotechnology",
    "Artificial Intelligence and Data Science",
    "Computer Science and Engineering (Artificial Intelligence and Machine Learning)",
    "Computer Science and Engineering (Internet of Things and Cyber Security including Blockchain Technology)",
    "Electronics Engineering (VLSI Design and Technology)"
  ],
  "Vasavi College of Engineering": [
    "Civil Engineering",
    "Computer Science and Engineering",
    "Computer Science (AI & ML)",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Mechanical Engineering"
  ],
  "Gokaraju Rangaraju Institute of Engineering and Technology (GRIET)": [
    "Civil Engineering",
    "Computer Science and Engineering",
    "CSE (AI & ML)",
    "CSE (Data Science)",
    "Information Technology",
    "Electrical and Electronics Engineering",
    "Electronics and Communication Engineering",
    "Mechanical Engineering"
  ],
  "BVRIT Hyderabad College of Engineering for Women": [
    "Artificial Intelligence and Machine Learning",
    "Computer Science and Engineering",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering"
  ],
  "Institute of Aeronautical Engineering (IARE)": [
    "Aeronautical Engineering",
    "Civil Engineering",
    "Computer Science and Engineering",
    "Electronics and Communication Engineering",
    "Mechanical Engineering",
    "Electrical and Electronics Engineering",
    "Automobile Engineering",
    "Information Technology"
  ],
  "Malla Reddy Engineering College (MREC)": [
    "Civil Engineering",
    "Computer Science and Engineering",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Mechanical Engineering"
  ],
  "Anurag University": [
    "Computer Science and Engineering",
    "Computer Science (Artificial Intelligence)",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Metallurgical Engineering"
  ],
  "Keshav Memorial Institute of Technology (KMIT)": [
    "Computer Science and Engineering",
    "Information Technology",
    "Artificial Intelligence and Machine Learning",
    "Data Science"
  ],
  "Vardhaman College of Engineering": [
    "Computer Science and Engineering",
    "CSE (AI & ML)",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Civil Engineering",
    "Mechanical Engineering"
  ],
  "CVR College of Engineering": [
    "Computer Science and Engineering",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Instrumentation Engineering",
    "Mechanical Engineering",
    "Civil Engineering"
  ],
  "G Narayanamma Institute of Technology and Science (for Women)": [
    "Computer Science and Engineering",
    "Information Technology",
    "Electrical and Electronics Engineering",
    "Electronics and Communication Engineering",
    "Electronics and Telematics Engineering"
  ],
  "J.B. Institute of Engineering & Technology (JBIET)": [
    "Computer Science and Engineering",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Civil Engineering",
    "Mechanical Engineering"
  ],
  "CMR College of Engineering & Technology": [
    "Artificial Intelligence and Machine Learning",
    "Computer Science and Engineering",
    "Information Technology",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Civil Engineering",
    "Mechanical Engineering"
  ]
};


// --- TYPE DEFINITIONS ---
type AppState = 'loading' | 'landing' | 'auth' | 'student-dashboard' | 'lecturer-dashboard' | 'exam' | 'result' | 'results-analysis' | 'live-proctoring' | 'help' | 'my-exams' | 'profile' | 'lecturer-proctor' | 'lecturer-report' | 'lecturer-live-exams';
type UserRole = 'student' | 'lecturer' | 'admin';
type ExamStatus = 'Scheduled' | 'Available' | 'Locked' | 'Completed' | 'Live';
type QuestionType = 'multiple-choice' | 'true-false' | 'short-answer' | 'essay';


interface UserProfile {
    _id: string; // MongoDB uses _id
    email: string;
    name: string;
    role: UserRole;
    phoneNumber: string;
    institution: string;
    department: string;
    studentId?: string;
    lecturerId?: string;
    year?: string;
    faceVerified: boolean;
    isActive: boolean;
    createdAt: string;
}

interface Question {
    _id: string;
    type: QuestionType;
    question: string;
    options?: string[];
    correctAnswer: string | number | boolean;
    marks: number;
}
// For creating questions before they have a DB ID
type NewQuestion = Omit<Question, '_id'>;


interface Exam {
    _id: string;
    title: string;
    description: string;
    courseCode: string;
    duration: number; // in minutes
    questions: Question[];
    scheduledDate: string;
    startTime: string;
    endTime: string;
    institution: string;
    department: string;
    targetYear: string;
    lecturerId: string;
    lecturerName: string;
    createdAt: string;
    status: ExamStatus;
    attempt?: {
        score: number;
        completedAt: string;
    }
}

interface ExamResult {
    score: number;
    totalMarks: number;
    examTitle: string;
    perQuestion?: any[];
}

// --- UI COMPONENTS (ShadCN UI Inspired, Enhanced) ---
const Select = React.forwardRef<HTMLSelectElement, { className?: string; children: React.ReactNode; [key: string]: any }>(({ className, children, ...props }, ref) => {
    return <select className={cn("flex h-10 w-full items-center justify-between rounded-md border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500", className)} ref={ref} {...props}>{children}</select>;
});
interface AnimatedCardProps {
    children: React.ReactNode;
    className?: string;
    delay?: number;
}
const AnimatedCard: React.FC<AnimatedCardProps> = ({ children, className, delay = 0 }) => (
    <motion.div
        className={cn("glass-card p-6 md:p-8", className)}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay }}
        whileHover={{ y: -5, boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.2)", transition: { duration: 0.2 } }}
    >
        {children}
    </motion.div>
);


const Button = ({ children, className, variant = 'default', isLoading = false, ...props }: { children: React.ReactNode; className?: string; variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link'; isLoading?: boolean; [key: string]: any }) => {
    const variants = {
        default: "bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-500/30",
        destructive: "bg-red-600 text-white hover:bg-red-500",
        outline: "border border-white/20 hover:bg-white/10",
        secondary: "bg-white/10 hover:bg-white/20",
        ghost: "hover:bg-white/10",
        link: "text-indigo-400 hover:underline",
    };
    return (
        <button
            className={cn("inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-950 disabled:opacity-50 disabled:pointer-events-none px-4 py-2", variants[variant], className)}
            disabled={isLoading}
            {...props}
        >
            {isLoading ? <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div> : children}
        </button>
    );
};

const Input = React.forwardRef<HTMLInputElement, { className?: string; [key: string]: any }>(({ className, ...props }, ref) => {
    return <input className={cn("flex h-10 w-full rounded-md border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500", className)} ref={ref} {...props} />;
});


const Label = ({ children, className, ...props }: { children: React.ReactNode; className?: string; [key: string]: any }) => {
    return <label className={cn("text-sm font-medium leading-none text-slate-300 peer-disabled:cursor-not-allowed peer-disabled:opacity-70", className)} {...props}>{children}</label>;
};

const Card = ({ children, className, ...props }: { children: React.ReactNode; className?: string; [key: string]: any }) => {
    return <div className={cn("rounded-xl border bg-slate-900/50 border-slate-800", className)} {...props}>{children}</div>;
};

const Dialog = ({ open, onOpenChange, children, className }: { open: boolean; onOpenChange: (open: boolean) => void; children: React.ReactNode; className?: string; }) => {
    return (
        <AnimatePresence>
            {open && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
                    onClick={() => onOpenChange(false)}
                >
                    <motion.div
                        initial={{ scale: 0.95, opacity: 0, y: 20 }}
                        animate={{ scale: 1, opacity: 1, y: 0 }}
                        exit={{ scale: 0.95, opacity: 0, y: 20 }}
                        transition={{ duration: 0.2 }}
                        className={cn("relative z-50 w-full max-w-lg glass-card p-6 rounded-2xl border-slate-700", className)}
                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                    >
                        {children}
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

const Badge = ({ children, className, variant = 'default' }: { children: React.ReactNode; className?: string; variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'live' }) => {
    const variants = {
        default: 'bg-slate-700 text-slate-200',
        success: 'bg-green-500/20 text-green-300 border border-green-500/30',
        warning: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30',
        danger: 'bg-red-500/20 text-red-300 border border-red-500/30',
        info: 'bg-sky-500/20 text-sky-300 border border-sky-500/30',
        live: 'bg-red-500/80 text-white animate-pulse',
    };
    return <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', variants[variant], className)}>{children}</span>
};

const Toaster = ({ toasts }: { toasts: { id: number; message: string; type: 'success' | 'error' }[] }) => (
    <div className="fixed bottom-0 right-0 p-4 z-50 w-full max-w-sm">
        <AnimatePresence>
            {toasts.map(toast => (
                <motion.div
                    key={toast.id}
                    layout
                    initial={{ opacity: 0, y: 50, scale: 0.3 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 20, scale: 0.5, transition: { duration: 0.2 } }}
                    className={cn(
                        "mt-2 p-4 rounded-lg shadow-lg text-white flex items-center space-x-2 text-sm font-medium glass-card border-slate-700",
                        toast.type === 'success' ? 'bg-green-600/30' : 'bg-red-600/30'
                    )}
                >
                    {toast.type === 'success' ? <CheckCircle className="h-5 w-5 text-green-400" /> : <XCircle className="h-5 w-5 text-red-400" />}
                    <span>{toast.message}</span>
                </motion.div>
            ))}
        </AnimatePresence>
    </div>
);


// --- CORE APP ---
export default function App() {
    const [appState, setAppState] = useState<AppState>('loading');
    const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
    const [authRole, setAuthRole] = useState<UserRole>('student');
    const [toasts, setToasts] = useState<{ id: number; message: string; type: 'success' | 'error' }[]>([]);
    const [exams, setExams] = useState<Exam[]>([]);
    const [currentExam, setCurrentExam] = useState<Exam | null>(null);
    const [lastResult, setLastResult] = useState<ExamResult | null>(null);
    const [selectedExamIdForProctoring, setSelectedExamIdForProctoring] = useState<string>('');

    useEffect(() => {
        const timer = setTimeout(() => setAppState('landing'), 1500);
        return () => clearTimeout(timer);
    }, []);

    const showToast = useCallback((message: string, type: 'success' | 'error' = 'success') => {
        setToasts(prev => [...prev, { id: Date.now(), message, type }]);
        setTimeout(() => setToasts(prev => prev.slice(1)), 4000);
    }, []);

    const fetchExams = useCallback(async () => {
        try {
            const url = currentUser ? `${API_URL}/exams?userId=${currentUser._id}` : `${API_URL}/exams`;
            const res = await fetch(url);
            if (!res.ok) throw new Error('Failed to fetch exams');
            const data = await res.json();
            console.log('[EXAMS] Fetched exams:', data.exams.length, 'exams');
            if (currentUser) {
                const withAttempts = data.exams.filter((e: any) => e.attemptForUser);
                console.log('[EXAMS] Exams with attemptForUser:', withAttempts.length);
                withAttempts.forEach((e: any) => {
                    console.log(`[EXAM] ${e.title}: attemptForUser.score=${e.attemptForUser?.score}, completedByUser=${e.completedByUser}`);
                });
            }
            setExams(data.exams);
        } catch (error: any) {
            showToast(error.message, 'error');
        }
    }, [currentUser, showToast]);

    useEffect(() => {
        if (currentUser) {
            fetchExams();
        }
    }, [currentUser, fetchExams]);


    const handleLogout = () => {
        setCurrentUser(null);
        setExams([]);
        setCurrentExam(null);
        navigateTo('landing');
        showToast('Successfully logged out.', 'success');
    };

    const navigateTo = (state: AppState, role?: UserRole) => {
        if (role) {
            setAuthRole(role);
        }
        setAppState('loading');
        setTimeout(() => setAppState(state), 400);
    }

    const onAuthSuccess = (user: UserProfile) => {
        setCurrentUser(user);
        navigateTo(user.role === 'student' ? 'student-dashboard' : 'lecturer-dashboard');
    };
    
    const handleStartExam = (examId: string) => {
        const examToStart = exams.find(e => e._id === examId);
        if (examToStart) {
            // Prevent starting if user already completed the exam or server indicates they cannot start
            if ((examToStart as any).completedByUser) {
                showToast('You have already submitted this exam.', 'error');
                return;
            }
            if (typeof (examToStart as any).canStartForUser !== 'undefined' && !(examToStart as any).canStartForUser) {
                showToast('This exam cannot be started at this time.', 'error');
                return;
            }
            setCurrentExam(examToStart);
            navigateTo('exam');
        } else {
            showToast('Could not find the selected exam.', 'error');
        }
    };
    
    const handleExamSubmit = (result: ExamResult) => {
        setLastResult(result);
        setCurrentExam(null);
        fetchExams(); // Refresh exams to show the completed one
        navigateTo('result');
    };

    const renderContent = () => {
        switch (appState) {
            case 'loading': return <LoadingScreen key="loading" />;
            case 'landing': return <LandingPage key="landing" onNavigate={navigateTo} />;
            case 'auth': return <AuthPage key="auth" initialRole={authRole} onAuthSuccess={onAuthSuccess} showToast={showToast} onBack={() => navigateTo('landing')} />;
            case 'student-dashboard': return currentUser && <StudentDashboard key="student-dashboard" user={currentUser} exams={exams} onLogout={handleLogout} onStartExam={handleStartExam} onBack={() => navigateTo('landing')} showToast={showToast} onUpdateUser={setCurrentUser} navigateTo={navigateTo} />;
            case 'my-exams': return currentUser && <MyExamsPage key="my-exams" user={currentUser} exams={exams} onLogout={handleLogout} onStartExam={handleStartExam} showToast={showToast} onUpdateUser={setCurrentUser} navigateTo={navigateTo} />;
            case 'results-analysis': return currentUser && <ResultsAnalysisPage key="results-analysis" user={currentUser} exams={exams} onLogout={handleLogout} onBack={() => navigateTo('student-dashboard')} showToast={showToast} onUpdateUser={setCurrentUser} navigateTo={navigateTo} />;
                case 'lecturer-dashboard': return currentUser && <LecturerDashboard key="lecturer-dashboard" user={currentUser} exams={exams} onLogout={handleLogout} onBack={() => navigateTo('landing')} onExamChange={fetchExams} showToast={showToast} onUpdateUser={setCurrentUser} navigateTo={navigateTo} setSelectedExamIdForProctoring={setSelectedExamIdForProctoring} />;
            case 'lecturer-live-exams': return currentUser && <LecturerLiveExamsList key="lecturer-live-exams" lecturerId={currentUser._id} onBack={() => navigateTo('lecturer-dashboard')} onSelectExam={(examId) => { setSelectedExamIdForProctoring(examId); navigateTo('lecturer-proctor'); }} />;
            case 'lecturer-proctor': return currentUser && selectedExamIdForProctoring && <LecturerProctoringMonitor key="lecturer-proctor" examId={selectedExamIdForProctoring} onBack={() => navigateTo('lecturer-live-exams')} showToast={showToast} />;
            case 'lecturer-report': return currentUser && selectedExamIdForProctoring && <LecturerExamReport key="lecturer-report" examId={selectedExamIdForProctoring} onBack={() => navigateTo('lecturer-dashboard')} showToast={showToast} />;
            case 'live-proctoring': return currentUser && <LiveProctoring key="live-proctoring" user={currentUser} onBack={() => navigateTo(currentUser?.role === 'student' ? 'student-dashboard' : 'lecturer-dashboard')} />;
            case 'help': return <HelpPage key="help" onBack={() => navigateTo(currentUser?.role === 'student' ? 'student-dashboard' : 'lecturer-dashboard')} />;
            case 'profile': return currentUser && <ProfilePage key="profile" user={currentUser} onLogout={handleLogout} onBack={() => navigateTo(currentUser?.role === 'student' ? 'student-dashboard' : 'lecturer-dashboard')} showToast={showToast} onUpdateUser={setCurrentUser} navigateTo={navigateTo} />;
            case 'exam': return currentUser && currentExam && <ExamScreen key="exam" exam={currentExam} user={currentUser} onExit={handleExamSubmit} showToast={showToast} />;
            case 'result': return lastResult && <ResultScreen key="result" result={lastResult} onDone={() => navigateTo('student-dashboard')} />;
            default: return <LandingPage key="default-landing" onNavigate={navigateTo} />;
        }
    };

    return (
        <div className="dark bg-slate-950 text-white min-h-screen font-sans">
            <AnimatePresence mode="wait">
                {renderContent()}
            </AnimatePresence>
            <Toaster toasts={toasts} />
        </div>
    );
}

// --- PAGES & LAYOUTS ---
const LoadingScreen = () => (
    <motion.div
        className="fixed inset-0 flex flex-col items-center justify-center bg-slate-950 z-50"
        exit={{ opacity: 0 }}
        transition={{ duration: 0.3 }}
    >
        <div className="relative flex items-center justify-center">
            <div className="absolute h-24 w-24 rounded-full border-t-2 border-b-2 border-indigo-500 animate-spin"></div>
            <BrainCircuit className="h-12 w-12 text-indigo-400" />
        </div>
        <motion.p
            className="mt-4 text-lg text-slate-300 tracking-widest"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
        >
            INVIGILO
        </motion.p>
    </motion.div>
);

const LandingPage = ({ onNavigate }: { onNavigate: (state: AppState, role: UserRole) => void }) => {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="min-h-screen flex flex-col items-center justify-center p-4 overflow-hidden relative"
        >
            <div className="absolute inset-0 z-0 bg-slate-950">
                <div className="absolute bottom-0 left-0 right-0 top-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]"></div>
            </div>
            <motion.div
                className="absolute -top-1/4 -left-1/4 h-1/2 w-1/2 rounded-full bg-indigo-500/20 blur-3xl"
                animate={{ x: [0, 100, 0, -50, 0], y: [0, 50, 100, 50, 0], scale: [1, 1.2, 1, 1.1, 1], rotate: [0, 0, 180, 180, 0] }}
                transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.div
                className="absolute -bottom-1/4 -right-1/4 h-1/2 w-1/2 rounded-full bg-purple-500/20 blur-3xl"
                animate={{ x: [0, -100, 0, 50, 0], y: [0, -50, -100, -50, 0], scale: [1, 1.1, 1.2, 1, 1], rotate: [0, 90, 180, 270, 360] }}
                transition={{ duration: 25, repeat: Infinity, ease: "easeInOut" }}
            />

            <main className="text-center z-10">
                <motion.h1
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="text-5xl md:text-7xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-400"
                >
                    Invigilo
                </motion.h1>
                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className="max-w-2xl mx-auto text-lg text-slate-300 mb-12"
                >
                    Next-generation AI proctoring for secure, fair, and intelligent online examinations.
                </motion.p>
                <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
                    <AnimatedCard delay={0.4}>
                        <div className="flex flex-col items-center text-center">
                            <div className="p-4 bg-indigo-500/10 rounded-full mb-4">
                                <GraduationCap className="h-12 w-12 text-indigo-300" />
                            </div>
                            <h3 className="text-2xl font-bold mb-2">Student Portal</h3>
                            <p className="text-slate-400 mb-6">Access your exams, review performance, and ensure system readiness in a secure environment.</p>
                            <Button className="w-full" onClick={() => onNavigate('auth', 'student')}>
                                Enter as Student <LogIn className="ml-2 h-4 w-4" />
                            </Button>
                        </div>
                    </AnimatedCard>
                    <AnimatedCard delay={0.6}>
                        <div className="flex flex-col items-center text-center">
                            <div className="p-4 bg-purple-500/10 rounded-full mb-4">
                                <School className="h-12 w-12 text-purple-300" />
                            </div>
                            <h3 className="text-2xl font-bold mb-2">Lecturer Dashboard</h3>
                            <p className="text-slate-400 mb-6">Create exams, generate questions with AI, monitor students, and analyze results.</p>
                            <Button className="w-full bg-purple-600 hover:bg-purple-500 shadow-purple-500/30" onClick={() => onNavigate('auth', 'lecturer')}>
                                Enter as Lecturer <User className="ml-2 h-4 w-4" />
                            </Button>
                        </div>
                    </AnimatedCard>
                    <AnimatedCard delay={0.8}>
                        <div className="flex flex-col items-center text-center">
                            <div className="p-4 bg-sky-500/10 rounded-full mb-4">
                                <ShieldCheck className="h-12 w-12 text-sky-300" />
                            </div>
                            <h3 className="text-2xl font-bold mb-2">Admin Panel</h3>
                            <p className="text-slate-400 mb-6">Manage system settings, view statistics, and oversee all platform activities.</p>
                            <Button className="w-full bg-sky-600 hover:bg-sky-500 shadow-sky-500/30" onClick={() => onNavigate('auth', 'admin')}>
                                Enter as Admin <ShieldCheck className="ml-2 h-4 w-4" />
                            </Button>
                        </div>
                    </AnimatedCard>
                </div>
            </main>
        </motion.div>
    );
};


const AuthPage = ({
  initialRole,
  onAuthSuccess,
  showToast,
  onBack,
}: {
  initialRole: UserRole;
  onAuthSuccess: (user: UserProfile) => void;
  showToast: (message: string, type: "success" | "error") => void;
  onBack: () => void;
}) => {
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signin");
    const [currentStep, setCurrentStep] = useState<"details" | "face">("details");
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
        const [signupPassword, setSignupPassword] = useState("");
        const [signupConfirmPassword, setSignupConfirmPassword] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null); // ✅ Persist webcam stream
  const [captureMessage, setCaptureMessage] = useState<string>("");
  const [institution, setInstitution] = useState("");
  const [department, setDepartment] = useState("");
  const formDataRef = useRef<any>({});
    const [enrollSamples, setEnrollSamples] = useState<string[]>([]);

    // --- Basic field validation (client-side UX only; server must still validate) ---
    const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;
    const IN_PHONE_REGEX = /^[6-9]\d{9}$/; // Indian mobile numbers: 10 digits starting with 6-9
    const validateEmail = (email: string) => EMAIL_REGEX.test(email.trim());
    const validateIndianPhone = (phone: string) => IN_PHONE_REGEX.test(phone.replace(/\s+/g, "").trim());

    // Password policy should match backend validate_password() in server/app.py
    const PASSWORD_RULES = {
        minLength: (v: string) => v.length >= 8,
        hasUpper: (v: string) => /[A-Z]/.test(v),
        hasLower: (v: string) => /[a-z]/.test(v),
        hasDigit: (v: string) => /[0-9]/.test(v),
        hasSymbol: (v: string) => /[!@#$%^&*\-_=+]/.test(v),
    };

    const getPasswordMissing = (v: string) => {
        const missing: string[] = [];
        if (!PASSWORD_RULES.minLength(v)) missing.push('at least 8 characters');
        if (!PASSWORD_RULES.hasUpper(v)) missing.push('one uppercase letter (A-Z)');
        if (!PASSWORD_RULES.hasLower(v)) missing.push('one lowercase letter (a-z)');
        if (!PASSWORD_RULES.hasDigit(v)) missing.push('one number (0-9)');
        if (!PASSWORD_RULES.hasSymbol(v)) missing.push('one special character (!@#$%^&*-_=+)');
        return missing;
    };

    const isPasswordValid = (v: string) => getPasswordMissing(v).length === 0;

    const handleForgotPassword = () => {
        showToast("Forgot password isn't implemented yet.", "error");
    };

  // --- CAMERA INIT & CLEANUP ---
    useEffect(() => {
        if (currentStep === "face" || authMode === "signin") {
            const videoElement = videoRef.current;
            if (!videoElement) return;

            navigator.mediaDevices.getUserMedia({ video: true })
                .then((stream) => {
                    // Store the stream so other code can access/stop it
                    streamRef.current = stream;

                    // Make sure the video is muted (autoplay policy) and plays inline
                    try {
                        videoElement.muted = true;
                        (videoElement as any).playsInline = true;
                    } catch (e) {
                        // ignore if properties not writable
                    }

                    videoElement.srcObject = stream;

                    // Required for some browsers; handle the play promise
                    videoElement.onloadedmetadata = () => {
                        const p = videoElement.play();
                        if (p && p instanceof Promise) p.catch(err => console.warn('autoplay prevented:', err));
                    };
                })
                .catch((err) => {
                    console.error("Error accessing webcam: ", err);
                    showToast("Could not access webcam. Please allow camera permissions.", "error");
                });

            return () => {
                // Prefer stopping the stored stream if available
                const stream = streamRef.current ?? (videoElement.srcObject as MediaStream | null);
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                if (videoElement) videoElement.srcObject = null;
                streamRef.current = null;
            };
        }
        // only re-run when these change
    }, [currentStep, authMode, showToast]);

  const captureFrame = (): string | null => {
        const video = videoRef.current;
        if (!video) return null;

        // Prefer real video dimensions, but fall back to bounding rect if not yet available.
        let w = video.videoWidth;
        let h = video.videoHeight;
        if (w === 0 || h === 0) {
            const rect = video.getBoundingClientRect();
            w = Math.max(1, Math.floor(rect.width));
            h = Math.max(1, Math.floor(rect.height));
        }

        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;

        try {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL("image/jpeg");
        } catch (err) {
            console.warn('captureFrame drawImage failed:', err);
            return null;
        }
  };

  const stopCamera = () => {
        const video = videoRef.current;
        const stream = streamRef.current ?? (video?.srcObject as MediaStream | null);

        if (stream) {
            stream.getTracks().forEach((t) => {
                try {
                    t.stop();
                } catch {
                    // no-op
                }
            });
        }

        if (video) video.srcObject = null;
        streamRef.current = null;
  };

  const startCamera = async () => {
        const video = videoRef.current;
        if (!video) return;

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = stream;

        try {
            video.muted = true;
            (video as any).playsInline = true;
        } catch {
            // ignore
        }

        video.srcObject = stream;
        const p = video.play();
        if (p && p instanceof Promise) p.catch(() => undefined);
  };

  const restartCamera = async () => {
        // Ensure any previous stream is fully torn down before re-acquiring
        stopCamera();
        // Small delay helps some browsers release the device immediately
        await new Promise((r) => setTimeout(r, 150));
        await startCamera();
  };

// Wait until video has non-zero dimensions; default timeout kept short for snappy UX
const waitForVideoReady = (video: HTMLVideoElement, timeout = 5000): Promise<void> => {
    return new Promise((resolve, reject) => {
        const start = Date.now();

        const onReady = () => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                cleanup();
                resolve();
            }
        };

    

        const interval = window.setInterval(() => {
            if (video.readyState >= 3 && video.videoWidth > 0 && video.videoHeight > 0) {
                cleanup();
                resolve();
            } else if (Date.now() - start > timeout) {
                cleanup();
                reject(new Error('Video not ready in time. Please check your camera and allow permissions.'));
            }
        }, 250);

        function cleanup() {
            clearInterval(interval);
            video.removeEventListener('loadedmetadata', onReady);
            video.removeEventListener('playing', onReady);
        }

        // Listen to key events that mean the video is usable
        video.addEventListener('loadedmetadata', onReady);
        video.addEventListener('playing', onReady);

        // Also kick it once in case it's already ready
        onReady();
    });
};

// Helper for small on-screen status used in the UI to aid debugging
const getVideoStatus = () => {
    const v = videoRef.current;
    if (!v) return 'no element';
    return `readyState=${v.readyState} ${v.videoWidth}x${v.videoHeight}`;
};

  const handleProceedToFaceStep = (e: FormEvent) => {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const formData = new FormData(form);
    const data: { [key: string]: any } = {};
    formData.forEach((value, key) => {
      data[key] = value;
    });

        // Client-side validations before moving to face step
        const email = String(data.email ?? "").trim();
        const phone = String(data.phoneNumber ?? "").trim();
        const pass = String(data.password ?? "");
        const confirm = String(data.confirmPassword ?? "");

        if (!validateEmail(email)) {
            showToast("Invalid email. Please use a valid format like name@example.com.", "error");
            return;
        }

        if (!validateIndianPhone(phone)) {
            showToast("Invalid phone number. Enter a 10-digit Indian mobile number starting with 6-9.", "error");
            return;
        }

        if (pass !== confirm) {
            showToast("Passwords do not match. Please re-enter the same password.", "error");
            return;
        }

        const missing = getPasswordMissing(pass);
        if (missing.length > 0) {
            showToast(`Password is invalid. Add: ${missing.join(', ')}.`, 'error');
            return;
        }

    formDataRef.current = data;
    setCurrentStep("face");
  };

    const handleFullSignUp = async () => {
    setIsLoading(true);
    setCaptureMessage("");

    const video = videoRef.current;
    if (!video) {
      showToast("Webcam not initialized", "error");
      setIsLoading(false);
      return;
    }

    try {
                setCaptureMessage('Preparing camera...');
                try {
                    await waitForVideoReady(video, 1000);
                } catch (err) {
                    // Try to re-acquire the camera once
                    console.warn('Initial video ready check failed, retrying getUserMedia...', err);
                    setCaptureMessage('Retrying camera... please allow permissions if prompted');
                    try {
                        const newStream = await navigator.mediaDevices.getUserMedia({ video: true });
                        streamRef.current = newStream;
                        video.srcObject = newStream;
                        await waitForVideoReady(video, 8000);
                    } catch (err2) {
                        // If still failing, bubble original error
                        throw err2;
                    }
                }

      // ✅ Countdown before capture
      for (let i = 3; i > 0; i--) {
        setCaptureMessage(i.toString());
        await new Promise((res) => setTimeout(res, 1000));
      }

            setCaptureMessage("Capturing...");
      const imageDataUrl = captureFrame();
      if (!imageDataUrl) {
        throw new Error("Could not capture image. Please ensure your camera is ready.");
      }

            // Allow multi-sample: include existing samples + this capture (unique by string)
            const samples = Array.from(new Set([...enrollSamples, imageDataUrl]));
            const finalData: any = { ...formDataRef.current, role: initialRole };
            // Never send confirmPassword to backend
            delete finalData.confirmPassword;
            if (samples.length > 1) finalData.imageDataUrls = samples;
            else finalData.imageDataUrl = imageDataUrl;

            // Client-side validation to avoid server 400s and improve debugging
            const requiredFields = ['fullName', 'email', 'phoneNumber', 'roleId', 'password', 'role', 'institution', 'department'];
            const missing = requiredFields.filter(f => !finalData[f] || finalData[f] === "");
            // Validate that at least one image field exists
            const hasImage = !!finalData.imageDataUrl || (Array.isArray(finalData.imageDataUrls) && finalData.imageDataUrls.length > 0);
            console.log('Register payload keys:', Object.keys(finalData));
            console.log('imageDataUrl length:', imageDataUrl ? imageDataUrl.length : 0);
            if (missing.length > 0 || !hasImage) {
                console.error('Missing registration fields:', missing);
                const imgMsg = hasImage ? '' : (missing.length > 0 ? '; image missing' : 'Image missing');
                showToast(`Missing required fields: ${missing.join(', ')}${imgMsg}`,'error');
                setIsLoading(false);
                setCaptureMessage('');
                return;
            }

            console.log('[REGISTER] Sending registration request to:', `${API_URL}/register`);
            console.log('[REGISTER] Payload has imageDataUrl:', !!finalData.imageDataUrl, 'length:', finalData.imageDataUrl?.length);
            
            const res = await fetch(`${API_URL}/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(finalData),
            });
            
            console.log('[REGISTER] Response status:', res.status);
            const data = await res.json();
            console.log('[REGISTER] Response data:', data);
            
            if (!res.ok) {
                        // Handle user already exists (conflict) gracefully
                        if (res.status === 409) {
                            showToast(data.error || data.message || "User already exists. Please sign in.", "error");
                            setAuthMode("signin");
                            setCurrentStep("details");
                            setIsLoading(false);
                            setCaptureMessage("");
                            return;
                        }
                        throw new Error(data.error || data.message || "Registration failed");
                    }

                    showToast(data.message, "success");
                    // Attempt to log the user in automatically after successful registration
                    try {
                        const loginRes = await fetch(`${API_URL}/login`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ identifier: finalData.email || finalData.roleId, password: finalData.password, role: finalData.role })
                        });
                        const loginData = await loginRes.json();
                        if (loginRes.ok) {
                            showToast(`Welcome, ${loginData.user.name}!`, 'success');
                            onAuthSuccess(loginData.user);
                        } else {
                            // fallback to signin screen
                            setAuthMode('signin');
                            setCurrentStep('details');
                        }
                    } catch (err) {
                        setAuthMode('signin');
                        setCurrentStep('details');
                    }
    } catch (error: any) {
      showToast(error.message, "error");
    } finally {
      setIsLoading(false);
      setCaptureMessage("");
    }
  };

  const handleSignIn = async (e: FormEvent) => {
    e.preventDefault();
        setIsLoading(true);
        setCaptureMessage("Verifying...");
    const form = e.target as HTMLFormElement;
    const formData = new FormData(form);
    const currentIdentifier = formData.get("identifier") as string;
    const currentPassword = formData.get("password") as string;
        // Attempt to ensure video is ready before capturing
        const video = videoRef.current;
        if (!video) {
            showToast('Webcam not initialized', 'error');
            setIsLoading(false);
            return;
        }

        try {
            await waitForVideoReady(video, 8000);
        } catch (err) {
            // Let user retry explicitly
            showToast('Video not ready. Click Verify to try again.', 'error');
            setIsLoading(false);
            setCaptureMessage('');
            return;
        }

        const imageDataUrl = captureFrame();
        if (!imageDataUrl) {
            console.error('[FACE] Could not capture image - captureFrame returned null');
            showToast("Could not capture image for verification.", "error");
            setIsLoading(false);
            setCaptureMessage("");
            return;
        }
        
        console.log('[FACE] Image captured successfully, length:', imageDataUrl.length);
        console.log('[FACE] Image data URL prefix:', imageDataUrl.substring(0, 50));

        try {
            console.log('[FACE] Sending verification request to:', `${API_URL}/verify-face`);
            const faceRes = await fetch(`${API_URL}/verify-face`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ identifier: currentIdentifier, role: initialRole, imageDataUrl }),
            });
            console.log('[FACE] Verification response status:', faceRes.status);
            const faceData = await faceRes.json();
            console.log('[FACE] Verification response data:', faceData);
            if (!faceRes.ok) {
                const sim = faceData?.similarity ? ` (similarity: ${Number(faceData.similarity).toFixed(3)})` : '';
                showToast(faceData.error || faceData.message || `Face verification failed${sim}`, 'error');
                // Do NOT auto-retry — user must click Verify again
                setIsLoading(false);
                setCaptureMessage('');
                return;
            }

            showToast("Face verified. Logging in...", "success");

            const loginRes = await fetch(`${API_URL}/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ identifier: currentIdentifier, password: currentPassword, role: initialRole }),
            });
            const loginData = await loginRes.json();
            if (!loginRes.ok) {
                showToast(loginData.error || loginData.message || 'Login failed', 'error');
                setIsLoading(false);
                setCaptureMessage('');
                return;
            }

            showToast(`Welcome back, ${loginData.user.name}!`, "success");
            onAuthSuccess(loginData.user);
        } catch (error: any) {
            showToast(error.message, "error");
        } finally {
            setIsLoading(false);
            setCaptureMessage("");
        }
  };

    // Optional: explicit retry handler if you want a separate control
    const handleVerifyRetry = async () => {
        // User explicitly wants the camera to restart on retry.
        setIsLoading(true);
        setCaptureMessage('Restarting camera...');

        try {
            await restartCamera();
            setCaptureMessage('');
            showToast('Camera restarted. Please verify again.', 'success');
        } catch (err: any) {
            console.error('restartCamera failed:', err);
            setCaptureMessage('');
            showToast(err?.message || 'Could not restart camera. Please check permissions.', 'error');
        } finally {
            setIsLoading(false);
        }
    };

  

  // ...rest of your JSX code remains untouched



  const idLabel = initialRole === "student" ? "Student ID" : "Lecturer ID";

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="min-h-screen flex items-center justify-center p-4 bg-slate-950">
      <Button variant="outline" className="absolute top-6 right-6" onClick={onBack}>
        <ChevronLeft className="h-4 w-4 mr-2" /> Back to Home
      </Button>
      <div className="w-full max-w-4xl grid md:grid-cols-2 gap-8 items-center">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
          <div className="p-8 bg-slate-800 rounded-xl shadow-lg">
            <img
                            src={initialRole === 'student' ? studentSidebarImg : lecturerSidebarImg}
                            alt={initialRole === 'student' ? 'Student portal' : 'Lecturer portal'}
                            className="rounded-lg w-full h-64 md:h-72 object-cover border border-slate-700 mb-6"
            />
            <h2 className="text-3xl font-bold mb-2 capitalize">{initialRole} Portal</h2>
            <p className="text-slate-400">
              Access your secure exam environment with AI-powered proctoring and real-time monitoring.
            </p>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
          <div className="flex justify-center mb-6">
            <div className="bg-slate-800 p-1 rounded-lg flex space-x-1">
              <Button
                variant={authMode === "signin" ? "default" : "ghost"}
                onClick={() => {
                  setAuthMode("signin");
                  setCurrentStep("details");
                }}
                className="w-28"
              >
                Sign In
              </Button>
              <Button
                variant={authMode === "signup" ? "default" : "ghost"}
                onClick={() => {
                  setAuthMode("signup");
                  setCurrentStep("details");
                }}
                className="w-28"
              >
                Sign Up
              </Button>
            </div>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={`${authMode}-${currentStep}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {/* --- Sign In Form --- */}
              {authMode === "signin" && (
                <form onSubmit={handleSignIn} className="space-y-4">
                  <h3 className="text-2xl font-bold text-center text-slate-100">Welcome Back</h3>

                  <div className="space-y-1">
                    <Label htmlFor="identifier">Email, Phone, or ID</Label>
                    <Input id="identifier" name="identifier" type="text" placeholder="Enter your identifier" required />
                  </div>
                  <div className="space-y-1 relative">
                    <Label htmlFor="password-signin">Password</Label>
                    <Input
                      id="password-signin"
                      name="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your Password"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-8 text-slate-400 hover:text-slate-200"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>

                                    <div className="flex justify-end">
                                        <button type="button" onClick={handleForgotPassword} className="text-sm text-indigo-400 hover:underline" disabled={isLoading}>
                                            Forgot password?
                                        </button>
                                    </div>

                  <div className="space-y-2 relative">
                    <Label>Face Verification</Label>
                                        <div className="w-full h-40 bg-slate-800 rounded-lg overflow-hidden relative">
                                            <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover"></video>
                                            <div className="absolute left-2 top-2 bg-black/50 text-xs text-white px-2 py-1 rounded">{getVideoStatus()}</div>
                                            {captureMessage && (
                                                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                                                    <p className="text-white text-lg font-bold">{captureMessage}</p>
                                                </div>
                                            )}
                                        </div>
                  </div>
                  <div className="pt-2">
                    <Button type="submit" className="w-full" isLoading={isLoading}>
                      Verify Face & Sign In
                    </Button>
                  </div>
                                    <div className="pt-2">
                                                                                <button type="button" onClick={handleVerifyRetry} disabled={isLoading} className="w-full mt-2 inline-flex items-center justify-center rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800">
                                            Retry Verification
                                        </button>
                                    </div>

                                    <div className="text-center text-sm text-slate-400 pt-1">
                                        Don&apos;t have an account?{" "}
                                        <button
                                            type="button"
                                            className="text-indigo-400 hover:underline"
                                            onClick={() => {
                                                setAuthMode("signup");
                                                setCurrentStep("details");
                                            }}
                                            disabled={isLoading}
                                        >
                                            Sign up
                                        </button>
                                    </div>
                </form>
              )}

              {/* --- Sign Up Details Step --- */}
              {authMode === "signup" && currentStep === "details" && (
                <form onSubmit={handleProceedToFaceStep} className="space-y-4">
                  <h3 className="text-2xl font-bold text-center text-slate-100">Create Account</h3>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="fullName">Full Name</Label>
                      <Input id="fullName" name="fullName" type="text" required />
                    </div>
                    <div>
                      <Label htmlFor="roleId">{idLabel}</Label>
                      <Input id="roleId" name="roleId" type="text" required />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="email">Email</Label>
                                            <Input
                                                id="email"
                                                name="email"
                                                type="email"
                                                required
                                                onBlur={(e: React.FocusEvent<HTMLInputElement>) => {
                                                    const v = e.target.value;
                                                    if (v && !validateEmail(v)) showToast("Invalid email. Please enter a valid email.", "error");
                                                }}
                                            />
                    </div>
                    <div>
                                            <Label htmlFor="phoneNumber">Phone Number (India)</Label>
                                            <Input
                                                id="phoneNumber"
                                                name="phoneNumber"
                                                type="tel"
                                                inputMode="numeric"
                                                placeholder="10-digit mobile number"
                                                required
                                                onBlur={(e: React.FocusEvent<HTMLInputElement>) => {
                                                    const v = e.target.value;
                                                    if (v && !validateIndianPhone(v))
                                                        showToast("Invalid phone number. Use 10 digits starting with 6-9.", "error");
                                                }}
                                            />
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="institution">Institution</Label>
                    <Select
                      id="institution"
                      name="institution"
                      value={institution}
                      onChange={(e: ChangeEvent<HTMLSelectElement>) => {
                        setInstitution(e.target.value);
                        setDepartment("");
                      }}
                      required
                    >
                      <option value="">Select Institution</option>
                      {Object.keys(INSTITUTIONS).map((inst) => (
                        <option key={inst} value={inst}>
                          {inst}
                        </option>
                      ))}
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="department">Department</Label>
                    <Select
                      id="department"
                      name="department"
                      value={department}
                      onChange={(e: ChangeEvent<HTMLSelectElement>) => setDepartment(e.target.value)}
                      disabled={!institution}
                      required
                    >
                      <option value="">Select Department</option>
                      {institution && INSTITUTIONS[institution]?.map((dept) => <option key={dept} value={dept}>{dept}</option>)}
                    </Select>
                  </div>
                  {initialRole === "student" && (
                    <div>
                      <Label htmlFor="year">Year of Study</Label>
                                            <Select id="year" name="year" required defaultValue="">
                                                <option value="" disabled>
                                                    Select Year
                                                </option>
                                                <option value="1">1</option>
                                                <option value="2">2</option>
                                                <option value="3">3</option>
                                                <option value="4">4</option>
                                            </Select>
                    </div>
                  )}
                  <div className="relative">
                    <Label htmlFor="password">Password</Label>
                                        <Input
                                            id="password"
                                            name="password"
                                            type={showPassword ? "text" : "password"}
                                            required
                                            value={signupPassword}
                                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSignupPassword(e.target.value)}
                                            onBlur={(e: React.FocusEvent<HTMLInputElement>) => {
                                                const v = e.target.value || '';
                                                if (!v) return;
                                                const missing = getPasswordMissing(v);
                                                if (missing.length > 0) {
                                                    showToast(`Password is invalid. Add: ${missing.join(', ')}.`, 'error');
                                                }
                                            }}
                                        />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-8 text-slate-400 hover:text-slate-200"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>

                                    {/* Password rules checklist */}
                                    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                                        <p className="text-xs font-semibold text-slate-300 mb-2">Password must contain:</p>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                            <div className="flex items-center gap-2 text-xs">
                                                {PASSWORD_RULES.minLength(signupPassword) ? (
                                                    <CheckCircle className="h-4 w-4 text-green-400" />
                                                ) : (
                                                    <XCircle className="h-4 w-4 text-slate-500" />
                                                )}
                                                <span className={cn(PASSWORD_RULES.minLength(signupPassword) ? 'text-slate-200' : 'text-slate-400')}>At least 8 characters</span>
                                            </div>
                                            <div className="flex items-center gap-2 text-xs">
                                                {PASSWORD_RULES.hasUpper(signupPassword) ? (
                                                    <CheckCircle className="h-4 w-4 text-green-400" />
                                                ) : (
                                                    <XCircle className="h-4 w-4 text-slate-500" />
                                                )}
                                                <span className={cn(PASSWORD_RULES.hasUpper(signupPassword) ? 'text-slate-200' : 'text-slate-400')}>One uppercase letter (A-Z)</span>
                                            </div>
                                            <div className="flex items-center gap-2 text-xs">
                                                {PASSWORD_RULES.hasLower(signupPassword) ? (
                                                    <CheckCircle className="h-4 w-4 text-green-400" />
                                                ) : (
                                                    <XCircle className="h-4 w-4 text-slate-500" />
                                                )}
                                                <span className={cn(PASSWORD_RULES.hasLower(signupPassword) ? 'text-slate-200' : 'text-slate-400')}>One lowercase letter (a-z)</span>
                                            </div>
                                            <div className="flex items-center gap-2 text-xs">
                                                {PASSWORD_RULES.hasDigit(signupPassword) ? (
                                                    <CheckCircle className="h-4 w-4 text-green-400" />
                                                ) : (
                                                    <XCircle className="h-4 w-4 text-slate-500" />
                                                )}
                                                <span className={cn(PASSWORD_RULES.hasDigit(signupPassword) ? 'text-slate-200' : 'text-slate-400')}>One number (0-9)</span>
                                            </div>
                                            <div className="flex items-center gap-2 text-xs sm:col-span-2">
                                                {PASSWORD_RULES.hasSymbol(signupPassword) ? (
                                                    <CheckCircle className="h-4 w-4 text-green-400" />
                                                ) : (
                                                    <XCircle className="h-4 w-4 text-slate-500" />
                                                )}
                                                <span className={cn(PASSWORD_RULES.hasSymbol(signupPassword) ? 'text-slate-200' : 'text-slate-400')}>One special character (!@#$%^&*-_=+)</span>
                                            </div>
                                        </div>
                                        {signupPassword.length > 0 && (
                                            <div className="mt-2 text-xs">
                                                {isPasswordValid(signupPassword) ? (
                                                    <span className="text-green-300">Password looks good.</span>
                                                ) : (
                                                    <span className="text-slate-400">Keep going—you're almost there.</span>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    <div className="relative">
                                        <Label htmlFor="confirmPassword">Re-enter Password</Label>
                                                                                <Input
                                                                                    id="confirmPassword"
                                                                                    name="confirmPassword"
                                                                                    type={showConfirmPassword ? "text" : "password"}
                                                                                    required
                                                                                    value={signupConfirmPassword}
                                                                                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSignupConfirmPassword(e.target.value)}
                                                                                    onBlur={(e: React.FocusEvent<HTMLInputElement>) => {
                                                                                        const confirm = e.target.value || '';
                                                                                        if (!confirm) return;
                                                                                        if (signupPassword && confirm !== signupPassword) {
                                                                                            showToast('Passwords do not match. Please re-enter the same password.', 'error');
                                                                                        }
                                                                                    }}
                                                                                />
                                        <button
                                            type="button"
                                            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                            className="absolute right-3 top-8 text-slate-400 hover:text-slate-200"
                                        >
                                            {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                        </button>
                                    </div>

                  <div className="pt-2">
                    <Button type="submit" className="w-full">
                      Proceed to Face Registration
                    </Button>
                  </div>

                                    <div className="text-center text-sm text-slate-400 pt-1">
                                        Already have an account?{" "}
                                        <button
                                            type="button"
                                            className="text-indigo-400 hover:underline"
                                            onClick={() => {
                                                setAuthMode("signin");
                                                setCurrentStep("details");
                                            }}
                                            disabled={isLoading}
                                        >
                                            Sign in
                                        </button>
                                    </div>
                </form>
              )}

              {/* --- Sign Up Face Step --- */}
              {authMode === "signup" && currentStep === "face" && (
                <div className="text-center space-y-4">
                  <h3 className="text-2xl font-bold text-slate-100">Register Face ID</h3>
                  <p className="text-slate-400">Center your face in the frame for verification.</p>
                                    <div className="w-48 h-48 bg-slate-800 rounded-full mx-auto overflow-hidden border-2 border-dashed border-slate-600 relative">
                                        <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover"></video>
                                        <div className="absolute left-2 top-2 bg-black/50 text-xs text-white px-2 py-1 rounded">{getVideoStatus()}</div>
                                        {captureMessage && (
                                            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                                                <p className="text-white text-3xl font-bold">{captureMessage}</p>
                                            </div>
                                        )}
                                    </div>
                                    <div className="space-y-2">
                                        <div className="flex items-center justify-between">
                                            <span className="text-xs text-slate-400">Samples captured: {enrollSamples.length} / 5</span>
                                            <Button type="button" variant="secondary" onClick={async () => {
                                                const v = videoRef.current; if (!v) { showToast('Camera not ready', 'error'); return; }
                                                try {
                                                    // Try instant capture; if it fails, reacquire stream once and retry immediately.
                                                    let dataUrl = captureFrame();
                                                    if (!dataUrl) {
                                                        try {
                                                            const newStream = await navigator.mediaDevices.getUserMedia({ video: true });
                                                            v.srcObject = newStream;
                                                        } catch {}
                                                        // No long waits—attempt quick ready check then capture
                                                        try { await waitForVideoReady(v, 1500); } catch {}
                                                        dataUrl = captureFrame();
                                                    }
                                                    if (!dataUrl) throw new Error('Capture failed');
                                                    setEnrollSamples(s => Array.from(new Set([...(s||[]), dataUrl])).slice(0,5));
                                                    showToast('Sample added', 'success');
                                                } catch (err:any) {
                                                    showToast(err.message || 'Failed to add sample', 'error');
                                                }
                                            }}>Add another sample</Button>
                                        </div>
                                        {enrollSamples.length > 0 && (
                                            <div className="grid grid-cols-5 gap-2">
                                                {enrollSamples.map((s, i) => (
                                                    <img key={i} src={s} className="w-full h-16 object-cover rounded border border-slate-700" />
                                                ))}
                                            </div>
                                        )}
                                        <Button onClick={handleFullSignUp} className="w-full" isLoading={isLoading}>
                                            Complete Signup
                                        </Button>
                                    </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </motion.div>
      </div>
    </motion.div>
  );
};




const DashboardLayout = ({ children, user, onLogout, onBack, onAction, onUpdateUser, showToast }: { children: React.ReactNode, user: UserProfile, onLogout?: () => void, onBack: () => void, onAction?: (action: string) => void, onUpdateUser?: (u: UserProfile) => void, showToast?: (msg:string, type:'success'|'error') => void }) => {
    const [faceDialogOpen, setFaceDialogOpen] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [faceSamples, setFaceSamples] = useState<string[]>([]);

    const handleAction = (action: string) => {
        if (onAction) onAction(action);
    };

    return (
        <>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex min-h-screen">
            <Sidebar user={user} onAction={handleAction} />
            <main className="flex-1 p-8 bg-slate-950/50 overflow-y-auto">
                <div className="flex justify-between items-center mb-8">
                    <h1 className="text-3xl font-bold text-white">Welcome back, {user.name.split(' ')[0]}!</h1>
                    <div className="flex items-center space-x-2">
                        <Button variant="outline" onClick={onBack}><ChevronLeft className="h-4 w-4 mr-2" /> Back to Home</Button>
                        {onLogout && <Button variant="destructive" onClick={onLogout}>Logout</Button>}
                    </div>
                </div>
                {children}
            </main>
        </motion.div>

        <Dialog open={faceDialogOpen} onOpenChange={setFaceDialogOpen} className="max-w-lg">
            <h2 className="text-xl font-bold mb-2">Face Samples</h2>
            <p className="text-slate-400 text-sm mb-2">Add up to 5 samples in different lighting/expressions. These help improve verification.</p>
            <div className="w-full h-48 bg-slate-800 rounded-lg overflow-hidden relative mb-2">
                <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover"></video>
            </div>
            <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Pending to add: {faceSamples.length}</span>
                <div className="space-x-2">
                    <Button variant="secondary" onClick={async () => {
                        const v = videoRef.current; if (!v) return;
                        try {
                            const dataUrl = (() => {
                                const canvas = document.createElement('canvas');
                                canvas.width = 320; canvas.height = 240;
                                const ctx = canvas.getContext('2d');
                                if (!ctx) return null;
                                ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
                                return canvas.toDataURL('image/jpeg');
                            })();
                            if (!dataUrl) throw new Error('Capture failed');
                            setFaceSamples(s => [...s, dataUrl].slice(0,5));
                        } catch (e:any) { showToast?.(e.message || 'Failed to capture', 'error'); }
                    }}>Capture Sample</Button>
                    <Button onClick={async () => {
                        if (faceSamples.length === 0) { setFaceDialogOpen(false); return; }
                        try {
                            const res = await fetch(`${API_URL}/users/${user._id}/face-samples`, {
                                method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ imageDataUrls: faceSamples })
                            });
                            const data = await res.json();
                            if (!res.ok) throw new Error(data.error || 'Failed to add samples');
                            showToast?.('Samples added', 'success');
                            setFaceSamples([]);
                            setFaceDialogOpen(false);
                        } catch (err:any) { showToast?.(err.message || 'Failed to add samples', 'error'); }
                    }}>Save</Button>
                </div>
            </div>
        </Dialog>
        </>
    );
};

// --- Results Analysis Page ---
const ProfilePage = ({ user, onLogout, onBack, showToast, onUpdateUser, navigateTo }: { user: UserProfile; onLogout: () => void; onBack: () => void; showToast: (message:string, type:'success'|'error') => void; onUpdateUser: (u: UserProfile) => void; navigateTo: (state: AppState) => void }) => {
    const [profileForm, setProfileForm] = useState<any>({
        name: user.name,
        phoneNumber: user.phoneNumber,
        institution: user.institution,
        department: user.department,
        year: user.year,
        studentId: user.studentId,
        lecturerId: user.lecturerId
    });
    const [isEditing, setIsEditing] = useState(false);

    const saveProfile = async () => {
        try {
            const res = await fetch(`${API_URL}/users/${user._id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(profileForm)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to update profile');
            if (onUpdateUser && data.user) onUpdateUser(data.user as UserProfile);
            setIsEditing(false);
            showToast('Profile updated successfully', 'success');
        } catch (err: any) {
            showToast(err.message || 'Failed to update profile', 'error');
        }
    };

    return (
        <DashboardLayout user={user} onLogout={onLogout} onBack={onBack} showToast={showToast} onUpdateUser={onUpdateUser} onAction={(a) => {
            if (a === 'dashboard') { navigateTo(user.role === 'student' ? 'student-dashboard' : 'lecturer-dashboard'); return; }
            if (a === 'my-exams') { navigateTo('my-exams'); return; }
            if (a === 'results') { navigateTo('results-analysis'); return; }
            if (a === 'help') { navigateTo('help'); return; }
        }}>
            <div className="max-w-5xl mx-auto space-y-6">
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold text-white">Profile Settings</h1>
                    {!isEditing && (
                        <Button onClick={() => setIsEditing(true)}>
                            <Edit className="h-4 w-4 mr-2" /> Edit Profile
                        </Button>
                    )}
                </div>

                <Card className="p-8 bg-slate-900 border-slate-800">
                    <div className="space-y-8">
                        {/* Account Information */}
                        <div>
                            <h2 className="text-xl font-semibold text-white mb-6 flex items-center">
                                <User className="h-5 w-5 mr-2" />
                                Account Information
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                    <Label className="text-base text-slate-300">Email</Label>
                                    <Input className="mt-2 bg-slate-800/50" value={user.email} disabled />
                                    <p className="text-xs text-slate-500 mt-1">Email cannot be changed</p>
                                </div>
                                <div>
                                    <Label className="text-base text-slate-300">Role</Label>
                                    <Input className="mt-2 bg-slate-800/50 capitalize" value={user.role} disabled />
                                </div>
                            </div>
                        </div>

                        {/* Personal Information */}
                        <div className="border-t border-slate-800 pt-8">
                            <h2 className="text-xl font-semibold text-white mb-6">Personal Information</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                    <Label className="text-base text-slate-300">Full Name</Label>
                                    <Input
                                        className="mt-2"
                                        value={profileForm.name}
                                        onChange={(e: any) => setProfileForm((p: any) => ({ ...p, name: e.target.value }))}
                                        disabled={!isEditing}
                                    />
                                </div>
                                <div>
                                    <Label className="text-base text-slate-300">Phone Number</Label>
                                    <Input
                                        className="mt-2"
                                        value={profileForm.phoneNumber}
                                        onChange={(e: any) => setProfileForm((p: any) => ({ ...p, phoneNumber: e.target.value }))}
                                        disabled={!isEditing}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Institution Details */}
                        <div className="border-t border-slate-800 pt-8">
                            <h2 className="text-xl font-semibold text-white mb-6">Institution Details</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                    <Label className="text-base text-slate-300">Institution</Label>
                                    <Select
                                        className="mt-2"
                                        value={profileForm.institution || ''}
                                        onChange={(e: any) => setProfileForm((p: any) => ({ ...p, institution: e.target.value, department: '' }))}
                                        disabled={!isEditing}
                                    >
                                        <option value="">Select Institution</option>
                                        {Object.keys(INSTITUTIONS).map(inst => (
                                            <option key={inst} value={inst}>{inst}</option>
                                        ))}
                                    </Select>
                                </div>
                                <div>
                                    <Label className="text-base text-slate-300">Department</Label>
                                    <Select
                                        className="mt-2"
                                        value={profileForm.department || ''}
                                        onChange={(e: any) => setProfileForm((p: any) => ({ ...p, department: e.target.value }))}
                                        disabled={!isEditing || !profileForm.institution}
                                    >
                                        <option value="">Select Department</option>
                                        {profileForm.institution && INSTITUTIONS[profileForm.institution]?.map((dept) => (
                                            <option key={dept} value={dept}>{dept}</option>
                                        ))}
                                    </Select>
                                </div>
                                {user.role === 'student' && (
                                    <div>
                                        <Label className="text-base text-slate-300">Year</Label>
                                        <Select
                                            className="mt-2"
                                            value={profileForm.year || ''}
                                            onChange={(e: any) => setProfileForm((p: any) => ({ ...p, year: e.target.value }))}
                                            disabled={!isEditing}
                                        >
                                            <option value="">Select Year</option>
                                            <option value="1">1</option>
                                            <option value="2">2</option>
                                            <option value="3">3</option>
                                            <option value="4">4</option>
                                        </Select>
                                    </div>
                                )}
                                <div>
                                    <Label className="text-base text-slate-300">{user.role === 'student' ? 'Student ID' : 'Lecturer ID'}</Label>
                                    <Input
                                        className="mt-2 bg-slate-800/50"
                                        value={user.role === 'student' ? user.studentId : user.lecturerId}
                                        disabled
                                    />
                                    <p className="text-xs text-slate-500 mt-1">ID cannot be changed</p>
                                </div>
                            </div>
                        </div>

                        {/* Action Buttons */}
                        {isEditing && (
                            <div className="flex justify-end space-x-3 pt-6 border-t border-slate-800">
                                <Button variant="outline" onClick={() => {
                                    setProfileForm({
                                        name: user.name,
                                        phoneNumber: user.phoneNumber,
                                        institution: user.institution,
                                        department: user.department,
                                        year: user.year,
                                        studentId: user.studentId,
                                        lecturerId: user.lecturerId
                                    });
                                    setIsEditing(false);
                                }} className="px-6">
                                    Cancel
                                </Button>
                                <Button onClick={saveProfile} className="px-6">
                                    <Save className="h-4 w-4 mr-2" /> Save Changes
                                </Button>
                            </div>
                        )}
                    </div>
                </Card>
            </div>
        </DashboardLayout>
    );
};

const ResultsAnalysisPage = ({ user, exams, onLogout, onBack, showToast, onUpdateUser, navigateTo }: { user: UserProfile; exams: Exam[]; onLogout: () => void; onBack: () => void; showToast: (message:string, type:'success'|'error') => void; onUpdateUser: (u: UserProfile) => void; navigateTo: (state: AppState) => void }) => {
    const [selectedExam, setSelectedExam] = useState<Exam | null>(null);
    const [resultAttempt, setResultAttempt] = useState<any>(null);
    const [expandedQuestion, setExpandedQuestion] = useState<number | null>(null);

    const userExams = exams.filter(exam => 
        exam.institution.toLowerCase() === user.institution.toLowerCase() && 
        exam.department.toLowerCase() === user.department.toLowerCase() && 
        exam.targetYear === user.year
    );

    const completedExams = userExams.filter(e => (e as any).attemptForUser || (e as any).completedByUser);
    const averageScore = completedExams.length > 0 ? Math.round(completedExams.reduce((acc, e) => {
        const attempt = (e as any).attemptForUser;
        return acc + (attempt?.score || 0);
    }, 0) / completedExams.length) : 0;

    const loadExamResults = async (exam: Exam) => {
        // Toggle: if clicking same exam, close it
        if (selectedExam?._id === exam._id) {
            setSelectedExam(null);
            setResultAttempt(null);
            setExpandedQuestion(null);
            return;
        }
        try {
            const res = await fetch(`${API_URL}/exams/${exam._id}/attempt?userId=${user._id}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load results');
            if (!data.attempt) {
                showToast('No results found for this exam.', 'error');
                return;
            }
            setResultAttempt(data.attempt);
            setSelectedExam(exam);
        } catch (err: any) {
            showToast(err.message || 'Failed to load results', 'error');
        }
    };

    return (
        <DashboardLayout user={user} onLogout={onLogout} onBack={onBack} showToast={showToast} onUpdateUser={onUpdateUser} onAction={(a) => {
            if (a === 'dashboard') { navigateTo('student-dashboard'); return; }
            if (a === 'my-exams') { navigateTo('my-exams'); return; }
            if (a === 'results') { /* already here */ return; }
            if (a === 'help') { navigateTo('help'); return; }
        }}>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold text-white">Results & Analysis</h1>
                </div>

                {/* Summary Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card className="p-6 bg-gradient-to-br from-indigo-600/20 to-indigo-800/20 border-indigo-500/50">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">Average Score</p>
                                <p className="text-4xl font-bold text-white mt-2">{averageScore}%</p>
                            </div>
                            <BrainCircuit className="h-12 w-12 text-indigo-400" />
                        </div>
                    </Card>
                    <Card className="p-6 bg-gradient-to-br from-green-600/20 to-green-800/20 border-green-500/50">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">Exams Completed</p>
                                <p className="text-4xl font-bold text-white mt-2">{completedExams.length}</p>
                            </div>
                            <Monitor className="h-12 w-12 text-green-400" />
                        </div>
                    </Card>
                    <Card className="p-6 bg-gradient-to-br from-yellow-600/20 to-yellow-800/20 border-yellow-500/50">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">Best Score</p>
                                <p className="text-4xl font-bold text-white mt-2">
                                    {completedExams.length > 0 ? Math.max(...completedExams.map(e => (e as any).attemptForUser?.score || 0)) : 0}%
                                </p>
                            </div>
                            <Monitor className="h-12 w-12 text-yellow-400" />
                        </div>
                    </Card>
                </div>

                {/* Exam Results List */}
                <Card className="p-6">
                    <h2 className="text-2xl font-bold text-white mb-4">Your Exam Results</h2>
                    {completedExams.length > 0 ? (
                        <div className="space-y-4">
                            {completedExams.map((exam: Exam) => {
                                const attempt = (exam as any).attemptForUser;
                                const score = attempt?.score || 0;
                                const correctCount = Array.isArray(attempt?.answers)
                                    ? attempt.answers.filter((a: any) => a?.isCorrect === true).length
                                    : 0;
                                const totalQuestions = Array.isArray(attempt?.answers)
                                    ? attempt.answers.length
                                    : (exam as any)?.questions?.length ?? 0;
                                return (
                                    <div key={exam._id} 
                                        className="p-4 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors cursor-pointer border border-slate-700 hover:border-indigo-500"
                                        onClick={() => loadExamResults(exam)}
                                    >
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <h3 className="text-lg font-semibold text-white">{exam.title}</h3>
                                                <p className="text-slate-400 text-sm mt-1">
                                                    Score: <span className="text-white font-medium">{score}%</span>
                                                    {totalQuestions ? (
                                                        <>
                                                            {' '}•{' '}
                                                            {correctCount}/{totalQuestions} correct
                                                        </>
                                                    ) : null}
                                                </p>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span
                                                    className={cn(
                                                        'px-3 py-1 rounded-full text-sm font-semibold',
                                                        score >= 70
                                                            ? 'bg-green-900/30 text-green-300 border border-green-600/40'
                                                            : score >= 40
                                                                ? 'bg-yellow-900/30 text-yellow-300 border border-yellow-600/40'
                                                                : 'bg-red-900/30 text-red-300 border border-red-600/40'
                                                    )}
                                                >
                                                    {score}%
                                                </span>
                                                <ChevronLeft className="h-4 w-4 text-slate-500 rotate-180" />
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <p className="text-slate-400 text-center py-12">No completed exams yet. Take an exam to see your results here!</p>
                    )}
                </Card>

                {/* Detailed Results View */}
                {selectedExam && resultAttempt && (
                    <Card className="p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-2xl font-bold text-white">{selectedExam.title} - Detailed Results</h2>
                            <Button variant="outline" onClick={() => { setSelectedExam(null); setResultAttempt(null); }}>Close</Button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            <div className="p-4 bg-slate-800 rounded-lg">
                                <p className="text-slate-400 text-sm mb-1">Final Score</p>
                                <p className="text-3xl font-bold text-white">{resultAttempt.score}%</p>
                            </div>
                            <div className="p-4 bg-slate-800 rounded-lg">
                                <p className="text-slate-400 text-sm mb-1">Completed At</p>
                                <p className="text-lg font-semibold text-white">{new Date(resultAttempt.completedAt).toLocaleString()}</p>
                            </div>
                        </div>

                        <h3 className="text-xl font-bold text-white mb-4">Question Breakdown</h3>
                        <div className="space-y-3">
                            {resultAttempt.perQuestion && resultAttempt.perQuestion.map((q: any, idx: number) => (
                                <div key={idx}>
                                    <div 
                                        onClick={() => setExpandedQuestion(expandedQuestion === idx ? null : idx)}
                                        className={cn(
                                            'p-4 rounded-lg border cursor-pointer transition-all hover:scale-[1.01]',
                                            q.correct ? 'bg-green-900/20 border-green-600/50 hover:border-green-500' : 'bg-red-900/20 border-red-600/50 hover:border-red-500'
                                        )}
                                    >
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <p className="text-white font-medium">Question {idx + 1}</p>
                                                {q.question && (
                                                    <p className="text-slate-300 text-sm mt-1 line-clamp-2">{q.question}</p>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {q.correct ? (
                                                    <CheckCircle className="h-5 w-5 text-green-400" />
                                                ) : (
                                                    <XCircle className="h-5 w-5 text-red-400" />
                                                )}
                                                <span className={cn('text-sm font-semibold', q.correct ? 'text-green-300' : 'text-red-300')}>
                                                    {q.correct ? 'Correct' : 'Incorrect'}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    {expandedQuestion === idx && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            className="mt-2 p-4 bg-slate-800 rounded-lg border border-slate-700 space-y-3"
                                        >
                                            <div>
                                                <p className="text-sm font-semibold text-slate-300 mb-2">Question:</p>
                                                <p className="text-white">{q.question}</p>
                                            </div>
                                            {q.options && q.options.length > 0 && (
                                                <div>
                                                    <p className="text-sm font-semibold text-slate-300 mb-2">Options:</p>
                                                    <div className="space-y-2">
                                                        {q.options.map((opt: string, optIdx: number) => {
                                                            const userAns = q.given !== undefined ? q.given : q.userAnswer;
                                                            const correctAns = q.expected !== undefined ? q.expected : q.correctAnswer;
                                                            return (
                                                            <div key={optIdx} className={cn(
                                                                'p-2 rounded border',
                                                                userAns === optIdx ? (q.correct ? 'border-green-500 bg-green-900/20' : 'border-red-500 bg-red-900/20') :
                                                                correctAns === optIdx ? 'border-green-500 bg-green-900/10' : 'border-slate-700'
                                                            )}>
                                                                <div className="flex items-center justify-between">
                                                                    <span className="text-slate-200">{String.fromCharCode(65 + optIdx)}. {opt}</span>
                                                                    <div className="flex gap-2">
                                                                        {userAns === optIdx && <span className="text-xs text-blue-400 font-medium">Your Answer</span>}
                                                                        {correctAns === optIdx && <span className="text-xs text-green-400 font-medium">✓ Correct</span>}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            )}
                                            {!q.options && (
                                                <div>
                                                    <p className="text-sm font-semibold text-slate-300 mb-1">Your Answer:</p>
                                                    <p className="text-white mb-3">{String(q.given || q.userAnswer || 'No answer')}</p>
                                                    <p className="text-sm font-semibold text-slate-300 mb-1">Correct Answer:</p>
                                                    <p className="text-green-400">{String(q.expected || q.correctAnswer)}</p>
                                                </div>
                                            )}
                                            <div className="flex items-center justify-between pt-3 border-t border-slate-700">
                                                <span className="text-sm text-slate-400">Marks: <span className={q.correct ? 'text-green-400' : 'text-red-400'}>{q.correct ? q.marks : 0}</span> / {q.marks}</span>
                                                <span className={cn('text-sm font-semibold', q.correct ? 'text-green-400' : 'text-red-400')}>{q.correct ? '✓ Correct' : '✗ Incorrect'}</span>
                                            </div>
                                        </motion.div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </Card>
                )}
            </div>
        </DashboardLayout>
    );
};

const MyExamsPage = ({ user, exams, onLogout, onStartExam, showToast, onUpdateUser, navigateTo }: { user: UserProfile; exams: Exam[]; onLogout: () => void; onStartExam: (examId: string) => void; showToast: (message:string, type:'success'|'error') => void; onUpdateUser: (u: UserProfile) => void; navigateTo: (state: AppState) => void }) => {
    const [activeTab, setActiveTab] = useState<'upcoming' | 'finished'>('upcoming');
    const [selectedExam, setSelectedExam] = useState<Exam | null>(null);
    const [detailsOpen, setDetailsOpen] = useState(false);
    const [resultsOpen, setResultsOpen] = useState(false);
    const [resultAttempt, setResultAttempt] = useState<any>(null);

    const userExams = exams.filter(exam => 
        exam.institution.toLowerCase() === user.institution.toLowerCase() && 
        exam.department.toLowerCase() === user.department.toLowerCase() && 
        exam.targetYear === user.year
    );

    const now = new Date();
    const isWithinWindow = (e: any) => {
        try {
            if (!e.scheduledDate) return e.status === 'Available' || e.status === 'Live';
            const date = new Date(e.scheduledDate);
            const [sh, sm] = (e.startTime || '00:00').split(':').map((n: string) => parseInt(n || '0', 10));
            const [eh, em] = (e.endTime || '23:59').split(':').map((n: string) => parseInt(n || '0', 10));
            const start = new Date(date.getFullYear(), date.getMonth(), date.getDate(), sh || 0, sm || 0);
            const end = new Date(date.getFullYear(), date.getMonth(), date.getDate(), eh || 23, em || 59);
            return now >= start && now <= end;
        } catch { return false; }
    };
    const isExpired = (e: any) => {
        try {
            if (!e.scheduledDate) return false;
            const date = new Date(e.scheduledDate);
            const [eh, em] = (e.endTime || '23:59').split(':').map((n: string) => parseInt(n || '0', 10));
            const end = new Date(date.getFullYear(), date.getMonth(), date.getDate(), eh || 23, em || 59);
            return now > end;
        } catch { return false; }
    };

    const upcomingExams = userExams.filter(e => 
        ((e.status === 'Scheduled' || e.status === 'Available' || e.status === 'Live' || e.status === 'Locked') && !isExpired(e)) &&
        !(e as any).completedByUser && !(e as any).attemptForUser
    );
    const finishedExams = userExams.filter(e => (e as any).attemptForUser || (e as any).completedByUser);

    const openDetails = (exam: Exam) => {
        setSelectedExam(exam);
        setDetailsOpen(true);
    };

    const openResults = async (exam: Exam) => {
        try {
            const res = await fetch(`${API_URL}/exams/${exam._id}/attempt?userId=${user._id}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load results');
            if (!data.attempt) {
                showToast('No results found for this exam yet.', 'error');
                return;
            }
            setResultAttempt(data.attempt);
            setSelectedExam(exam);
            setResultsOpen(true);
        } catch (err: any) {
            showToast(err.message || 'Failed to load results', 'error');
        }
    };

    return (
        <DashboardLayout user={user} onLogout={onLogout} onBack={() => navigateTo('student-dashboard')} showToast={showToast} onUpdateUser={onUpdateUser} onAction={(a) => {
            if (a === 'dashboard') { navigateTo('student-dashboard'); return; }
            if (a === 'my-exams') { /* already here */ return; }
            if (a === 'results') { navigateTo('results-analysis'); return; }
            if (a === 'profile') { navigateTo('profile'); return; }
            if (a === 'live-proctoring') { navigateTo('live-proctoring'); return; }
            if (a === 'help') { navigateTo('help'); return; }
        }}>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold text-white">My Exams</h1>
                </div>

                {/* Tabs */}
                <div className="flex space-x-2 border-b border-slate-800">
                    <button
                        onClick={() => setActiveTab('upcoming')}
                        className={cn(
                            'px-6 py-3 font-semibold transition-colors duration-200',
                            activeTab === 'upcoming' 
                                ? 'text-indigo-400 border-b-2 border-indigo-400' 
                                : 'text-slate-400 hover:text-slate-200'
                        )}
                    >
                        Upcoming ({upcomingExams.length})
                    </button>
                    <button
                        onClick={() => setActiveTab('finished')}
                        className={cn(
                            'px-6 py-3 font-semibold transition-colors duration-200',
                            activeTab === 'finished' 
                                ? 'text-indigo-400 border-b-2 border-indigo-400' 
                                : 'text-slate-400 hover:text-slate-200'
                        )}
                    >
                        Finished ({finishedExams.length})
                    </button>
                </div>

                {/* Content */}
                <div className="space-y-4">
                    {activeTab === 'upcoming' && (
                        <>
                            {upcomingExams.length > 0 ? upcomingExams.map((exam: Exam) => (
                                <Card key={exam._id} className="p-4 hover:border-indigo-500 transition-colors duration-200">
                                    <div className="flex items-center justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-3 mb-2">
                                                <h3 className="font-semibold text-white text-lg">{exam.title}</h3>
                                                <Badge variant={exam.status === 'Available' ? 'success' : exam.status === 'Scheduled' ? 'info' : exam.status === 'Live' ? 'live' : 'warning'}>{exam.status}</Badge>
                                            </div>
                                            <p className="text-sm text-slate-400 mb-1">{exam.courseCode} | {exam.description}</p>
                                            <p className="text-sm text-slate-400">{new Date(exam.scheduledDate).toLocaleDateString()} @ {exam.startTime} - {exam.endTime}</p>
                                            <p className="text-xs text-slate-500 mt-1">Duration: {exam.duration} minutes | Questions: {exam.questions?.length || 0}</p>
                                        </div>
                                        <div className="flex space-x-2">
                                            <Button variant="outline" onClick={() => openDetails(exam)}>Details</Button>
                                            {(() => {
                                                const completed = (exam as any).completedByUser || (exam as any).attemptForUser;
                                                const serverCanStart = (exam as any).canStartForUser;
                                                const canStart = serverCanStart !== undefined ? serverCanStart : (exam.status === 'Available' || exam.status === 'Live' || isWithinWindow(exam));
                                                
                                                if (completed) return <Button disabled>Completed</Button>;
                                                return (
                                                    <Button onClick={() => onStartExam(exam._id)} disabled={!canStart}>
                                                        {exam.status === 'Locked' ? <Lock className="h-4 w-4 mr-2"/> : null}
                                                        {canStart ? 'Start Exam' : 'Not Available'}
                                                    </Button>
                                                );
                                            })()}
                                        </div>
                                    </div>
                                </Card>
                            )) : <p className="text-slate-400 text-center py-8">No upcoming exams scheduled.</p>}
                        </>
                    )}

                    {activeTab === 'finished' && (
                        <>
                            {finishedExams.length > 0 ? finishedExams.map((exam: Exam) => (
                                <Card key={exam._id} className="p-4 hover:border-green-500 transition-colors duration-200">
                                    <div className="flex items-center justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-3 mb-2">
                                                <h3 className="font-semibold text-white text-lg">{exam.title}</h3>
                                                <Badge variant="success">Completed</Badge>
                                            </div>
                                            <p className="text-sm text-slate-400 mb-1">{exam.courseCode} | {exam.description}</p>
                                            <p className="text-sm text-slate-400">Completed: {new Date((exam as any).attemptForUser?.completedAt || Date.now()).toLocaleDateString()}</p>
                                            <p className="text-sm text-green-400 mt-1 font-semibold">Score: {(exam as any).attemptForUser?.score || exam.attempt?.score || 0}%</p>
                                        </div>
                                        <div className="flex space-x-2">
                                            <Button onClick={() => openResults(exam)}>See Results</Button>
                                        </div>
                                    </div>
                                </Card>
                            )) : <p className="text-slate-400 text-center py-8">No finished exams yet.</p>}
                        </>
                    )}
                </div>
            </div>

            {/* Details Dialog */}
            <Dialog open={detailsOpen} onOpenChange={setDetailsOpen} className="max-w-2xl">
                <div className="p-6 space-y-4">
                    <h2 className="text-2xl font-bold text-white">{selectedExam?.title}</h2>
                    <div className="space-y-2 text-sm">
                        <p className="text-slate-300"><span className="font-semibold">Course Code:</span> {selectedExam?.courseCode}</p>
                        <p className="text-slate-300"><span className="font-semibold">Description:</span> {selectedExam?.description}</p>
                        <p className="text-slate-300"><span className="font-semibold">Date:</span> {selectedExam && new Date(selectedExam.scheduledDate).toLocaleDateString()}</p>
                        <p className="text-slate-300"><span className="font-semibold">Time:</span> {selectedExam?.startTime} - {selectedExam?.endTime}</p>
                        <p className="text-slate-300"><span className="font-semibold">Duration:</span> {selectedExam?.duration} minutes</p>
                        <p className="text-slate-300"><span className="font-semibold">Questions:</span> {selectedExam?.questions?.length || 0}</p>
                        <p className="text-slate-300"><span className="font-semibold">Lecturer:</span> {selectedExam?.lecturerName}</p>
                    </div>
                    <Button onClick={() => setDetailsOpen(false)} className="w-full mt-4">Close</Button>
                </div>
            </Dialog>

            {/* Results Dialog */}
            <Dialog open={resultsOpen} onOpenChange={setResultsOpen} className="max-w-2xl">
                <div className="p-6 space-y-4 max-h-[80vh] overflow-y-auto">
                    <h2 className="text-2xl font-bold text-white">{selectedExam?.title} - Results</h2>
                    <div className="bg-slate-800 p-4 rounded-lg">
                        <div className="flex justify-between items-center">
                            <span className="text-slate-300">Your Score:</span>
                            <span className="text-3xl font-bold text-green-400">{resultAttempt?.score || 0}%</span>
                        </div>
                        <div className="flex justify-between items-center mt-2">
                            <span className="text-slate-400 text-sm">Points: {resultAttempt?.pointsEarned || 0} / {resultAttempt?.totalMarks || 0}</span>
                            <span className="text-slate-400 text-sm">Completed: {resultAttempt?.completedAt ? new Date(resultAttempt.completedAt).toLocaleString() : 'N/A'}</span>
                        </div>
                    </div>
                    {resultAttempt?.perQuestion && resultAttempt.perQuestion.length > 0 && (
                        <div className="space-y-3">
                            <h3 className="font-semibold text-white">Question Breakdown:</h3>
                            {resultAttempt.perQuestion.map((q: any, i: number) => (
                                <div key={i} className="bg-slate-800/50 p-3 rounded border border-slate-700">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <p className="text-sm text-slate-300 mb-1">Q{i + 1}: {q.question}</p>
                                            <p className="text-xs text-slate-400">Your Answer: <span className={q.correct ? 'text-green-400' : 'text-red-400'}>{String(q.given || q.userAnswer || 'No answer')}</span></p>
                                            {!q.correct && <p className="text-xs text-slate-500">Correct Answer: {String(q.expected || q.correctAnswer)}</p>}
                                        </div>
                                        <div className="ml-4">
                                            {q.correct ? <CheckCircle className="h-5 w-5 text-green-400" /> : <XCircle className="h-5 w-5 text-red-400" />}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                    <Button onClick={() => setResultsOpen(false)} className="w-full mt-4">Close</Button>
                </div>
            </Dialog>
        </DashboardLayout>
    );
};

const StudentDashboard = ({ user, exams, onLogout, onStartExam, onBack, showToast, onUpdateUser, navigateTo }: { user: UserProfile; exams: Exam[]; onLogout: () => void; onStartExam: (examId: string) => void; onBack: () => void; showToast: (message:string, type:'success'|'error') => void; onUpdateUser: (u: UserProfile) => void; navigateTo: (state: AppState) => void }) => {
    const [systemCheckOpen, setSystemCheckOpen] = useState(false);
    const [resultsOpen, setResultsOpen] = useState(false);
    const [resultExamTitle, setResultExamTitle] = useState<string>('');
    const [resultAttempt, setResultAttempt] = useState<any>(null);
    
    const userExams = exams.filter(exam => 
        exam.institution.toLowerCase() === user.institution.toLowerCase() && 
        exam.department.toLowerCase() === user.department.toLowerCase() && 
        exam.targetYear === user.year
    );

    const now = new Date();
    const isWithinWindow = (e: any) => {
        try {
            if (!e.scheduledDate) return e.status === 'Available' || e.status === 'Live';
            const date = new Date(e.scheduledDate);
            const [sh, sm] = (e.startTime || '00:00').split(':').map((n: string) => parseInt(n || '0', 10));
            const [eh, em] = (e.endTime || '23:59').split(':').map((n: string) => parseInt(n || '0', 10));
            const start = new Date(date.getFullYear(), date.getMonth(), date.getDate(), sh || 0, sm || 0);
            const end = new Date(date.getFullYear(), date.getMonth(), date.getDate(), eh || 23, em || 59);
            return now >= start && now <= end;
        } catch { return false; }
    };
    const isExpired = (e: any) => {
        try {
            if (!e.scheduledDate) return false;
            const date = new Date(e.scheduledDate);
            const [eh, em] = (e.endTime || '23:59').split(':').map((n: string) => parseInt(n || '0', 10));
            const end = new Date(date.getFullYear(), date.getMonth(), date.getDate(), eh || 23, em || 59);
            return now > end;
        } catch { return false; }
    };

    const liveExams = userExams.filter(e => !(e as any).attemptForUser && ((e.status === 'Live') || (e.status === 'Available' && isWithinWindow(e))));
    const upcomingExams = userExams.filter(e => !(e as any).attemptForUser && (e.status === 'Scheduled' || e.status === 'Available' || e.status === 'Locked') && !isExpired(e));
    const completedExams = userExams.filter(e => (e as any).attemptForUser); // treat exams with attempts as completed for this user
    const averageScore = completedExams.length > 0 ? Math.round(completedExams.reduce((acc, e) => {
        const attempt = (e as any).attemptForUser;
        return acc + (attempt?.score || 0);
    }, 0) / completedExams.length) : 0;

    const openResults = async (exam: any) => {
        try {
            const res = await fetch(`${API_URL}/exams/${exam._id}/attempt?userId=${user._id}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load results');
            if (!data.attempt) {
                showToast('No results found for this exam yet.', 'error');
                return;
            }
            setResultAttempt(data.attempt);
            setResultExamTitle(exam.title);
            setResultsOpen(true);
        } catch (err: any) {
            showToast(err.message || 'Failed to load results', 'error');
        }
    };

    return (
    <DashboardLayout user={user} onLogout={onLogout} onBack={onBack} showToast={showToast} onUpdateUser={onUpdateUser} onAction={(a) => {
        // Keep students on the main dashboard unless explicitly opening tools
        if (a === 'dashboard') { setSystemCheckOpen(false); return; }
        if (a === 'my-exams') { navigateTo('my-exams'); return; }
        if (a === 'results') { navigateTo('results-analysis'); return; }
        if (a === 'help') { navigateTo('help'); return; }
        if (a === 'profile') { navigateTo('profile'); return; }
    }}>
            <Card className="p-4 mb-8 bg-slate-900 border-slate-800">
                <div className="flex justify-between items-center">
                    <div>
                        <h2 className="text-lg font-semibold text-white">Pre-Exam System Check</h2>
                        <p className="text-sm text-slate-400">Ensure your system is ready for a secure exam environment.</p>
                    </div>
                    <Button onClick={() => setSystemCheckOpen(true)}>Run System Check</Button>
                </div>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                    {liveExams.length > 0 && (
                        <div>
                            <h2 className="text-2xl font-bold text-white mb-4">Live Exams</h2>
                            <div className="space-y-4">
                                {liveExams.map((exam: Exam) => (
                                    <Card key={exam._id} className="p-4 flex items-center justify-between border-green-500/50 hover:border-green-500 transition-colors duration-200">
                                        <div>
                                            <div className="flex items-center space-x-3">
                                                <h3 className="font-semibold text-white">{exam.title}</h3>
                                                <Badge variant='live'>{exam.status}</Badge>
                                            </div>
                                            <p className="text-sm text-slate-400">{exam.courseCode} | Ends at {exam.endTime}</p>
                                        </div>
                                        <Button onClick={() => onStartExam(exam._id)}>Enter Exam</Button>
                                    </Card>
                                ))}
                            </div>
                        </div>
                    )}
                    <div>
                        <h2 className="text-2xl font-bold text-white mb-4">Upcoming Exams</h2>
                        <div className="space-y-4">
                            {upcomingExams.length > 0 ? upcomingExams.map((exam: Exam) => (
                                <Card key={exam._id} className="p-4 flex items-center justify-between hover:border-indigo-500 transition-colors duration-200">
                                    <div>
                                        <div className="flex items-center space-x-3">
                                            <h3 className="font-semibold text-white">{exam.title}</h3>
                                            <Badge variant={exam.status === 'Available' ? 'success' : exam.status === 'Scheduled' ? 'info' : 'warning'}>{exam.status}</Badge>
                                        </div>
                                        <p className="text-sm text-slate-400">{exam.courseCode} | {new Date(exam.scheduledDate).toLocaleDateString()} @ {exam.startTime}</p>
                                    </div>
                                        {(() => {
                                        // Prefer server-provided flags when available
                                        const completed = (exam as any).completedByUser || (exam as any).attemptForUser;
                                        const serverCanStart = (exam as any).canStartForUser;
                                        if (completed) {
                                            return <Button disabled>Completed</Button>;
                                        }
                                        if (typeof serverCanStart !== 'undefined') {
                                            return (
                                                <Button onClick={() => onStartExam(exam._id)} disabled={!serverCanStart}>
                                                    {exam.status === 'Locked' ? <Lock className="h-4 w-4 mr-2"/> : null}
                                                    {serverCanStart ? 'Start Exam' : 'View Details'}
                                                </Button>
                                            );
                                        }
                                        // Fallback to client-side scheduled window calculation
                                        const isWithinScheduledWindow = (() => {
                                            try {
                                                const date = new Date(exam.scheduledDate);
                                                const [sh, sm] = (exam.startTime || '00:00').split(':').map(Number);
                                                const [eh, em] = (exam.endTime || '23:59').split(':').map(Number);
                                                const start = new Date(date.getFullYear(), date.getMonth(), date.getDate(), sh || 0, sm || 0);
                                                const end = new Date(date.getFullYear(), date.getMonth(), date.getDate(), eh || 23, em || 59);
                                                const now = new Date();
                                                return now >= start && now <= end;
                                            } catch (e) {
                                                return false;
                                            }
                                        })();
                                        const canStart = exam.status === 'Available' || isWithinScheduledWindow;
                                        return (
                                            <Button onClick={() => onStartExam(exam._id)} disabled={!canStart}>
                                                {exam.status === 'Locked' ? <Lock className="h-4 w-4 mr-2"/> : null}
                                                {canStart ? 'Start Exam' : 'View Details'}
                                            </Button>
                                        );
                                    })()}
                                </Card>
                            )) : <p className="text-slate-400">No upcoming exams scheduled.</p>}
                        </div>
                    </div>
                </div>

                <div className="space-y-6">
                    <Card className="p-4 bg-slate-900 border-slate-800">
                        <h3 className="font-semibold mb-4 text-white">Quick Stats</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between items-center">
                                <span className="text-slate-300">Average Score</span>
                                <span className="font-bold text-2xl text-green-400">{averageScore}%</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-slate-300">Exams Completed</span>
                                <span className="font-bold text-2xl text-white">{completedExams.length}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-slate-300">Upcoming Exams</span>
                                <span className="font-bold text-2xl text-white">{upcomingExams.length + liveExams.length}</span>
                            </div>
                        </div>
                    </Card>
                    <Card className="p-4 bg-slate-900 border-slate-800">
                        <h3 className="font-semibold mb-4 text-white">Recent Results</h3>
                        <div className="space-y-3">
                            {completedExams.slice(0, 3).map((exam: Exam) => (
                                <div key={exam._id} className="flex justify-between items-center">
                                    <div>
                                        <p className="text-sm font-medium text-slate-200">{exam.title}</p>
                                        <p className="text-xs text-slate-500">{(exam as any).attemptForUser?.completedAt || exam.attempt?.completedAt}</p>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                        <Badge variant={((exam as any).attemptForUser?.score || exam.attempt?.score || 0) >= 80 ? 'success' : 'warning'}>{(exam as any).attemptForUser?.score ?? exam.attempt?.score}%</Badge>
                                        <Button variant="outline" onClick={() => openResults(exam)}>See Results</Button>
                                    </div>
                                </div>
                            ))}
                            {completedExams.length === 0 && <p className="text-slate-400">No results available yet.</p>}
                        </div>
                    </Card>
                </div>
            </div>
            
            <SystemCheckDialog open={systemCheckOpen} onOpenChange={setSystemCheckOpen} />

            <Dialog open={resultsOpen} onOpenChange={setResultsOpen} className="max-w-2xl">
                <h2 className="text-xl font-bold mb-2">Results: {resultExamTitle}</h2>
                {resultAttempt ? (
                    <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-2">
                        <div className="flex justify-between items-center p-2 bg-slate-800 rounded">
                            <span className="text-slate-300">Score</span>
                            <span className="text-white font-bold">{resultAttempt.score}%</span>
                        </div>
                        {resultAttempt.perQuestion && resultAttempt.perQuestion.map((pq: any, idx: number) => (
                            <div key={idx} className={cn('p-2 rounded border', pq.correct ? 'border-green-600 bg-green-900/10' : 'border-red-600 bg-red-900/10')}>
                                <div className="text-sm text-white font-medium">{idx + 1}. {pq.question}</div>
                                <div className="text-xs text-slate-300">Your answer: {String(pq.given)}</div>
                                <div className="text-xs text-slate-300">Correct answer: {String(pq.expected)}</div>
                                <div className="text-xs text-slate-300">Marks: {pq.marks} — {pq.correct ? 'Correct' : 'Incorrect'}</div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-slate-400">No attempt data available.</div>
                )}
                <div className="flex justify-end mt-3">
                    <Button variant="outline" onClick={() => setResultsOpen(false)}>Close</Button>
                </div>
            </Dialog>
        </DashboardLayout>
    );
};
const LecturerDashboard = ({ user, exams, onLogout, onBack, onExamChange, showToast, onUpdateUser, navigateTo, setSelectedExamIdForProctoring }: { user: UserProfile; exams: Exam[]; onLogout: () => void; onBack: () => void; onExamChange: () => void; showToast: (message: string, type: 'success' | 'error') => void; onUpdateUser: (u: UserProfile) => void; navigateTo: (state: AppState) => void; setSelectedExamIdForProctoring: (id: string) => void }) => {
    const lecturerExams = exams.filter((exam) => exam.lecturerId === user._id);
    const [createExamOpen, setCreateExamOpen] = useState(false);
    const [examToDelete, setExamToDelete] = useState<Exam | null>(null);
    const [examToEdit, setExamToEdit] = useState<Exam | null>(null);
    const [proctorOpen, setProctorOpen] = useState(false);
    const [proctorExamId, setProctorExamId] = useState<string | null>(null);
    const [adminStats, setAdminStats] = useState<{ totalStudents: number; liveExams: number; activeAlerts: number; systemUptime: string; serverUptimeHours?: number }>({ totalStudents: 0, liveExams: 0, activeAlerts: 0, systemUptime: '99.9%' });
    const [reportOpen, setReportOpen] = useState(false);
    const [reportData, setReportData] = useState<any>(null);
    const [reportExamTitle, setReportExamTitle] = useState<string>('');
    const [attemptOpen, setAttemptOpen] = useState(false);
    const [selectedAttempt, setSelectedAttempt] = useState<any | null>(null);
    const [attemptEvents, setAttemptEvents] = useState<any[]>([]);


    

    const handleDelete = async () => {
        if (!examToDelete) return;

        try {
            const res = await fetch(`${API_URL}/exams/${examToDelete._id}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('Failed to delete exam');
            
            showToast('Exam deleted successfully!', 'success');
            onExamChange(); // Re-fetch exams
        } catch (error: any) {
            showToast(error.message, 'error');
        } finally {
            setExamToDelete(null); // Close confirmation dialog
        }
    };

    const handleToggleLock = async (exam: Exam) => {
        const newStatus = exam.status === 'Locked' ? 'Scheduled' : 'Locked';
        try {
            const res = await fetch(`${API_URL}/exams/${exam._id}/status`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: newStatus })
            });
            if (!res.ok) throw new Error('Failed to update exam status');
            showToast(`Exam status updated to ${newStatus}`, 'success');
            onExamChange();
        } catch (error: any) {
            showToast(error.message, 'error');
        }
    };

    const handleChangeStatus = async (exam: Exam, newStatus: string) => {
        try {
            const res = await fetch(`${API_URL}/exams/${exam._id}/status`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: newStatus })
            });
            if (!res.ok) throw new Error('Failed to update exam status');
            showToast(`Exam status changed to ${newStatus}`, 'success');
            onExamChange();
        } catch (error: any) {
            showToast(error.message, 'error');
        }
    };


    const StatCard = ({ title, value, icon, colorClass }: { title: string, value: string | number, icon: React.ReactNode, colorClass: string }) => (
        <Card className={cn("p-4 flex items-center space-x-4", colorClass)}>
            <div className="p-3 bg-white/10 rounded-lg">{icon}</div>
            <div>
                <p className="text-sm text-slate-300">{title}</p>
                <p className="text-2xl font-bold text-white">{value}</p>
            </div>
        </Card>
    );

    // Fetch admin stats for lecturer dashboard (only if admin role)
    useEffect(() => {
        const fetchStats = async () => {
            // Skip if not admin to avoid 403 errors
            if (user.role !== 'admin') {
                return;
            }
            try {
                const res = await fetch(`${API_URL}/admin/stats`, { headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id } });
                const data = await res.json();
                if (res.ok) {
                    setAdminStats({ totalStudents: data.totalStudents || 0, liveExams: data.liveExams || 0, activeAlerts: data.activeAlerts || 0, systemUptime: data.systemUptime || '99.9%', serverUptimeHours: data.serverUptimeHours });
                }
            } catch (err) {
                console.error('Failed to load admin stats', err);
            }
        };
        fetchStats();
    }, [user._id, user.role]);

    const fetchReport = async (examId: string, examTitle?: string) => {
        try {
            const res = await fetch(`${API_URL}/exams/${examId}/report`, { headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id } });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to fetch report');
            setReportData(data);
            setReportExamTitle(examTitle || 'Exam Report');
            setReportOpen(true);
        } catch (err: any) {
            showToast(err.message || 'Failed to load report', 'error');
        }
    };

    const openAttemptDetails = async (attempt: any) => {
        setSelectedAttempt(attempt);
        setAttemptEvents([]);
        setAttemptOpen(true);
        try {
            const examId = reportData?.exam?._id || reportData?.exam?.id || '';
            if (!examId || !attempt?.userId) return;
            const res = await fetch(`${API_URL}/exams/${examId}/proctoring/${attempt.userId}`, { headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id } });
            const data = await res.json();
            if (res.ok && Array.isArray(data.events)) setAttemptEvents(data.events);
        } catch (err) {
            // ignore silently
        }
    };

    return (
    <DashboardLayout user={user} onLogout={onLogout} onBack={onBack} showToast={showToast} onUpdateUser={onUpdateUser} onAction={(a) => {
        if (a === 'overview' || a === 'dashboard') { /* default overview; no-op */ return; }
        if (a === 'create-exam') { setCreateExamOpen(true); return; }
        if (a === 'live-proctoring') { navigateTo('live-proctoring'); return; }
        if (a === 'help') { navigateTo('help'); return; }
        if (a === 'profile') { navigateTo('profile'); return; }
    }}>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard title="Total Students" value={adminStats.totalStudents} icon={<Users className="h-6 w-6 text-indigo-300"/>} colorClass="border-indigo-500/50" />
                <StatCard title="Live Exams" value={adminStats.liveExams} icon={<Monitor className="h-6 w-6 text-green-300"/>} colorClass="border-green-500/50" />
                <StatCard title="Active Alerts" value={adminStats.activeAlerts} icon={<AlertTriangle className="h-6 w-6 text-yellow-300"/>} colorClass="border-yellow-500/50" />
                <StatCard title="System Uptime" value={adminStats.systemUptime} icon={<ShieldCheck className="h-6 w-6 text-sky-300"/>} colorClass="border-sky-500/50" />
            </div>

            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-white">Your Exams</h2>
                <Button onClick={() => setCreateExamOpen(true)}><PlusCircle className="h-4 w-4 mr-2" /> Create Exam</Button>
            </div>
            <div className="space-y-4">
                {lecturerExams.length > 0 ? lecturerExams.map((exam: Exam) => (
                    <Card key={exam._id} className="p-4 grid grid-cols-5 gap-4 items-center hover:border-indigo-500 transition-colors duration-200">
                        <div className="col-span-2">
                            <p className="font-semibold text-white">{exam.title}</p>
                            <p className="text-sm text-slate-400">{exam.courseCode}</p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400">Questions</p>
                            <p className="font-semibold">{exam.questions.length}</p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400">Status</p>
                            <select
                                value={exam.status}
                                onChange={(e) => handleChangeStatus(exam, e.target.value)}
                                className={cn(
                                    'px-3 py-1 rounded-full text-xs font-semibold border-0 focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer',
                                    exam.status === 'Live' && 'bg-green-900/30 text-green-300',
                                    exam.status === 'Available' && 'bg-blue-900/30 text-blue-300',
                                    exam.status === 'Scheduled' && 'bg-slate-700 text-slate-300',
                                    exam.status === 'Locked' && 'bg-red-900/30 text-red-300'
                                )}
                            >
                                <option value="Scheduled">Scheduled</option>
                                <option value="Available">Available</option>
                                <option value="Live">Live</option>
                                <option value="Locked">Locked</option>
                            </select>
                        </div>
                            <div className="flex justify-end items-center space-x-1">
                            <Button variant="ghost" size="sm" onClick={() => { setExamToEdit(exam); setCreateExamOpen(true); }} title="Edit Exam">
                                Edit
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => navigateTo('lecturer-live-exams')} title="Live Monitoring">
                                Monitor
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => { setSelectedExamIdForProctoring(exam._id); navigateTo('lecturer-report'); }} title="View Report">
                                View Report
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => handleToggleLock(exam)} title={exam.status === 'Locked' ? 'Unlock Exam' : 'Lock Exam'}>
                                {exam.status === 'Locked' ? <Unlock className="h-4 w-4 text-green-400" /> : <Lock className="h-4 w-4 text-yellow-400"/>}
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => setExamToDelete(exam)} title="Delete Exam">
                                <Trash2 className="h-4 w-4 text-red-500" />
                            </Button>
                        </div>
                    </Card>
                )) : <p className="text-slate-400 text-center py-8">You haven't created any exams yet.</p>}
            </div>
            <CreateExamDialog open={createExamOpen} onOpenChange={(open) => { if (!open) setExamToEdit(null); setCreateExamOpen(open); }} lecturer={user} onExamCreated={() => { setExamToEdit(null); onExamChange(); }} showToast={showToast} examToEdit={examToEdit || undefined} />
            <ProctorDashboard open={proctorOpen} onOpenChange={setProctorOpen} examId={proctorExamId} user={user} />
            <Dialog open={reportOpen} onOpenChange={setReportOpen} className="max-w-4xl">
                <h2 className="text-2xl font-bold mb-2">Report: {reportExamTitle}</h2>
                <div className="space-y-3">
                    {reportData ? (
                        <div>
                            <p className="text-sm text-slate-400">Average Score: {reportData.averageScore}%</p>
                            <h3 className="font-semibold mt-3 mb-2">Per Question Stats</h3>
                            <div className="space-y-2 max-h-64 overflow-y-auto p-2">
                                {reportData.perQuestionStats && reportData.perQuestionStats.map((q: any) => (
                                    <div key={q.questionId} className="p-2 bg-slate-800 rounded">
                                        <div className="font-medium text-white">{q.question}</div>
                                        <div className="text-xs text-slate-300">Attempts: {q.attempts} — Correct: {q.correctCount} ({q.correctRatio}%)</div>
                                    </div>
                                ))}
                            </div>
                            <h3 className="font-semibold mt-4 mb-2">Student Attempts</h3>
                            <div className="space-y-2 max-h-60 overflow-y-auto p-2">
                                {reportData.attempts && reportData.attempts.length > 0 ? reportData.attempts.map((a: any, idx: number) => (
                                    <div key={idx} className="p-2 bg-slate-800 rounded flex items-center justify-between">
                                        <div>
                                            <div className="font-medium text-white">{a.userName || a.userId}</div>
                                            <div className="text-xs text-slate-300">Score: {a.score}% — Completed: {a.completedAt ? new Date(a.completedAt).toLocaleString() : 'N/A'}</div>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <Button size="sm" variant="outline" onClick={() => openAttemptDetails(a)}>View Details</Button>
                                        </div>
                                    </div>
                                )) : <div className="text-slate-400">No attempts yet.</div>}
                            </div>
                        </div>
                    ) : <div className="text-slate-400">No report data available.</div>}
                </div>
                <div className="flex justify-end mt-4">
                    <Button variant="outline" onClick={() => setReportOpen(false)}>Close</Button>
                </div>
            </Dialog>
            <Dialog open={attemptOpen} onOpenChange={setAttemptOpen} className="max-w-4xl">
                <h2 className="text-2xl font-bold mb-2">Attempt Details</h2>
                {selectedAttempt ? (
                    <div className="space-y-4">
                        <div className="bg-slate-900 p-3 rounded">
                            <div className="font-medium text-white">{selectedAttempt.userName || selectedAttempt.userId}</div>
                            <div className="text-xs text-slate-300">Score: {selectedAttempt.score}% — Completed: {selectedAttempt.completedAt ? new Date(selectedAttempt.completedAt).toLocaleString() : 'N/A'}</div>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-2">Answers</h3>
                            <div className="space-y-2 max-h-60 overflow-y-auto">
                                {selectedAttempt.perQuestion && selectedAttempt.perQuestion.map((pq: any, idx: number) => (
                                    <div key={idx} className={cn('p-2 rounded border', pq.correct ? 'border-green-600 bg-green-900/10' : 'border-red-600 bg-red-900/10')}>
                                        <div className="text-sm text-white font-medium">{idx + 1}. {pq.question}</div>
                                        <div className="text-xs text-slate-300">Answer: {String(pq.given)} — Expected: {String(pq.expected)} — Marks: {pq.marks}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-2">Proctoring Timeline</h3>
                            <div className="space-y-2 max-h-64 overflow-y-auto">
                                {attemptEvents.length === 0 && <div className="text-slate-400">No proctoring events recorded.</div>}
                                {attemptEvents.map((ev: any) => (
                                    <div key={ev._id} className="p-2 bg-slate-800 rounded">
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <div className="text-sm font-medium text-white">{ev.eventType}</div>
                                                <div className="text-xs text-slate-400">{new Date(ev.timestamp).toLocaleString()}</div>
                                                {ev.severity && (
                                                    <span className={cn(
                                                        'inline-block px-2 py-0.5 text-xs rounded mt-1',
                                                        ev.severity === 'high' && 'bg-red-600/20 text-red-400 border border-red-500/30',
                                                        ev.severity === 'medium' && 'bg-yellow-600/20 text-yellow-400 border border-yellow-500/30',
                                                        ev.severity === 'low' && 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                                                    )}>
                                                        {ev.severity.toUpperCase()}
                                                    </span>
                                                )}
                                            </div>
                                            {/* Display frame evidence (new field) or legacy snapshot */}
                                            {(ev.frameEvidence || ev.details?.snapshot) && (
                                                <div className="ml-3">
                                                    <img 
                                                        src={ev.frameEvidence || ev.details.snapshot} 
                                                        alt="evidence" 
                                                        className="w-32 h-24 object-cover rounded border border-slate-700" 
                                                    />
                                                </div>
                                            )}
                                        </div>
                                        {ev.details?.message && (
                                            <div className="text-xs text-slate-300 mt-2 italic">{ev.details.message}</div>
                                        )}
                                        <details className="text-xs mt-2">
                                            <summary className="cursor-pointer text-slate-400 hover:text-slate-300">Show raw details</summary>
                                            <pre className="text-xs mt-1 text-slate-300 bg-black/10 p-2 rounded overflow-x-auto">{JSON.stringify(ev.details, null, 2)}</pre>
                                        </details>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                ) : <div className="text-slate-400">No attempt selected.</div>}
                <div className="flex justify-end mt-4">
                    <Button variant="outline" onClick={() => setAttemptOpen(false)}>Close</Button>
                </div>
            </Dialog>
            
            <Dialog open={!!examToDelete} onOpenChange={() => setExamToDelete(null)}>
                <h2 className="text-xl font-bold text-white">Confirm Deletion</h2>
                <p className="text-slate-400 my-4">Are you sure you want to delete the exam "{examToDelete?.title}"? This action cannot be undone.</p>
                <div className="flex justify-end space-x-2">
                    <Button variant="outline" onClick={() => setExamToDelete(null)}>Cancel</Button>
                    <Button variant="destructive" onClick={handleDelete}>Delete</Button>
                </div>
            </Dialog>
        </DashboardLayout>
    );
};
const ExamScreen = ({ exam, user, onExit, showToast }: { exam: Exam; user: UserProfile; onExit: (result: ExamResult) => void; showToast: (msg: string, type: 'success'|'error') => void; }) => {
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState<{[key: string]: any}>({});
    const [timeLeft, setTimeLeft] = useState(exam.duration * 60);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [showSubmitConfirm, setShowSubmitConfirm] = useState(false);
    const [submitStatus, setSubmitStatus] = useState<'idle' | 'review' | 'sending' | 'error'>('idle');
    const [submitError, setSubmitError] = useState<string | null>(null);
    const [secureModeEnabled, setSecureModeEnabled] = useState(false);
    const [secureModeBusy, setSecureModeBusy] = useState(false);
    const [autoSubmitRequested, setAutoSubmitRequested] = useState(false);
    const [securityModalOpen, setSecurityModalOpen] = useState(false);
    const [securityTitle, setSecurityTitle] = useState<string>('Security warning');
    const [securityMessage, setSecurityMessage] = useState<string>('');
    const [securityCount, setSecurityCount] = useState<number>(0);
    const [securityMax, setSecurityMax] = useState<number>(5);
    const securityModalOpenRef = useRef(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const proctoringIntervalRef = useRef<any>(null);
    const proctoringAbortControllerRef = useRef<AbortController | null>(null); // To abort all pending requests
    const [proctoringStopped, setProctoringStopped] = useState(false);
    const [proctoringKey, setProctoringKey] = useState(0); // Used to restart proctoring
    // Note: switched to continuous MediaRecorder with timeslice; no need for manual chunks buffer.
    // const audioChunksRef = useRef<Blob[]>([]);

    const lockRef = useRef<any>(null);

    // Proctor decision / pause state (lecturer can pause/terminate)
    const [proctorDecision, setProctorDecision] = useState<{ status: 'active' | 'paused' | 'terminated'; reason?: string | null; updatedAt?: string }>(
        { status: 'active' }
    );
    const [proctorPauseOpen, setProctorPauseOpen] = useState(false);
    const pauseReasonRef = useRef<string>('');
    const proctorDecisionPollRef = useRef<number | null>(null);
    const proctorDecisionAbortRef = useRef<AbortController | null>(null);

    // Camera proctoring reliability + observability
    const proctorSchedulerRef = useRef<number | null>(null);
    const proctorInFlightRef = useRef(false);
    const lastFrameAtRef = useRef<number>(0);
    const backoffUntilRef = useRef<number>(0);
    const jitterSeedRef = useRef<number>(Math.floor(Math.random() * 100000));
    const [proctorStats, setProctorStats] = useState({
        framesSent: 0,
        uploadErrors: 0,
        lastUploadMs: 0,
        lastFrameAt: 0,
        backoff: false,
        paused: false,
        cameraOk: true,
    });

    // Strict proctoring policy knobs
    const [proctorPolicy] = useState(() => ({
        // If server reports "no face" continuously for this long, raise a violation.
        faceMissingGraceMs: 4000,
        // If camera appears covered/dark for this long, raise a violation.
        darkGraceMs: 5000,
        // How often we log repeated violations of the same type (cooldown)
        violationCooldownMs: 7000,
    }));

    const faceMissingSinceRef = useRef<number | null>(null);
    const darkSinceRef = useRef<number | null>(null);
    const lastViolationAtRef = useRef<Record<string, number>>({});

    // Adaptive encoding settings (updated based on latency/backoff)
    const encodeQualityRef = useRef<number>(0.72); // 0..1
    const encodeWidthRef = useRef<number>(320);

    // --- Screen capture (student sharing) ---
    const [screenShareEnabled, setScreenShareEnabled] = useState(false);
    const [screenShareBusy, setScreenShareBusy] = useState(false);
    const [lastScreenThumb, setLastScreenThumb] = useState<string | null>(null);
    const screenVideoRef = useRef<HTMLVideoElement | null>(null);
    const screenStreamRef = useRef<MediaStream | null>(null);
    const screenCaptureTimerRef = useRef<number | null>(null);

    const stopScreenShare = useCallback(() => {
        try {
            if (screenCaptureTimerRef.current) {
                window.clearInterval(screenCaptureTimerRef.current);
                screenCaptureTimerRef.current = null;
            }
            if (screenStreamRef.current) {
                screenStreamRef.current.getTracks().forEach(t => t.stop());
            }
        } catch {
            // ignore
        } finally {
            screenStreamRef.current = null;
            if (screenVideoRef.current) {
                try { (screenVideoRef.current as any).srcObject = null; } catch {}
            }
            setScreenShareEnabled(false);
        }
    }, []);

    const uploadEvidence = useCallback(async (dataUrl: string, meta: { evidenceType: string; violationType: string; violationScore?: number }) => {
        try {
            // Backend expects multipart/form-data with a binary file.
            const r = await fetch(dataUrl);
            const blob = await r.blob();
            const file = new File([blob], `${meta.violationType || 'evidence'}-${Date.now()}.jpg`, { type: blob.type || 'image/jpeg' });

            const form = new FormData();
            form.append('file', file);
            form.append('examId', exam._id);
            form.append('userId', user._id);
            form.append('evidenceType', meta.evidenceType);
            form.append('violationType', meta.violationType);
            form.append('violationScore', String(meta.violationScore ?? 0));

            const res = await fetch(`${API_URL}/upload-evidence`, {
                method: 'POST',
                body: form,
            });

            const j = await res.json();
            if (!res.ok) return null;
            return j;
        } catch {
            return null;
        }
    }, [exam._id, user._id]);

    const captureScreenFrame = useCallback((targetW: number = 420, quality: number = 0.7): string | null => {
        const v = screenVideoRef.current;
        if (!v || v.readyState < 2) return null;
        const srcW = v.videoWidth || 1280;
        const srcH = v.videoHeight || 720;
        const scale = targetW / srcW;
        const targetH = Math.max(1, Math.round(srcH * scale));
        const canvas = document.createElement('canvas');
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext('2d');
        try {
            ctx?.drawImage(v, 0, 0, targetW, targetH);
            return canvas.toDataURL('image/jpeg', quality);
        } catch {
            return null;
        }
    }, []);

    const startScreenShare = useCallback(async () => {
        if (screenShareEnabled || screenShareBusy) return;
        setScreenShareBusy(true);
        try {
            if (!('mediaDevices' in navigator) || !(navigator.mediaDevices as any).getDisplayMedia) {
                showToast('Screen sharing is not supported by this browser.', 'error');
                return;
            }

            // Must be called via user gesture.
            const stream = await (navigator.mediaDevices as any).getDisplayMedia({
                video: { frameRate: 5 },
                audio: false,
            });

            // keep a hidden video element to read pixels
            const v = document.createElement('video');
            v.muted = true;
            (v as any).playsInline = true;
            v.srcObject = stream;
            await v.play().catch(() => {});
            screenVideoRef.current = v;
            screenStreamRef.current = stream;

            // If user stops sharing via browser UI, reflect it
            try {
                const track = stream.getVideoTracks?.()[0];
                if (track) {
                    track.onended = () => {
                        showToast('Screen sharing stopped.', 'error');
                        stopScreenShare();
                    };
                }
            } catch {}

            setScreenShareEnabled(true);
            showToast('Screen sharing enabled. Keep sharing during the exam.', 'success');

            // Initial capture immediately (best-effort)
            const frame = captureScreenFrame();
            if (frame) {
                setLastScreenThumb(frame);
                await uploadEvidence(frame, { evidenceType: 'screenshot', violationType: 'screen_snapshot' });
            }

            // Periodic snapshots (low frequency; keeps lecturer “screen” preview fresh)
            if (screenCaptureTimerRef.current) window.clearInterval(screenCaptureTimerRef.current);
            screenCaptureTimerRef.current = window.setInterval(async () => {
                if (proctorDecision.status !== 'active') return;
                const shot = captureScreenFrame();
                if (!shot) return;
                setLastScreenThumb(shot);
                await uploadEvidence(shot, { evidenceType: 'screenshot', violationType: 'screen_snapshot' });
            }, 15000);
        } catch (err: any) {
            const msg = err?.name === 'NotAllowedError'
                ? 'Screen sharing permission denied.'
                : err?.message || 'Failed to start screen sharing.';
            showToast(msg, 'error');
            stopScreenShare();
        } finally {
            setScreenShareBusy(false);
        }
    }, [captureScreenFrame, proctorDecision.status, screenShareBusy, screenShareEnabled, showToast, stopScreenShare, uploadEvidence]);

    useEffect(() => {
        securityModalOpenRef.current = securityModalOpen;
    }, [securityModalOpen]);

    const logProctorEvent = useCallback(async (eventType: string, details: any = {}) => {
        try {
            await fetch(`${API_URL}/proctor/event`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    examId: exam._id,
                    userId: user._id,
                    eventType,
                    details: { ...details, at: new Date().toISOString() }
                })
            });
        } catch {
            // best-effort: don't block exam if logging fails
        }
    }, [exam._id, user._id]);

    const requestPauseFromClient = useCallback(async (reason: string) => {
        // Freeze the exam immediately. Lecturer can later resume.
        pauseReasonRef.current = reason;
        setProctorDecision(prev => ({ ...prev, status: 'paused', reason }));
        setProctorPauseOpen(true);
        // Stop uploads to avoid wasting bandwidth while awaiting decision.
        try { stopProctoring(); } catch {}

        // Best-effort notify backend so lecturer dashboards can see state consistently.
        try {
            await fetch(`${API_URL}/exams/${exam._id}/students/${user._id}/proctor-status`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id },
                body: JSON.stringify({ status: 'paused', reason })
            });
        } catch {
            // ignore (polling will reconcile)
        }
    }, [exam._id, user._id]);

    const canLogViolation = useCallback((type: string) => {
        const now = Date.now();
        const last = lastViolationAtRef.current[type] || 0;
        if (now - last < proctorPolicy.violationCooldownMs) return false;
        lastViolationAtRef.current[type] = now;
        return true;
    }, [proctorPolicy.violationCooldownMs]);

    const disableBrowserLock = useCallback(() => {
        try {
            lockRef.current?.disable?.();
        } catch {
            // ignore
        } finally {
            lockRef.current = null;
            setSecureModeEnabled(false);
        }
    }, []);

    const enableBrowserLock = useCallback(async () => {
        // Must be triggered by a user gesture to allow fullscreen.
        if (secureModeEnabled) return true;
        setSecureModeBusy(true);
        try {
            const lock = new (BrowserLock as any)(exam._id, user._id, API_URL);
            lock.onViolation = (title: string, message: string) => {
                // Show a professional in-app modal (instead of browser alert)
                setSecurityTitle(title || 'Security warning');
                setSecurityMessage(message || 'Security policy violation detected.');
                setSecurityCount(lock.getWarningCount?.() ?? 0);
                setSecurityMax(lock.maxWarnings ?? 5);
                setSecurityModalOpen(true);
                logProctorEvent('browser_lock_violation', { title, message, count: lock.getWarningCount?.() ?? undefined });
            };
            lock.onMaxWarnings = (count: number, details?: { title?: string; message?: string }) => {
                setSecurityTitle(details?.title || 'Final warning');
                setSecurityMessage(details?.message || 'Maximum violations reached. The exam will be submitted now.');
                setSecurityCount(count);
                setSecurityMax(lock.maxWarnings ?? 5);
                setSecurityModalOpen(true);
                logProctorEvent('browser_lock_max_warnings', { count, details });

                // Defer actual submission to an effect declared later
                setAutoSubmitRequested(true);
            };

            const ok = await lock.enable();
            if (!ok) {
                showToast('Secure mode could not be enabled. Please allow fullscreen and try again.', 'error');
                logProctorEvent('browser_lock_enable_failed', {});
                return false;
            }

            lockRef.current = lock;
            setSecureModeEnabled(true);
            showToast('Secure exam mode enabled.', 'success');
            logProctorEvent('browser_lock_enabled', {});
            setSecurityMax(lock.maxWarnings ?? 5);
            return true;
        } catch (err: any) {
            console.error('BrowserLock enable failed:', err);
            showToast(err?.message || 'Could not enable secure mode.', 'error');
            logProctorEvent('browser_lock_enable_failed', { error: String(err?.message || err) });
            return false;
        } finally {
            setSecureModeBusy(false);
        }
    }, [exam._id, logProctorEvent, secureModeEnabled, showToast, user._id]);

    // When the warning modal is acknowledged, try to re-enter fullscreen and continue.
    const handleAcknowledgeSecurityWarning = useCallback(async () => {
        setSecurityModalOpen(false);

        // If it was a final (>= max) warning, request auto-submit.
        if (securityCount >= securityMax) {
            setAutoSubmitRequested(true);
            return;
        }

        // Best-effort re-lock (fullscreen) to continue the exam neatly.
        try {
            await lockRef.current?.enterFullscreen?.();
        } catch {
            // If fullscreen fails, student can click Enable secure mode again.
        }
    }, [securityCount, securityMax]);

    // Function to stop all proctoring activities
    function stopProctoring() {
        if (proctoringStopped) return; // Already stopped
        
        console.log('[SUBMIT] Aborting all pending proctoring requests...');
        // Abort all ongoing fetch requests immediately
        if (proctoringAbortControllerRef.current) {
            proctoringAbortControllerRef.current.abort();
            proctoringAbortControllerRef.current = null;
            console.log('[SUBMIT] All pending requests aborted');
        }
        
        console.log('[SUBMIT] Stopping proctoring interval...');
        if (proctoringIntervalRef.current) {
            clearInterval(proctoringIntervalRef.current);
            proctoringIntervalRef.current = null;
        }
        
        console.log('[SUBMIT] Stopping media recorder...');
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                mediaRecorderRef.current.stop();
            } catch (e) {
                console.log('[SUBMIT] Error stopping media recorder:', e);
            }
        }
        
        console.log('[SUBMIT] Stopping audio stream tracks...');
        if (mediaRecorderRef.current && mediaRecorderRef.current.stream) {
            mediaRecorderRef.current.stream.getTracks().forEach(track => {
                track.stop();
                console.log('[SUBMIT] Stopped audio track:', track.kind);
            });
        }
        
        console.log('[SUBMIT] Stopping video stream...');
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => {
                track.stop();
                console.log('[SUBMIT] Stopped video track:', track.kind);
            });
        }
        
        setProctoringStopped(true);
        console.log('[SUBMIT] All proctoring stopped successfully');

        // Also stop screen sharing (best-effort)
        try { stopScreenShare(); } catch {}
    }

    // Handle actual submission after confirmation
    async function handleSubmit() {
        if (isSubmitting) return; // Guard against double clicks or retries while pending

        setIsSubmitting(true);
        setSubmitStatus('sending');
        setSubmitError(null);
        console.log('[SUBMIT] Starting exam submission...', { examId: exam._id, userId: user._id, answerCount: Object.keys(answers).length });
        
        // Ensure proctoring is fully stopped (in case it was restarted after cancel)
        console.log('[SUBMIT] Ensuring proctoring is completely stopped before final submission...');
        stopProctoring();
        
        console.log('[SUBMIT] Submitting immediately - requests already aborted');
        
        const SLOW_SUBMISSION_WARNING_MS = 45000;
        let warningShown = false;
        const warningTimer = window.setTimeout(() => {
            warningShown = true;
            showToast('Submission is taking longer than expected. Please stay on this page while we finish.', 'error');
        }, SLOW_SUBMISSION_WARNING_MS);

        try {
            const url = `${API_URL}/exams/${exam._id}/submit`;
            console.log('[SUBMIT] Now submitting to:', url);
            
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: user._id, answers })
            });
            
            console.log('[SUBMIT] Submit response status:', res.status);
            const data = await res.json();
            console.log('[SUBMIT] Submit response data:', data);
            
            if (!res.ok) throw new Error(data.error || 'Failed to submit exam');

            showToast('Exam submitted successfully!', 'success');
            setShowSubmitConfirm(false);
            onExit({ score: data.score, totalMarks: data.totalMarks, examTitle: exam.title, perQuestion: data.perQuestion });

        } catch (error: any) {
            console.error('[SUBMIT] Exam submission error:', error);
            let message = error?.message || 'Failed to submit exam. Please try again.';
            if (warningShown && message === 'Failed to fetch') {
                message = 'Network interruption detected during submission. Please reconnect and try again.';
            }
            showToast(message, 'error');
            setSubmitStatus('error');
            setSubmitError(message);
            setIsSubmitting(false);
        } finally {
            clearTimeout(warningTimer);
        }
    }

    useEffect(() => {
        if (!autoSubmitRequested) return;
        if (submitStatus === 'sending') return;

        // Ensure lock is released and proctoring is stopped before submit
        disableBrowserLock();
        try {
            stopProctoring();
        } catch {}

        setShowSubmitConfirm(false);
        setSubmitStatus('idle');
        setSubmitError(null);
        setAutoSubmitRequested(false);

        // Trigger the existing submission pipeline
        handleSubmit();
    }, [autoSubmitRequested, disableBrowserLock, handleSubmit, stopProctoring, submitStatus]);

    useEffect(() => {
        return () => {
            disableBrowserLock();
            try { stopScreenShare(); } catch {}
        };
    }, [disableBrowserLock, stopScreenShare]);

    // Function to restart proctoring if student cancels submission
    const restartProctoring = useCallback(() => {
        console.log('[PROCTORING] Restarting proctoring after cancel...');
        setProctoringStopped(false);
        // Increment key to force useEffect to re-run and restart proctoring
        setProctoringKey(prev => prev + 1);
    }, []);

    // Handle first submit button click - stop proctoring and show confirmation
    const handleInitialSubmitClick = useCallback(() => {
        if (submitStatus === 'sending') return; // ignore clicks while finalizing
        console.log('[SUBMIT] Initial submit button clicked');
        // Release lock so user can interact with dialogs / exit fullscreen if needed
        disableBrowserLock();
        stopProctoring();
        setSubmitError(null);
        setSubmitStatus('review');
        setShowSubmitConfirm(true);
    }, [disableBrowserLock, stopProctoring, submitStatus]);

    // Handle cancel button in confirmation dialog - RESTART PROCTORING
    const handleCancelSubmit = useCallback(() => {
        if (submitStatus === 'sending') return; // cannot cancel while submitting
        console.log('[SUBMIT] User cancelled submission - restarting proctoring');
        setShowSubmitConfirm(false);
        setSubmitStatus('idle');
        setSubmitError(null);
        restartProctoring();
        // Cancel button click counts as a user gesture; try restoring secure mode
        enableBrowserLock();
    }, [enableBrowserLock, restartProctoring, submitStatus]);

    const handleReturnAfterError = useCallback(() => {
        console.log('[SUBMIT] Returning to exam after submission error');
        setShowSubmitConfirm(false);
        setSubmitStatus('idle');
        setSubmitError(null);
        setIsSubmitting(false);
        restartProctoring();
    }, [restartProctoring]);

    const handleSubmitDialogChange = useCallback((open: boolean) => {
        if (open) {
            // State is controlled elsewhere; ignore programmatic open requests from dialog internals.
            return;
        }

        if (!showSubmitConfirm) return; // already closed

        if (submitStatus === 'sending') {
            // Ignore attempts to close while submission is in-flight; keep dialog visible.
            setShowSubmitConfirm(true);
            return;
        }

        if (submitStatus === 'error') {
            handleReturnAfterError();
        } else {
            handleCancelSubmit();
        }
    }, [handleCancelSubmit, handleReturnAfterError, showSubmitConfirm, submitStatus]);

    // Helpers for camera health + scheduling
    const clearProctorScheduler = useCallback(() => {
        if (proctorSchedulerRef.current) {
            window.clearTimeout(proctorSchedulerRef.current);
            proctorSchedulerRef.current = null;
        }
    }, []);

    const isCameraHealthy = useCallback(() => {
        const video = videoRef.current;
        if (!video) return false;
        // readyState >= HAVE_CURRENT_DATA
        if (video.readyState < 2) return false;
        if (!video.videoWidth || !video.videoHeight) return false;
        return true;
    }, []);

    const computeFrameBrightness = useCallback((dataUrl: string): number | null => {
        // Fast, approximate luminance check using a tiny downscaled canvas.
        try {
            if (!dataUrl.startsWith('data:image')) return null;
            const img = new Image();
            img.src = dataUrl;
            // WARNING: sync access to pixels requires image decode; we use a small canvas + best-effort.
            // This is acceptable at our low frame rate and helps strict realism.
            // If decode isn't ready immediately, return null and let next tick check.
            if (!img.complete) return null;
            const w = 32;
            const h = 24;
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d', { willReadFrequently: true } as any) as CanvasRenderingContext2D | null;
            if (!ctx) return null;
            ctx.drawImage(img, 0, 0, w, h);
            const pixels = ctx.getImageData(0, 0, w, h).data;
            let sum = 0;
            const n = w * h;
            for (let i = 0; i < pixels.length; i += 4) {
                // relative luminance approximation
                sum += (0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2]);
            }
            return sum / n;
        } catch {
            return null;
        }
    }, []);

    // Proctoring Loop (upgraded)
    useEffect(() => {
        if (submitStatus !== 'idle') {
            return () => {};
        }

        // If proctor has paused or terminated this student, do not run proctoring.
        if (proctorDecision.status !== 'active') {
            return () => {};
        }

        // Require secure mode before starting proctoring (realistic proctoring flow)
        if (!secureModeEnabled) {
            return () => {};
        }

        // Reset proctoring stopped state when starting a new exam
        console.log('[PROCTORING] Initializing proctoring for exam:', exam._id);
        setProctoringStopped(false);
        
        // Simple network backoff to avoid spamming server if unreachable
        const minIntervalMs = 2500; // faster than before, but adaptive
        const maxIntervalMs = 6000;
        const backoffMs = () => 10000; // 10s backoff on network failure
        const computeNextInterval = () => {
            // If a warning modal is open, pause uploads to reduce noise and avoid multiple violations.
            if (securityModalOpenRef.current) return maxIntervalMs;
            // If page isn't visible, slow down (still logs server-side, but reduce bandwidth)
            if (document.hidden) return maxIntervalMs;
            // If camera looks unhealthy, slow down.
            if (!isCameraHealthy()) return maxIntervalMs;
            // Normal operation
            return minIntervalMs;
        };

        const withJitter = (ms: number) => {
            // Add jitter so multiple clients don't synchronize (reduces burst load)
            jitterSeedRef.current = (jitterSeedRef.current * 9301 + 49297) % 233280;
            const jitter = (jitterSeedRef.current / 233280) * 400; // 0..400ms
            return Math.max(400, Math.floor(ms + jitter));
        };

        const startProctoring = async () => {
            if (submitStatus !== 'idle') {
                return;
            }
            console.log('[PROCTORING] Starting camera and audio streams...');
            
            // Reset proctoring state on server (clears reference background and violation counts)
            try {
                await fetch(`${API_URL}/proctor/reset`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ examId: exam._id, userId: user._id })
                });
                console.log('[PROCTORING] Server state reset - fresh reference image will be captured');
            } catch (err) {
                console.error('[PROCTORING] Failed to reset server state:', err);
            }
            
            // Create a new AbortController for this proctoring session
            proctoringAbortControllerRef.current = new AbortController();
            const abortSignal = proctoringAbortControllerRef.current.signal;
            console.log('[PROCTORING] Created new AbortController for request management');
            
            try {
                // Start video stream
                const videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (videoRef.current) {
                    // Ensure autoplay policies are satisfied
                    try {
                        videoRef.current.muted = true;
                        (videoRef.current as any).playsInline = true;
                    } catch (e) {}
                    videoRef.current.srcObject = videoStream;
                }

                // Track video track end events (camera unplugged / permission revoked)
                try {
                    const [videoTrack] = videoStream.getVideoTracks();
                    if (videoTrack) {
                        videoTrack.onended = () => {
                            setProctorStats(s => ({ ...s, cameraOk: false }));
                            logProctorEvent('camera_stream_ended', { kind: videoTrack.kind, label: videoTrack.label });
                        };
                    }
                } catch {}

                // Start audio stream and recorder
                const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorderRef.current = new MediaRecorder(audioStream);
                mediaRecorderRef.current.ondataavailable = async (event) => {
                    // Check if proctoring was stopped before processing
                    if (abortSignal.aborted) {
                        console.log('[PROCTORING] Audio processing skipped - proctoring stopped');
                        return;
                    }
                    
                    try {
                        const blob = event.data;
                        if (!blob || blob.size === 0) return;
                        const reader = new FileReader();
                        reader.readAsDataURL(blob);
                        reader.onloadend = async () => {
                            if (abortSignal.aborted) return; // Check again before sending
                            
                            const base64Audio = (reader.result as string) || '';
                            if (!base64Audio.includes(',')) return;
                            try {
                                const res = await fetch(`${API_URL}/proctor/audio`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ 
                                        audioData: base64Audio.split(',')[1],
                                        examId: exam._id,
                                        userId: user._id
                                    }),
                                    signal: abortSignal // Add abort signal to cancel request
                                });
                                await res.json();
                                // Audio events are now handled server-side (only voice detection triggers events)
                            } catch (error: any) { 
                                // Don't log AbortError - it's expected when stopping
                                if (error.name !== 'AbortError') {
                                    console.error('[PROCTORING] Audio error:', error);
                                }
                            }
                        };
                    } catch { /* ignore */ }
                };

                // Start adaptive scheduler (instead of fixed setInterval)
                const tick = async () => {
                    if (abortSignal.aborted) return;

                    const now = Date.now();
                    const paused = document.hidden || securityModalOpenRef.current;
                    const cameraOk = isCameraHealthy();
                    setProctorStats(s => ({ ...s, paused, cameraOk, backoff: now < backoffUntilRef.current }));

                    // Ensure recorder is running with 1s timeslice for continuous small chunks
                    try {
                        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'inactive' && !abortSignal.aborted) {
                            mediaRecorderRef.current.start(1000);
                        }
                    } catch {}

                    // If we're backing off or paused, schedule next tick and skip sending.
                    if (now < backoffUntilRef.current || paused || !cameraOk) {
                        const interval = withJitter(computeNextInterval());
                        proctorSchedulerRef.current = window.setTimeout(tick, interval);
                        return;
                    }

                    // Backpressure: only one in-flight frame upload at a time.
                    if (proctorInFlightRef.current) {
                        const interval = withJitter(computeNextInterval());
                        proctorSchedulerRef.current = window.setTimeout(tick, interval);
                        return;
                    }

                    const imageDataUrl = captureFrame();
                    if (!imageDataUrl) {
                        // camera might be warming up; try later
                        const interval = withJitter(computeNextInterval());
                        proctorSchedulerRef.current = window.setTimeout(tick, interval);
                        return;
                    }

                    // Strict rule: covered/dark camera detection (client-side)
                    const bright = computeFrameBrightness(imageDataUrl);
                    const isDark = typeof bright === 'number' ? bright < 35 : false; // threshold tuned for JPEG baseline
                    if (isDark) {
                        if (!darkSinceRef.current) darkSinceRef.current = Date.now();
                    } else {
                        darkSinceRef.current = null;
                    }
                    if (darkSinceRef.current && Date.now() - darkSinceRef.current >= proctorPolicy.darkGraceMs) {
                        if (canLogViolation('camera_dark')) {
                            logProctorEvent('camera_dark_or_covered', {
                                message: 'Camera feed appears dark/covered for an extended period.',
                                brightness: bright,
                                graceMs: proctorPolicy.darkGraceMs,
                                frameEvidence: imageDataUrl,
                                severity: 'high'
                            });

                            // High severity: pause immediately until lecturer decides
                            requestPauseFromClient('Camera appears covered/dark for too long. Waiting for invigilator approval.');
                        }
                    }

                    proctorInFlightRef.current = true;
                    lastFrameAtRef.current = now;
                    setProctorStats(s => ({ ...s, lastFrameAt: now }));

                    const t0 = performance.now();
                    try {
                        const res = await fetch(`${API_URL}/proctor`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                imageDataUrl,
                                userId: user._id,
                                examId: exam._id,
                                examActive: submitStatus === 'idle'
                            }),
                            signal: abortSignal
                        });

                        const t1 = performance.now();
                        const uploadMs = Math.round(t1 - t0);
                        setProctorStats(s => ({ ...s, lastUploadMs: uploadMs }));

                        // Adaptive encoding: if uploads are slow, reduce size/quality; if fast, slowly recover.
                        if (uploadMs > 1200) {
                            encodeQualityRef.current = Math.max(0.45, +(encodeQualityRef.current - 0.05).toFixed(2));
                            encodeWidthRef.current = Math.max(240, encodeWidthRef.current - 40);
                        } else if (uploadMs < 450) {
                            encodeQualityRef.current = Math.min(0.8, +(encodeQualityRef.current + 0.02).toFixed(2));
                            encodeWidthRef.current = Math.min(360, encodeWidthRef.current + 20);
                        }

                        if (!res.ok) {
                            backoffUntilRef.current = Date.now() + backoffMs();
                            setProctorStats(s => ({ ...s, uploadErrors: s.uploadErrors + 1, backoff: true }));
                            proctorInFlightRef.current = false;
                            const interval = maxIntervalMs;
                            proctorSchedulerRef.current = window.setTimeout(tick, interval);
                            return;
                        }

                        const data = await res.json();
                        if (abortSignal.aborted) return;

                        setProctorStats(s => ({ ...s, framesSent: s.framesSent + 1 }));

                        if (data && data.error) {
                            console.error('Proctoring error:', data.error);
                        } else {
                            // Strict rule: face missing escalation using server output
                            const faceCount = typeof data?.faceCount === 'number' ? data.faceCount : null;
                            if (faceCount === 0) {
                                if (!faceMissingSinceRef.current) faceMissingSinceRef.current = Date.now();
                            } else {
                                faceMissingSinceRef.current = null;
                            }

                            if (faceMissingSinceRef.current && Date.now() - faceMissingSinceRef.current >= proctorPolicy.faceMissingGraceMs) {
                                if (canLogViolation('face_missing')) {
                                    logProctorEvent('face_missing', {
                                        message: 'No face detected for an extended period.',
                                        graceMs: proctorPolicy.faceMissingGraceMs,
                                        faceCount,
                                        frameEvidence: imageDataUrl,
                                        severity: 'high'
                                    });

                                    // High severity: pause immediately until lecturer decides
                                    requestPauseFromClient('Face not detected for too long. Exam is paused pending invigilator decision.');
                                }
                            }

                            // Strict: multiple faces (client reinforces server detection)
                            if (typeof faceCount === 'number' && faceCount > 1) {
                                if (canLogViolation('multiple_faces')) {
                                    logProctorEvent('multiple_faces_detected', {
                                        message: `Multiple faces detected (${faceCount}).`,
                                        faceCount,
                                        frameEvidence: imageDataUrl,
                                        severity: 'high'
                                    });

                                    // High severity: pause immediately until lecturer decides
                                    requestPauseFromClient('Multiple faces detected. Exam is paused pending invigilator decision.');
                                }
                            }
                        }
                    } catch (err: any) {
                        if (err?.name !== 'AbortError') {
                            console.error('Image proctoring error:', err);
                            backoffUntilRef.current = Date.now() + backoffMs();
                            setProctorStats(s => ({ ...s, uploadErrors: s.uploadErrors + 1, backoff: true }));

                            // When in backoff, drop quality aggressively to recover quickly.
                            encodeQualityRef.current = Math.max(0.4, +(encodeQualityRef.current - 0.08).toFixed(2));
                            encodeWidthRef.current = Math.max(220, encodeWidthRef.current - 60);
                        }
                    } finally {
                        proctorInFlightRef.current = false;
                        if (!abortSignal.aborted) {
                            const interval = withJitter(computeNextInterval());
                            proctorSchedulerRef.current = window.setTimeout(tick, interval);
                        }
                    }
                };

                // Kick off first tick
                clearProctorScheduler();
                proctorSchedulerRef.current = window.setTimeout(tick, withJitter(computeNextInterval()));

                console.log('[PROCTORING] Proctoring started successfully - adaptive scheduler running');
            } catch (error) {
                console.error("[PROCTORING] Failed to start proctoring streams:", error);
                showToast("Could not start camera or microphone for proctoring.", "error");
            }
        };

        startProctoring();

        // Cleanup function
        return () => {
            console.log('[PROCTORING] Cleanup: stopping all proctoring activities');

            clearProctorScheduler();
            proctorInFlightRef.current = false;
            
            // Abort all pending requests
            if (proctoringAbortControllerRef.current) {
                proctoringAbortControllerRef.current.abort();
                proctoringAbortControllerRef.current = null;
            }
            
            if (proctoringIntervalRef.current) {
                clearInterval(proctoringIntervalRef.current);
                proctoringIntervalRef.current = null;
            }
            if (videoRef.current && videoRef.current.srcObject) {
                (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
            }
            if (mediaRecorderRef.current && mediaRecorderRef.current.stream) {
                mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [clearProctorScheduler, exam._id, isCameraHealthy, proctorDecision.status, requestPauseFromClient, secureModeEnabled, showToast, submitStatus, user._id, proctoringKey]);

    // Poll server for lecturer decisions (pause/resume/terminate)
    useEffect(() => {
        // Stop polling if exam is submitting/exited
        if (submitStatus !== 'idle') return;

        // Clear any existing poll timers
        if (proctorDecisionPollRef.current) {
            window.clearInterval(proctorDecisionPollRef.current);
            proctorDecisionPollRef.current = null;
        }

        // Abort any in-flight status calls
        if (proctorDecisionAbortRef.current) {
            proctorDecisionAbortRef.current.abort();
            proctorDecisionAbortRef.current = null;
        }

        const poll = async () => {
            try {
                const ac = new AbortController();
                proctorDecisionAbortRef.current = ac;
                const res = await fetch(`${API_URL}/exams/${exam._id}/students/${user._id}/proctor-status`, {
                    headers: { 'X-User-Id': user._id },
                    signal: ac.signal
                });
                if (!res.ok) return;
                const data = await res.json();
                const status = data?.status?.status as ('active' | 'paused' | 'terminated') | undefined;
                const reason = data?.status?.reason as string | undefined;

                if (!status) return;

                setProctorDecision(prev => {
                    if (prev.status === status && (prev.reason || '') === (reason || '')) return prev;
                    return { status, reason, updatedAt: data?.status?.updatedAt };
                });

                if (status === 'paused') {
                    pauseReasonRef.current = reason || pauseReasonRef.current || 'Paused by invigilator.';
                    setProctorPauseOpen(true);
                    try { stopProctoring(); } catch {}
                    try { stopScreenShare(); } catch {}
                }

                if (status === 'terminated') {
                    // Force submission immediately
                    showToast(reason || 'Your exam was terminated by the invigilator.', 'error');
                    try { disableBrowserLock(); } catch {}
                    try { stopProctoring(); } catch {}
                    try { stopScreenShare(); } catch {}
                    // Trigger final submit
                    handleSubmit();
                }

                if (status === 'active') {
                    setProctorPauseOpen(false);
                    // Resume proctoring if we were stopped due to pause
                    if (proctoringStopped) {
                        setProctoringStopped(false);
                        setProctoringKey(k => k + 1);
                    }
                }
            } catch (e: any) {
                if (e?.name === 'AbortError') return;
            }
        };

        // initial sync, then interval polling
        poll();
        proctorDecisionPollRef.current = window.setInterval(poll, 2000);

        return () => {
            if (proctorDecisionPollRef.current) {
                window.clearInterval(proctorDecisionPollRef.current);
                proctorDecisionPollRef.current = null;
            }
            if (proctorDecisionAbortRef.current) {
                proctorDecisionAbortRef.current.abort();
                proctorDecisionAbortRef.current = null;
            }
        };
    }, [disableBrowserLock, exam._id, handleSubmit, proctoringStopped, showToast, stopProctoring, submitStatus, user._id]);

    const captureFrame = (): string | null => {
        const video = videoRef.current;
        if (!video || video.readyState < 3) return null;
        // Downscale to reduce bandwidth and backend load, keep aspect ratio
        const srcW = video.videoWidth || 640;
        const srcH = video.videoHeight || 480;
        const targetW = encodeWidthRef.current; // adaptive width
        const scale = targetW / srcW;
        const targetH = Math.max(1, Math.round(srcH * scale));

        const canvas = document.createElement('canvas');
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext('2d');
        try {
            ctx?.drawImage(video, 0, 0, targetW, targetH);
            // Adaptive quality reduces payload when network is slow
            return canvas.toDataURL('image/jpeg', encodeQualityRef.current);
        } catch (e) {
            return null;
        }
    };

    // Timer logic
    useEffect(() => {
        // When paused/terminated, freeze timer.
        if (proctorDecision.status !== 'active') {
            return;
        }
        if (timeLeft <= 0) {
            handleSubmit();
            return;
        }
        const timerId = setInterval(() => {
            setTimeLeft(t => t - 1);
        }, 1000);
        return () => clearInterval(timerId);
    }, [proctorDecision.status, timeLeft, handleSubmit]);
    
    // Tab switch monitoring
    useEffect(() => {
        const handleVisibilityChange = () => {
            if (document.hidden) {
                showToast("Tab switch detected. Please remain on the exam page.", 'error');
                // Record as a proctor event for lecturer timeline
                try {
                    fetch(`${API_URL}/proctor/event`, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ examId: exam._id, userId: user._id, eventType: 'tab_switch', details: { hidden: true, at: new Date().toISOString() } })
                    });
                } catch {}
            }
        };
        document.addEventListener("visibilitychange", handleVisibilityChange);
        return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
    }, [showToast]);

    const handleAnswerChange = (questionId: string, answer: any) => {
        setAnswers(prev => ({...prev, [questionId]: answer}));
    };
    
    const formatTime = (seconds: number) => {
        const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
        const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toString().padStart(2, '0');
        return `${h}:${m}:${s}`;
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex h-screen bg-slate-900">
            <div className="w-72 border-r border-slate-800 p-4 flex flex-col">
                <h2 className="text-lg font-semibold text-white mb-2">{exam.title}</h2>
                <p className="text-sm text-slate-400 mb-4">{exam.questions.length} Questions</p>
                <div className="grid grid-cols-5 gap-2 overflow-y-auto flex-1">
                    {exam.questions.map((q, index) => (
                        <button key={q._id} onClick={() => setCurrentQuestion(index)} className={cn(
                            "h-10 w-10 rounded-md flex items-center justify-center font-medium transition-colors duration-200",
                            index === currentQuestion ? "bg-indigo-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700",
                            answers[q._id] !== undefined ? "border-2 border-green-500" : ""
                        )}>
                            {index + 1}
                        </button>
                    ))}
                </div>
                <div className="mt-auto space-y-2">
                    <div className="text-xs text-slate-400 text-center mb-2">
                        {Object.keys(answers).length} of {exam.questions.length} answered
                    </div>
                    <Button variant="destructive" className="w-full" disabled={isSubmitting} onClick={handleInitialSubmitClick}>
                        {isSubmitting ? 'Submitting...' : 'Submit Exam'}
                    </Button>
                </div>
            </div>
            <main className="flex-1 flex flex-col">
                <header className="flex justify-between items-center p-4 border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
                    <div>
                        <p className="text-slate-300">Question {currentQuestion + 1} of {exam.questions.length}</p>
                    </div>
                    <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2 text-yellow-400">
                            <AlertTriangle className="h-5 w-5" />
                            <span className="text-sm font-medium">Proctoring Active</span>
                        </div>
                        {/* Optional connectivity indicator when proctor uploads are degraded */}
                        <div className="hidden md:flex items-center space-x-2">
                            {/* Using a simple heuristic: if we recently backed off uploads, show badge */}
                            {/* Note: this badge updates on next render cycles driven by other state; lightweight UX hint only */}
                            {/* In a more advanced setup, lift degraded state into React state for deterministic updates */}
                            <span className="text-xs px-2 py-1 rounded bg-yellow-500/20 text-yellow-300 border border-yellow-600/30">Network adapting…</span>
                        </div>
                        <div className="flex items-center space-x-2">
                            <Timer className="h-5 w-5 text-slate-400"/>
                            <span className={cn("font-mono text-lg", timeLeft < 600 ? "text-red-400" : "text-white")}>{formatTime(timeLeft)}</span>
                        </div>
                    </div>
                </header>
                <div className="flex-1 p-8 overflow-y-auto relative">
                    {/* Proctor pause overlay (blocks the exam until lecturer decision) */}
                    {proctorPauseOpen && (
                        <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
                            <div className="w-full max-w-lg rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl">
                                <div className="flex items-start gap-3">
                                    <div className="mt-0.5 rounded-full bg-yellow-500/20 p-2 text-yellow-300">
                                        <AlertTriangle className="h-5 w-5" />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-lg font-semibold text-white">Exam paused</h3>
                                        <p className="mt-1 text-sm text-slate-300">
                                            {proctorDecision.reason || pauseReasonRef.current || 'Paused by invigilator. Please wait for a decision.'}
                                        </p>
                                        <div className="mt-4 rounded-lg border border-slate-700 bg-slate-800/70 p-3">
                                            <p className="text-xs text-slate-300">Your answers are safe. Don’t refresh the page.</p>
                                            <p className="text-xs text-slate-400 mt-1">We’ll automatically resume when the invigilator allows you to continue.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    <Dialog open={securityModalOpen} onOpenChange={(open) => {
                        // prevent closing by clicking outside; only the OK/Submit button closes it
                        if (!open) {
                            setSecurityModalOpen(true);
                            return;
                        }
                        setSecurityModalOpen(open);
                    }}>
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            <AlertTriangle className="h-5 w-5 text-yellow-400" />
                            {securityTitle}
                        </h2>
                        <p className="text-slate-300 mt-3">{securityMessage}</p>
                        <div className="mt-4 flex items-center justify-between">
                            <div className="text-sm text-slate-400">
                                Warning <span className="text-slate-200 font-semibold">{securityCount}</span>/{securityMax}
                            </div>
                            <Button onClick={handleAcknowledgeSecurityWarning} className={securityCount >= securityMax ? 'bg-red-600 hover:bg-red-500' : ''}>
                                {securityCount >= securityMax ? 'Submit exam' : 'OK'}
                            </Button>
                        </div>
                        <p className="text-xs text-slate-500 mt-3">
                            After you click OK, secure mode will be restored automatically and you can continue from where you stopped.
                        </p>
                    </Dialog>

                    {!secureModeEnabled && (
                        <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-6">
                            <div className="w-full max-w-lg rounded-xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl">
                                <h3 className="text-xl font-semibold text-white mb-2">Start Secure Exam Mode</h3>
                                <p className="text-slate-300 text-sm mb-4">
                                    This exam requires fullscreen and proctoring restrictions. Click below to enable secure mode.
                                </p>
                                <ul className="text-slate-400 text-sm space-y-1 mb-5 list-disc pl-5">
                                    <li>Stay in fullscreen during the exam</li>
                                    <li>Tab switching and focus changes are recorded</li>
                                    <li>DevTools shortcuts and copy/paste are blocked where possible</li>
                                </ul>
                                <div className="flex items-center gap-3">
                                    <Button className="flex-1" onClick={enableBrowserLock} isLoading={secureModeBusy}>
                                        Enable Secure Mode
                                    </Button>
                                    <Button
                                        variant="outline"
                                        className="flex-1"
                                        onClick={() => showToast('Secure mode is required to start the exam.', 'error')}
                                        disabled={secureModeBusy}
                                    >
                                        Not now
                                    </Button>
                                </div>
                                <p className="text-xs text-slate-500 mt-4">
                                    Note: Browsers can’t fully block OS-level shortcuts. These controls are best-effort and all suspicious activity is logged.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Screen-share control (must be a user gesture) */}
                    {secureModeEnabled && (
                        <div className="absolute top-44 right-4 z-10 w-48">
                            <div className="rounded-lg border border-slate-700 bg-slate-900/70 backdrop-blur p-3">
                                <div className="flex items-center justify-between">
                                    <div className="text-[11px] font-semibold text-slate-200">Screen</div>
                                    <div className={cn('text-[10px] px-2 py-0.5 rounded-full border', screenShareEnabled ? 'bg-emerald-500/10 text-emerald-200 border-emerald-500/20' : 'bg-slate-800/50 text-slate-300 border-slate-700')}>
                                        {screenShareEnabled ? 'Sharing' : 'Off'}
                                    </div>
                                </div>
                                {lastScreenThumb ? (
                                    <img src={lastScreenThumb} alt="Screen snapshot" className="mt-2 h-20 w-full object-cover rounded border border-slate-700" />
                                ) : (
                                    <div className="mt-2 h-20 w-full rounded border border-slate-700 bg-slate-800/40 flex items-center justify-center text-[10px] text-slate-400">
                                        No preview yet
                                    </div>
                                )}
                                <div className="mt-2 flex gap-2">
                                    {!screenShareEnabled ? (
                                        <Button size="sm" className="flex-1" onClick={startScreenShare} isLoading={screenShareBusy}>
                                            Share
                                        </Button>
                                    ) : (
                                        <Button size="sm" variant="outline" className="flex-1" onClick={stopScreenShare}>
                                            Stop
                                        </Button>
                                    )}
                                </div>
                                <div className="mt-2 text-[10px] text-slate-500">
                                    Tip: choose “Entire Screen” for best evidence quality.
                                </div>
                            </div>
                        </div>
                    )}
                    <video ref={videoRef} autoPlay playsInline muted className="absolute top-4 right-4 w-48 h-36 rounded-md object-cover border-2 border-slate-700"></video>
                    {secureModeEnabled && (
                        <div className="absolute top-4 right-56 z-10 rounded-lg border border-slate-700 bg-slate-900/70 backdrop-blur px-3 py-2 text-xs text-slate-200">
                            <div className="flex items-center gap-2">
                                <span className={cn('h-2 w-2 rounded-full', proctorStats.cameraOk ? 'bg-green-400' : 'bg-red-400')} />
                                <span className="font-semibold">Proctor</span>
                                {proctorStats.paused && <span className="text-yellow-300">Paused</span>}
                                {proctorStats.backoff && <span className="text-yellow-300">Backoff</span>}
                            </div>
                            <div className="mt-1 text-slate-300">
                                Frames: {proctorStats.framesSent} · Errors: {proctorStats.uploadErrors}
                            </div>
                            <div className="mt-0.5 text-slate-400">
                                Last upload: {proctorStats.lastUploadMs}ms · Q: {Math.round(encodeQualityRef.current * 100)} · W: {encodeWidthRef.current}px
                            </div>
                        </div>
                    )}
                    <div className="max-w-4xl mx-auto">
                        {exam.questions[currentQuestion] ? (
                            <QuestionRenderer 
                                question={exam.questions[currentQuestion]} 
                                onAnswer={(answer) => handleAnswerChange(exam.questions[currentQuestion]._id, answer)}
                                savedAnswer={answers[exam.questions[currentQuestion]._id]}
                            />
                        ) : (
                            <h2 className="text-xl font-semibold mb-6 text-slate-100">No question to display.</h2>
                        )}
                    </div>
                </div>
                <footer className="p-4 border-t border-slate-800 flex justify-between">
                    <Button variant="outline" disabled={currentQuestion === 0} onClick={() => setCurrentQuestion(p => p - 1)}>Previous</Button>
                    <Button disabled={!exam.questions || currentQuestion === exam.questions.length - 1} onClick={() => setCurrentQuestion(p => p + 1)}>Next</Button>
                </footer>
            </main>
            
            {/* Submit Confirmation Dialog */}
            <Dialog open={showSubmitConfirm} onOpenChange={handleSubmitDialogChange}>
                <div className="text-center p-6">
                    <AlertTriangle className="h-16 w-16 text-yellow-400 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-white mb-2">Submit Exam?</h2>
                    <p className="text-slate-400 mb-4">
                        Are you sure you want to submit your exam? This action cannot be undone.
                    </p>
                    <div className="bg-slate-800/50 rounded-lg p-4 mb-6">
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-slate-400">Questions Answered:</span>
                            <span className="text-white font-semibold">{Object.keys(answers).length} / {exam.questions.length}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-400">Time Remaining:</span>
                            <span className="text-white font-semibold">{formatTime(timeLeft)}</span>
                        </div>
                    </div>
                    {Object.keys(answers).length < exam.questions.length && submitStatus !== 'sending' && (
                        <div className="bg-yellow-500/20 text-yellow-300 p-3 rounded-lg mb-4 text-sm">
                            ⚠️ You have unanswered questions. They will be marked as incorrect.
                        </div>
                    )}

                    {submitStatus === 'sending' && (
                        <div className="bg-blue-500/10 text-blue-300 p-3 rounded-lg mb-4 text-sm flex items-center justify-center space-x-2">
                            <span className="animate-pulse">Submitting your exam... Please wait.</span>
                        </div>
                    )}

                    {submitStatus === 'error' && submitError && (
                        <div className="bg-red-500/10 text-red-300 p-3 rounded-lg mb-4 text-sm">
                            <p className="font-semibold mb-1">Submission failed</p>
                            <p>{submitError}</p>
                        </div>
                    )}

                    <div className="flex space-x-3">
                        <Button
                            variant="outline"
                            className="flex-1"
                            onClick={submitStatus === 'error' ? handleReturnAfterError : handleCancelSubmit}
                            disabled={submitStatus === 'sending'}
                        >
                            {submitStatus === 'error' ? 'Return to Exam' : 'Cancel'}
                        </Button>
                        <Button
                            variant="destructive"
                            className="flex-1"
                            onClick={handleSubmit}
                            disabled={submitStatus === 'sending'}
                        >
                            {submitStatus === 'sending' ? 'Submitting...' : submitStatus === 'error' ? 'Try Again' : 'Submit Now'}
                        </Button>
                    </div>
                </div>
            </Dialog>
        </motion.div>
    );
};
const ResultScreen = ({ result, onDone }: { result: ExamResult, onDone: () => void }) => {
    const totalQuestions = result.perQuestion?.length ?? 0;
    const correctCount = (result.perQuestion ?? []).filter(q => q.correct).length;
    const wrongCount = Math.max(0, totalQuestions - correctCount);

    const score = Number.isFinite(Number(result.score)) ? Number(result.score) : 0;
    const scoreTone = score >= 85 ? 'emerald' : score >= 60 ? 'sky' : 'amber';
    const scoreRing = scoreTone === 'emerald'
        ? 'from-emerald-400/30 via-emerald-400/10 to-transparent'
        : scoreTone === 'sky'
            ? 'from-sky-400/30 via-sky-400/10 to-transparent'
            : 'from-amber-400/30 via-amber-400/10 to-transparent';
    const scoreText = scoreTone === 'emerald' ? 'text-emerald-300' : scoreTone === 'sky' ? 'text-sky-300' : 'text-amber-300';

    return (
        <motion.div
            className="min-h-screen w-full bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
        >
            {/* Top bar */}
            <div className="sticky top-0 z-10 border-b border-slate-800/70 bg-slate-950/80 backdrop-blur">
                <div className="mx-auto max-w-7xl px-4 py-4 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 min-w-0">
                        <div className="h-10 w-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                            <CheckCircle className="h-6 w-6 text-emerald-300" />
                        </div>
                        <div className="min-w-0">
                            <div className="text-slate-200 font-semibold truncate">Exam submitted</div>
                            <div className="text-slate-400 text-sm truncate">{result.examTitle}</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button onClick={onDone} className="px-4">Back to Dashboard</Button>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="mx-auto max-w-7xl px-4 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                    {/* Hero score */}
                    <Card className="lg:col-span-5 p-6 sm:p-8 relative overflow-hidden">
                        <div className={cn('pointer-events-none absolute inset-0 bg-gradient-to-br', scoreRing)} />
                        <div className="relative">
                            <div className="flex items-start justify-between gap-4">
                                <div>
                                    <div className="text-slate-200 text-xl font-bold">Your result</div>
                                    <div className="text-slate-400 text-sm mt-1">You’ve completed the exam successfully.</div>
                                </div>
                                <div className={cn('px-3 py-1 rounded-full text-xs font-semibold border',
                                    scoreTone === 'emerald'
                                        ? 'bg-emerald-500/10 text-emerald-200 border-emerald-500/20'
                                        : scoreTone === 'sky'
                                            ? 'bg-sky-500/10 text-sky-200 border-sky-500/20'
                                            : 'bg-amber-500/10 text-amber-200 border-amber-500/20'
                                )}>
                                    Final Score
                                </div>
                            </div>

                            <div className="mt-8 flex items-end gap-4">
                                <div className={cn('text-7xl sm:text-8xl font-extrabold leading-none tracking-tight', scoreText)}>
                                    {Math.round(score)}
                                </div>
                                <div className="pb-2 text-slate-300 font-semibold text-xl">%</div>
                            </div>

                            <div className="mt-6 grid grid-cols-3 gap-3">
                                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                                    <div className="text-slate-400 text-xs">Questions</div>
                                    <div className="text-slate-100 font-bold text-lg mt-1">{totalQuestions}</div>
                                </div>
                                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                                    <div className="text-slate-400 text-xs">Correct</div>
                                    <div className="text-emerald-200 font-bold text-lg mt-1">{correctCount}</div>
                                </div>
                                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                                    <div className="text-slate-400 text-xs">Incorrect</div>
                                    <div className="text-rose-200 font-bold text-lg mt-1">{wrongCount}</div>
                                </div>
                            </div>

                            <div className="mt-6 text-sm text-slate-400">
                                Tip: You can review your answers below. If you were paused/terminated by a proctor, the event timeline remains available in the proctoring logs.
                            </div>
                        </div>
                    </Card>

                    {/* Breakdown */}
                    <Card className="lg:col-span-7 p-6 sm:p-8">
                        <div className="flex items-center justify-between gap-4">
                            <div>
                                <div className="text-slate-200 text-lg font-bold">Question breakdown</div>
                                <div className="text-slate-400 text-sm mt-1">Detailed per-question evaluation.</div>
                            </div>
                            <div className="text-slate-400 text-xs">
                                {totalQuestions > 0 ? `${correctCount}/${totalQuestions} correct` : 'No questions found'}
                            </div>
                        </div>

                        {result.perQuestion && result.perQuestion.length > 0 ? (
                            <div className="mt-5 rounded-xl border border-slate-800 overflow-hidden">
                                <div className="max-h-[60vh] overflow-y-auto">
                                    {result.perQuestion.map((q, idx) => (
                                        <div
                                            key={idx}
                                            className={cn(
                                                'px-4 sm:px-5 py-4 border-b border-slate-800 last:border-b-0',
                                                q.correct ? 'bg-emerald-500/5' : 'bg-rose-500/5'
                                            )}
                                        >
                                            <div className="flex items-start justify-between gap-4">
                                                <div className="min-w-0">
                                                    <div className="font-semibold text-slate-100 text-sm">
                                                        {idx + 1}. {q.question}
                                                    </div>
                                                    <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                                                        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-2">
                                                            <div className="text-slate-400">Your answer</div>
                                                            <div className="text-slate-200 mt-0.5 break-words">{String(q.given)}</div>
                                                        </div>
                                                        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-2">
                                                            <div className="text-slate-400">Correct answer</div>
                                                            <div className="text-slate-200 mt-0.5 break-words">{String(q.expected)}</div>
                                                        </div>
                                                    </div>
                                                    <div className="mt-2 text-xs text-slate-400">Marks: {q.marks}</div>
                                                </div>
                                                <div className={cn(
                                                    'shrink-0 px-3 py-1 rounded-full text-xs font-semibold border',
                                                    q.correct
                                                        ? 'bg-emerald-500/10 text-emerald-200 border-emerald-500/20'
                                                        : 'bg-rose-500/10 text-rose-200 border-rose-500/20'
                                                )}>
                                                    {q.correct ? 'Correct' : 'Incorrect'}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/30 p-6 text-slate-400">
                                No per-question breakdown was provided for this exam.
                            </div>
                        )}
                    </Card>
                </div>
            </div>
        </motion.div>
    );
};
const SystemCheckDialog = ({ open, onOpenChange }: { open: boolean, onOpenChange: (open: boolean) => void }) => {
    const checks = [
        { name: 'Camera', icon: <Video className="h-8 w-8 text-green-400"/>, status: 'Working' },
        { name: 'Microphone', icon: <Mic className="h-8 w-8 text-green-400"/>, status: 'Working' },
        { name: 'Internet', icon: <Wifi className="h-8 w-8 text-green-400"/>, status: 'Stable' },
        { name: 'Browser', icon: <Globe className="h-8 w-8 text-green-400"/>, status: 'Compatible' },
    ];
    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <div className="text-center">
                <h2 className="text-2xl font-bold mb-2 text-white">System Check</h2>
                <p className="text-slate-400 mb-6">We're checking if your system is ready for the exam.</p>
                <div className="grid grid-cols-2 gap-4 mb-6">
                    {checks.map(check => (
                        <div key={check.name} className="bg-slate-800/50 p-4 rounded-lg">
                            {check.icon}
                            <p className="font-semibold mt-2 text-slate-200">{check.name}</p>
                            <p className="text-sm text-green-400">{check.status}</p>
                        </div>
                    ))}
                </div>
                <div className="bg-green-500/20 text-green-300 p-3 rounded-lg flex items-center justify-center space-x-2 mb-6">
                    <ShieldCheck className="h-5 w-5"/>
                    <span className="font-medium text-sm">All Systems Ready</span>
                </div>
                <Button onClick={() => onOpenChange(false)} className="w-full">Close</Button>
            </div>
        </Dialog>
    );
};
const CreateExamDialog = ({ open, onOpenChange, lecturer, onExamCreated, showToast, examToEdit }: { open: boolean, onOpenChange: (open: boolean) => void; lecturer: UserProfile, onExamCreated: () => void, showToast: (message: string, type: 'success' | 'error') => void; examToEdit?: Exam }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [department, setDepartment] = useState('');
    const [questions, setQuestions] = useState<NewQuestion[]>([]);
    const [showQuestionForm, setShowQuestionForm] = useState(false);
    const [showAIGenerator, setShowAIGenerator] = useState(false);

    const handleAddQuestion = (question: NewQuestion) => {
        setQuestions(prev => [...prev, question]);
    };

    const handleAddMultipleQuestions = (newQuestions: NewQuestion[]) => {
        setQuestions(prev => [...prev, ...newQuestions]);
    };

    const handleRemoveQuestion = (index: number) => {
        setQuestions(prev => prev.filter((_, i) => i !== index));
    };
    
    useEffect(() => {
        if (open && examToEdit) {
            // prefill form values when editing
            setDepartment(examToEdit.department || '');
            setQuestions(examToEdit.questions || []);
            // we will populate the form fields via DOM when the dialog renders using defaultValue
        } else if (!open) {
            // clear when closed
            setDepartment('');
            setQuestions([]);
        }
    }, [open, examToEdit]);

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setIsLoading(true);

        const formData = new FormData(e.currentTarget);
        
        const examDetails = {
            title: formData.get('title'),
            courseCode: formData.get('courseCode'),
            description: formData.get('description'),
            scheduledDate: formData.get('scheduledDate'),
            startTime: formData.get('startTime'),
            endTime: formData.get('endTime'),
            duration: Number(formData.get('duration')),
            institution: lecturer.institution,
            department: department,
            targetYear: formData.get('targetYear'),
            lecturerId: lecturer._id,
            lecturerName: lecturer.name,
            questions: questions,
        };
        
        try {
            let res;
            if (examToEdit) {
                // Update existing exam
                res = await fetch(`${API_URL}/exams/${examToEdit._id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...examDetails })
                });
            } else {
                res = await fetch(`${API_URL}/exams`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(examDetails)
                });
            }

            const data = await res.json();
            if (!res.ok) throw new Error(data.error || (examToEdit ? 'Failed to update exam' : 'Failed to create exam'));

            showToast(examToEdit ? 'Exam updated successfully!' : 'Exam created successfully!', 'success');
            try { e.currentTarget?.reset?.(); } catch (err) {}
            setQuestions([]);
            onExamCreated();
            onOpenChange(false);

        } catch (error: any) {
            showToast(error.message, 'error');
        } finally {
            setIsLoading(false);
        }
    };
    
    return (
        <Dialog open={open} onOpenChange={onOpenChange} className="max-w-3xl">
            <h2 className="text-2xl font-bold mb-4 text-white">Create New Exam</h2>
            <form onSubmit={handleSubmit} className="space-y-4 max-h-[80vh] overflow-y-auto pr-2">
                <div className="grid grid-cols-2 gap-4">
                    <div><Label htmlFor="exam-title">Exam Title</Label><Input id="exam-title" name="title" required defaultValue={examToEdit?.title || ''} /></div>
                    <div><Label htmlFor="course-code">Course Code</Label><Input id="course-code" name="courseCode" required defaultValue={examToEdit?.courseCode || ''} /></div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                        <Label>Institution</Label>
                        <Input value={lecturer.institution} disabled />
                    </div>
                    <div className="space-y-1">
                        <Label htmlFor="department-create">Department</Label>
                        <Select id="department-create" name="department" value={department || examToEdit?.department || ''} onChange={(e: ChangeEvent<HTMLSelectElement>) => setDepartment(e.target.value)} required>
                            <option value="">Select Department</option>
                            {INSTITUTIONS[lecturer.institution]?.map(dept => <option key={dept} value={dept}>{dept}</option>)}
                        </Select>
                    </div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                    <div><Label htmlFor="exam-date">Exam Date</Label><Input id="exam-date" name="scheduledDate" type="date" required defaultValue={examToEdit?.scheduledDate ? new Date(examToEdit.scheduledDate).toISOString().slice(0,10) : ''} /></div>
                    <div><Label htmlFor="start-time">Start Time</Label><Input id="start-time" name="startTime" type="time" required defaultValue={examToEdit?.startTime || ''} /></div>
                    <div><Label htmlFor="duration">Duration (mins)</Label><Input id="duration" name="duration" type="number" required defaultValue={examToEdit?.duration ? String(examToEdit.duration) : ''} /></div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div><Label htmlFor="targetYear">Target Year</Label><Input id="targetYear" name="targetYear" placeholder="e.g., 3" required defaultValue={examToEdit?.targetYear || ''} /></div>
                </div>
                <div><Label htmlFor="description">Description</Label><textarea id="description" name="description" rows={2} className="w-full rounded-md border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm" required defaultValue={examToEdit?.description || ''}></textarea></div>
                
                <div>
                    <h3 className="text-lg font-semibold mb-2 text-white">Questions ({questions.length})</h3>
                    <div className="space-y-2 max-h-48 overflow-y-auto p-2 border border-slate-700 rounded-md">
                        {questions.map((q, index) => (
                            <div key={index} className="bg-slate-800 p-2 rounded-md flex justify-between items-center">
                                <p className="text-sm truncate flex-1">{index + 1}. {q.question}</p>
                                <Button type="button" variant="ghost" size="sm" onClick={() => handleRemoveQuestion(index)}><Trash2 className="h-4 w-4 text-red-500"/></Button>
                            </div>
                        ))}
                        {questions.length === 0 && <p className="text-sm text-slate-500 text-center">No questions added yet.</p>}
                    </div>
                    <div className="flex space-x-2 mt-2">
                        <Button type="button" variant="secondary" className="flex-1" onClick={() => setShowAIGenerator(true)}><Cpu className="h-4 w-4 mr-2"/> Generate with AI</Button>
                        <Button type="button" variant="secondary" className="flex-1" onClick={() => setShowQuestionForm(true)}><PlusCircle className="h-4 w-4 mr-2"/> Type Manually</Button>
                    </div>
                    
                    <AnimatePresence>
                        {showQuestionForm && (
                            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                                <AddQuestionForm onAddQuestion={handleAddQuestion} onDone={() => setShowQuestionForm(false)} />
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                <div className="pt-4 flex justify-end space-x-2">
                    <Button variant="outline" type="button" onClick={() => onOpenChange(false)}>Cancel</Button>
                    <Button type="submit" isLoading={isLoading}>Create Exam</Button>
                </div>
            </form>
             <AIGenerateQuestionsDialog 
                open={showAIGenerator} 
                onOpenChange={setShowAIGenerator} 
                onAddQuestions={handleAddMultipleQuestions}
                showToast={showToast}
            />
        </Dialog>
    );
};
const AddQuestionForm = ({ onAddQuestion, onDone }: { onAddQuestion: (q: NewQuestion) => void; onDone: () => void; }) => {
    const [questionType, setQuestionType] = useState<QuestionType>('multiple-choice');
    const [questionText, setQuestionText] = useState('');
    const [options, setOptions] = useState(['', '', '', '']);
    const [correctAnswer, setCorrectAnswer] = useState('');
    const [marks, setMarks] = useState(1);

    const handleOptionChange = (index: number, value: string) => {
        const newOptions = [...options];
        newOptions[index] = value;
        setOptions(newOptions);
    };

    const handleAdd = () => {
        let questionToAdd: NewQuestion;
        switch (questionType) {
            case 'multiple-choice':
                questionToAdd = { type: 'multiple-choice', question: questionText, options, correctAnswer: Number(correctAnswer), marks };
                break;
            case 'true-false':
                questionToAdd = { type: 'true-false', question: questionText, correctAnswer: correctAnswer === 'true', marks };
                break;
            default:
                questionToAdd = { type: 'short-answer', question: questionText, correctAnswer, marks };
                break;
        }
        onAddQuestion(questionToAdd);
        setQuestionText('');
        setOptions(['', '', '', '']);
        setCorrectAnswer('');
        onDone();
    };

    return (
        <div className="p-4 mt-4 space-y-3 bg-slate-800/50 rounded-lg border border-slate-700">
            <h4 className="font-semibold text-white">New Question</h4>
            <div className="grid grid-cols-3 gap-4">
                <div className="col-span-2 space-y-1">
                    <Label htmlFor="q-type">Type</Label>
                    <Select id="q-type" value={questionType} onChange={(e: ChangeEvent<HTMLSelectElement>) => setQuestionType(e.target.value as QuestionType)}>
                        <option value="multiple-choice">Multiple Choice</option>
                        <option value="true-false">True/False</option>
                        <option value="short-answer">Short Answer</option>
                        <option value="essay">Essay</option>
                    </Select>
                </div>
                 <div className="space-y-1">
                    <Label htmlFor="q-marks">Marks</Label>
                    <Input id="q-marks" type="number" value={marks} onChange={(e: ChangeEvent<HTMLInputElement>) => setMarks(Number(e.target.value))} />
                </div>
            </div>
            <div className="space-y-1">
                <Label htmlFor="q-text">Question Text</Label>
                <textarea id="q-text" rows={2} value={questionText} onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setQuestionText(e.target.value)} className="w-full rounded-md border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm" />
            </div>
            {questionType === 'multiple-choice' && (
                <div className="space-y-2">
                    <Label>Options</Label>
                    {options.map((opt, i) => (
                        <Input key={i} placeholder={`Option ${i + 1}`} value={opt} onChange={(e: ChangeEvent<HTMLInputElement>) => handleOptionChange(i, e.target.value)} />
                    ))}
                    <Label htmlFor="q-correct-mc">Correct Option Number</Label>
                    <Input id="q-correct-mc" type="number" min="1" max="4" value={correctAnswer} onChange={(e: ChangeEvent<HTMLInputElement>) => setCorrectAnswer(e.target.value)} />
                </div>
            )}
             {questionType === 'true-false' && (
                <div>
                    <Label htmlFor="q-correct-tf">Correct Answer</Label>
                    <Select id="q-correct-tf" value={correctAnswer} onChange={(e: ChangeEvent<HTMLSelectElement>) => setCorrectAnswer(e.target.value)}>
                        <option value="" disabled>Select Answer</option>
                        <option value="true">True</option>
                        <option value="false">False</option>
                    </Select>
                </div>
            )}
             {(questionType === 'short-answer' || questionType === 'essay') && (
                <div>
                    <Label htmlFor="q-correct-text">Correct Answer / Keywords</Label>
                    <Input id="q-correct-text" value={correctAnswer} onChange={(e: ChangeEvent<HTMLInputElement>) => setCorrectAnswer(e.target.value)} />
                </div>
            )}
            <div className="flex justify-end space-x-2 pt-2">
                <Button type="button" variant="outline" onClick={onDone}>Cancel</Button>
                <Button type="button" onClick={handleAdd}>Add to Exam</Button>
            </div>
        </div>
    );
};
const AIGenerateQuestionsDialog = ({ open, onOpenChange, onAddQuestions, showToast }: { open: boolean, onOpenChange: (open: boolean) => void; onAddQuestions: (questions: NewQuestion[]) => void; showToast: (message: string, type: 'success' | 'error') => void; }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [generatedQuestions, setGeneratedQuestions] = useState<NewQuestion[]>([]);

    const handleGenerate = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setIsLoading(true);
        setGeneratedQuestions([]);
        
        const formData = new FormData(e.currentTarget);
        const generationParams = {
            topic: formData.get('topic'),
            difficulty: formData.get('difficulty'),
            num_questions: Number(formData.get('num_questions')),
            question_type: formData.get('question_type'),
        };

        try {
            const res = await fetch(`${API_URL}/ai-generate-questions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(generationParams)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to generate questions');
            
            setGeneratedQuestions(data.questions);
            showToast('Questions generated successfully!', 'success');
        } catch (error: any) {
            showToast(error.message, 'error');
        } finally {
            setIsLoading(false);
        }
    };

    const handleAdd = () => {
        onAddQuestions(generatedQuestions);
        onOpenChange(false);
        setGeneratedQuestions([]);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange} className="max-w-2xl">
            <h2 className="text-2xl font-bold mb-4 text-white">Generate Questions with AI</h2>
            <form onSubmit={handleGenerate} className="space-y-4">
                <div className="space-y-1">
                    <Label htmlFor="ai-topic">Topic</Label>
                    <Input id="ai-topic" name="topic" placeholder="e.g., Python Loops, World War II" required/>
                </div>
                <div className="grid grid-cols-3 gap-4">
                     <div className="space-y-1">
                        <Label htmlFor="ai-num">Number of Questions</Label>
                        <Input id="ai-num" name="num_questions" type="number" defaultValue={5} required/>
                    </div>
                     <div className="space-y-1">
                        <Label htmlFor="ai-difficulty">Difficulty</Label>
                        <Select id="ai-difficulty" name="difficulty" defaultValue="Medium" required>
                            <option>Easy</option>
                            <option>Medium</option>
                            <option>Hard</option>
                        </Select>
                    </div>
                     <div className="space-y-1">
                        <Label htmlFor="ai-q-type">Question Type</Label>
                        <Select id="ai-q-type" name="question_type" defaultValue="multiple-choice" required>
                            <option value="multiple-choice">Multiple Choice</option>
                            <option value="true-false">True/False</option>
                            <option value="short-answer">Short Answer</option>
                        </Select>
                    </div>
                </div>
                <Button type="submit" isLoading={isLoading} className="w-full">Generate</Button>
            </form>

            {generatedQuestions.length > 0 && (
                <div className="mt-6">
                    <h3 className="font-semibold text-white mb-2">Review Generated Questions</h3>
                    <div className="space-y-2 max-h-40 overflow-y-auto p-2 border border-slate-700 rounded-md">
                         {generatedQuestions.map((q, index) => (
                            <div key={index} className="bg-slate-800 p-2 rounded-md">
                                <p className="text-sm font-medium">{index + 1}. {q.question}</p>
                                <p className="text-xs text-green-400 mt-1">Correct Answer: {String(q.correctAnswer)}</p>
                            </div>
                        ))}
                    </div>
                    <Button onClick={handleAdd} className="w-full mt-4">Add these questions to Exam</Button>
                </div>
            )}
        </Dialog>
    );
};

// --- Live Proctoring Page ---
const LiveProctoring = ({ user, onBack }: { user: UserProfile; onBack: () => void }) => {
    // Student view (keep the existing simple recent list)
    const [events, setEvents] = React.useState<any[]>([]);

    // Lecturer view (new grid + alert queue)
    const [examId, setExamId] = React.useState<string>('');
    const [summary, setSummary] = React.useState<{ userId: string; name: string; count: number; lastEvent: any; countsByType?: any }[]>([]);
    const [latestByUser, setLatestByUser] = React.useState<Record<string, any>>({});
    const [decisionByUser, setDecisionByUser] = React.useState<Record<string, { status: 'active'|'paused'|'terminated'; reason?: string }>>({});
    const [page, setPage] = React.useState(0);
    const pageSize = 15; // 5 columns x 3 rows

    // alert queue: holds unresolved high/critical events without collisions
    const [alertQueue, setAlertQueue] = React.useState<any[]>([]);
    const [activeAlert, setActiveAlert] = React.useState<any | null>(null);
    const alertSeenRef = React.useRef<Record<string, number>>({});

    const socketRef = React.useRef<Socket | null>(null);
    const pollRef = React.useRef<number | null>(null);

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical':
            case 'high': return 'text-red-300 bg-red-500/10 border-red-500/40';
            case 'medium': return 'text-orange-300 bg-orange-500/10 border-orange-500/40';
            case 'warning': return 'text-yellow-300 bg-yellow-500/10 border-yellow-500/40';
            default: return 'text-blue-300 bg-blue-500/10 border-blue-500/40';
        }
    };

    const normalizeEventKey = (ev: any) => {
        const ts = ev?.timestamp || ev?.time || '';
        const u = ev?.userId || ev?.user || 'unknown';
        const ex = ev?.examId || 'unknown';
        const type = ev?.eventType || ev?.violationType || 'event';
        // If two events arrive same ms, this still collides; but we also keep a TTL that still results in safe dedupe behavior.
        return `${ex}:${u}:${type}:${ts}`;
    };

    const enqueueAlert = React.useCallback((ev: any) => {
        const sev = String(ev?.severity || ev?.details?.severity || 'info').toLowerCase();
        if (!(sev === 'high' || sev === 'critical')) return;

        const k = normalizeEventKey(ev);
        const now = Date.now();
        const last = alertSeenRef.current[k] || 0;
        if (now - last < 10_000) return; // dedupe window
        alertSeenRef.current[k] = now;

        setAlertQueue((q) => {
            // Prevent endless growth: cap at 100
            const next = [ev, ...q];
            return next.slice(0, 100);
        });
    }, []);

    // Show next alert if none currently active
    React.useEffect(() => {
        if (activeAlert) return;
        if (alertQueue.length === 0) return;
        setActiveAlert(alertQueue[0]);
        setAlertQueue(q => q.slice(1));
    }, [activeAlert, alertQueue]);

    const refreshSummary = React.useCallback(async () => {
        if (!examId) return;
        try {
            const res = await fetch(`${API_URL}/exams/${examId}/proctoring`, { headers: { 'X-User-Id': user._id } });
            const data = await res.json();
            if (!res.ok) return;
            const s = Array.isArray(data?.summary) ? data.summary : [];
            setSummary(s);
            setLatestByUser(prev => {
                const next = { ...prev };
                for (const row of s) {
                    if (row?.userId && row?.lastEvent) next[row.userId] = row.lastEvent;
                }
                return next;
            });
        } catch {}
    }, [examId, user._id]);

    const refreshDecisions = React.useCallback(async (userIds: string[]) => {
        if (!examId) return;
        try {
            const results = await Promise.all(userIds.map(async (uid) => {
                try {
                    const r = await fetch(`${API_URL}/exams/${examId}/students/${uid}/proctor-status`, { headers: { 'X-User-Id': user._id } });
                    const j = await r.json();
                    if (!r.ok) return null;
                    return { uid, status: j?.status?.status as any, reason: j?.status?.reason as any };
                } catch {
                    return null;
                }
            }));
            setDecisionByUser(prev => {
                const next = { ...prev };
                for (const it of results) {
                    if (!it?.uid || !it?.status) continue;
                    next[it.uid] = { status: it.status, reason: it.reason };
                }
                return next;
            });
        } catch {}
    }, [examId, user._id]);

    // Student polling (unchanged behavior)
    React.useEffect(() => {
        if (user.role !== 'student') return;
        let timer: number | null = null;
        const fetchEvents = async () => {
            try {
                const endpoint = `${API_URL}/proctoring/recent?userId=${user._id}&limit=50`;
                const res = await fetch(endpoint, { headers: { 'X-User-Id': user._id } });
                const data = await res.json();
                if (res.ok && data.events) setEvents(data.events);
            } catch {}
        };
        fetchEvents();
        timer = window.setInterval(fetchEvents, 2000);
        return () => { if (timer) window.clearInterval(timer); };
    }, [user]);

    // Lecturer: poll summary + connect socket
    React.useEffect(() => {
        if (user.role !== 'lecturer') return;
        if (!examId) return;

        refreshSummary();
        if (pollRef.current) window.clearInterval(pollRef.current);
        pollRef.current = window.setInterval(() => {
            refreshSummary();
        }, 2500);

        // SocketIO connect for instant alerts
        try {
            if (socketRef.current) {
                socketRef.current.disconnect();
                socketRef.current = null;
            }
            // Extract base URL from API_URL (remove /api suffix)
            const baseUrl = API_URL.replace(/\/api$/, '');
            const s = io(`${baseUrl}/proctor`, {
                transports: ['websocket'],
            });
            socketRef.current = s;

            s.on('connect', () => {
                s.emit('join_exam', { examId });
            });
            s.on('violation_detected', (payload: any) => {
                // Shape it like a normal proctor event for display
                const ev = {
                    _id: payload?.id,
                    examId: payload?.examId,
                    userId: payload?.userId,
                    eventType: payload?.violationType,
                    severity: payload?.severity,
                    details: {
                        score: payload?.score,
                        message: payload?.message,
                        snapshot: payload?.snapshot || payload?.details?.snapshot,
                        screen: payload?.screen || payload?.details?.screen,
                    },
                    timestamp: payload?.timestamp,
                };
                setLatestByUser(prev => ({ ...prev, [String(ev.userId || 'unknown')]: ev }));
                enqueueAlert(ev);
            });
            s.on('proctor_decision', (payload: any) => {
                const uid = String(payload?.userId || '');
                if (!uid) return;
                setDecisionByUser(prev => ({ ...prev, [uid]: { status: payload?.status, reason: payload?.reason } }));
            });
        } catch {
            // keep polling-only fallback
        }

        return () => {
            if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
            if (socketRef.current) {
                socketRef.current.disconnect();
                socketRef.current = null;
            }
        };
    }, [enqueueAlert, examId, refreshSummary, user._id, user.role]);

    React.useEffect(() => {
        if (user.role !== 'lecturer') return;
        const ids = summary.map(s => s.userId).filter(Boolean);
        if (ids.length === 0) return;
        refreshDecisions(ids.slice(0, 50));
    }, [refreshDecisions, summary, user.role]);

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="min-h-screen w-full bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900"
        >
            <div className="sticky top-0 z-10 border-b border-slate-800/70 bg-slate-950/80 backdrop-blur">
                <div className="mx-auto max-w-7xl px-4 py-4 flex items-center justify-between gap-4">
                    <div className="min-w-0">
                        <h2 className="text-xl sm:text-2xl font-bold text-slate-100 truncate">Live Proctoring Monitor</h2>
                        <p className="text-xs sm:text-sm text-slate-400 truncate">Real-time alerts + invigilator decisions (pause / allow / stop).</p>
                    </div>
                    <Button variant="outline" onClick={onBack}>Back to Dashboard</Button>
                </div>
            </div>

            <div className="mx-auto max-w-7xl px-4 py-6">

            {user.role === 'student' ? (
                <div className="grid grid-cols-1 gap-6">
                    <Card className="p-6 bg-slate-900 border-slate-800">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-semibold text-white">Recent Proctoring Events</h3>
                            <div className="flex items-center space-x-2">
                                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
                                <span className="text-sm text-slate-400">Live Updates</span>
                            </div>
                        </div>
                        <div className="space-y-3 max-h-[600px] overflow-y-auto">
                            {events.length > 0 ? events.map((evt, idx) => (
                                <div key={evt._id || idx} className={cn('p-4 rounded-lg border', getSeverityColor(evt.severity || 'info'))}>
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2 mb-1">
                                                <span className="font-semibold text-white capitalize">
                                                    {(evt.eventType || 'event').replace(/_/g, ' ')}
                                                </span>
                                                <Badge variant={evt.severity === 'high' ? 'danger' : evt.severity === 'medium' ? 'warning' : 'info'}>
                                                    {evt.severity || 'info'}
                                                </Badge>
                                            </div>
                                            <p className="text-xs text-slate-500">{new Date(evt.timestamp).toLocaleString()}</p>
                                        </div>
                                        {evt.details?.snapshot && (
                                            <img src={evt.details.snapshot} alt="Snapshot" className="h-16 w-20 object-cover rounded ml-3" />
                                        )}
                                    </div>
                                </div>
                            )) : (
                                <div className="text-center py-12 text-slate-400">
                                    <AlertTriangle className="h-12 w-12 mx-auto mb-3 text-slate-600" />
                                    <p>No recent proctoring events</p>
                                </div>
                            )}
                        </div>
                    </Card>
                </div>
            ) : (
                <div className="space-y-6">
                    <Card className="p-5 bg-slate-900 border-slate-800">
                        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                            <div className="flex-1">
                                <h3 className="text-lg font-semibold text-white">Select exam to monitor</h3>
                                <p className="text-sm text-slate-400">Enter an exam ID, then you’ll see a live 5×3 grid and an action queue for high-severity alerts.</p>
                            </div>
                            <div className="flex gap-2">
                                <Input value={examId} onChange={(e: any) => setExamId(e.target.value)} placeholder="Exam ID" className="w-80" />
                                <Button onClick={() => { setPage(0); refreshSummary(); }}>Load</Button>
                            </div>
                        </div>
                    </Card>

                    {/* Action-required alert modal (one at a time, no collisions) */}
                    <Dialog open={!!activeAlert} onOpenChange={(open) => { if (!open) setActiveAlert(null); }}>
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            <AlertTriangle className="h-5 w-5 text-red-400" />
                            Action required
                        </h2>
                        {activeAlert && (
                            <div className="mt-3 space-y-3">
                                <div className={cn('rounded-lg border p-3', getSeverityColor(String(activeAlert.severity || 'high')))}>
                                    <div className="text-sm text-slate-200 font-semibold">
                                        {(activeAlert.eventType || 'violation').replace(/_/g, ' ')}
                                    </div>
                                    <div className="mt-1 text-xs text-slate-300">Student: {activeAlert.userId} • Exam: {activeAlert.examId}</div>
                                    <div className="mt-2 text-sm text-slate-200">{activeAlert?.details?.message || activeAlert?.message || 'High severity event detected.'}</div>
                                </div>

                                {(activeAlert?.details?.snapshot || activeAlert?.details?.screen) && (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        {activeAlert?.details?.snapshot && (
                                            <div>
                                                <div className="text-[11px] text-slate-400 mb-1">Camera</div>
                                                <img src={activeAlert.details.snapshot} alt="Camera evidence" className="w-full max-h-64 object-cover rounded border border-slate-700" />
                                            </div>
                                        )}
                                        {activeAlert?.details?.screen && (
                                            <div>
                                                <div className="text-[11px] text-slate-400 mb-1">Screen</div>
                                                <img src={activeAlert.details.screen} alt="Screen evidence" className="w-full max-h-64 object-cover rounded border border-slate-700" />
                                            </div>
                                        )}
                                    </div>
                                )}

                                <div className="flex flex-col md:flex-row gap-2 md:justify-end">
                                    <Button variant="outline" onClick={async () => {
                                        // keep paused
                                        try {
                                            await fetch(`${API_URL}/exams/${activeAlert.examId}/students/${activeAlert.userId}/proctor-status`, {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id },
                                                body: JSON.stringify({ status: 'paused', reason: 'Paused for review by invigilator.' })
                                            });
                                        } catch {}
                                        setActiveAlert(null);
                                    }}>Keep paused</Button>
                                    <Button onClick={async () => {
                                        try {
                                            await fetch(`${API_URL}/exams/${activeAlert.examId}/students/${activeAlert.userId}/proctor-status`, {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id },
                                                body: JSON.stringify({ status: 'active', reason: 'Allowed to continue.' })
                                            });
                                        } catch {}
                                        setActiveAlert(null);
                                    }}>Allow continue</Button>
                                    <Button variant="destructive" onClick={async () => {
                                        try {
                                            await fetch(`${API_URL}/exams/${activeAlert.examId}/students/${activeAlert.userId}/proctor-status`, {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id },
                                                body: JSON.stringify({ status: 'terminated', reason: 'Terminated by invigilator due to high severity violation.' })
                                            });
                                        } catch {}
                                        setActiveAlert(null);
                                    }}>Terminate exam</Button>
                                </div>
                            </div>
                        )}
                    </Dialog>

                    <Card className="p-6 bg-slate-900 border-slate-800">
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <h3 className="text-xl font-semibold text-white">Live proctoring grid</h3>
                                <p className="text-sm text-slate-400">Showing {pageSize} students per page (5 columns × 3 rows).</p>
                            </div>
                            <div className="flex items-center gap-2">
                                <Button variant="outline" onClick={() => setPage(p => Math.max(0, p - 1))}>Prev</Button>
                                <Button variant="outline" onClick={() => setPage(p => p + 1)}>Next</Button>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
                            {summary.slice(page * pageSize, page * pageSize + pageSize).map((s) => {
                                const ev = latestByUser[s.userId] || s.lastEvent;
                                const sev = String(ev?.severity || 'info');
                                const decision = decisionByUser[s.userId]?.status || 'active';
                                const decisionBadge = decision === 'paused' ? 'warning' : decision === 'terminated' ? 'danger' : 'info';
                                return (
                                    <div key={s.userId} className="rounded-xl border border-slate-800 bg-slate-950/30 overflow-hidden">
                                        <div className="p-3 border-b border-slate-800 flex items-center justify-between">
                                            <div className="min-w-0">
                                                <div className="text-sm font-semibold text-white truncate">{s.name || s.userId}</div>
                                                <div className="text-[11px] text-slate-400 truncate">{s.userId}</div>
                                            </div>
                                            <div className="flex flex-col items-end gap-1">
                                                <Badge variant={decisionBadge as any}>{decision}</Badge>
                                                <Badge variant={sev === 'high' ? 'danger' : sev === 'medium' ? 'warning' : 'info'}>{sev}</Badge>
                                            </div>
                                        </div>

                                        <div className="bg-slate-900">
                                            <div className="aspect-video">
                                                {ev?.details?.snapshot ? (
                                                    <img src={ev.details.snapshot} alt="Student camera" className="h-full w-full object-cover" />
                                                ) : (
                                                    <div className="h-full w-full flex items-center justify-center text-slate-500 text-xs">Camera preview not available</div>
                                                )}
                                            </div>
                                            <div className="p-2 border-t border-slate-800">
                                                <div className="text-[10px] text-slate-400 mb-1">Screen</div>
                                                {ev?.details?.screen ? (
                                                    <img src={ev.details.screen} alt="Student screen" className="h-16 w-full object-cover rounded border border-slate-700" />
                                                ) : (
                                                    <div className="h-16 w-full rounded border border-slate-700 bg-slate-800/40 flex items-center justify-center text-[10px] text-slate-500">
                                                        Screen preview not available
                                                    </div>
                                                )}
                                            </div>
                                        </div>

                                        <div className="p-3">
                                            <div className={cn('rounded-lg border p-2', getSeverityColor(sev))}>
                                                <div className="text-xs font-semibold text-white truncate">{(ev?.eventType || 'No events').replace(/_/g, ' ')}</div>
                                                <div className="mt-1 text-[11px] text-slate-300 line-clamp-2">
                                                    {ev?.details?.message || ev?.message || '—'}
                                                </div>
                                            </div>

                                            <div className="mt-3 grid grid-cols-3 gap-2">
                                                <Button variant="outline" size="sm" onClick={async () => {
                                                    try {
                                                        await fetch(`${API_URL}/exams/${examId}/students/${s.userId}/proctor-status`, {
                                                            method: 'POST',
                                                            headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id },
                                                            body: JSON.stringify({ status: 'paused', reason: 'Paused by invigilator.' })
                                                        });
                                                    } catch {}
                                                }}>Pause</Button>
                                                <Button size="sm" onClick={async () => {
                                                    try {
                                                        await fetch(`${API_URL}/exams/${examId}/students/${s.userId}/proctor-status`, {
                                                            method: 'POST',
                                                            headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id },
                                                            body: JSON.stringify({ status: 'active', reason: 'Allowed to continue.' })
                                                        });
                                                    } catch {}
                                                }}>Allow</Button>
                                                <Button variant="destructive" size="sm" onClick={async () => {
                                                    try {
                                                        await fetch(`${API_URL}/exams/${examId}/students/${s.userId}/proctor-status`, {
                                                            method: 'POST',
                                                            headers: { 'Content-Type': 'application/json', 'X-User-Id': user._id },
                                                            body: JSON.stringify({ status: 'terminated', reason: 'Terminated by invigilator.' })
                                                        });
                                                    } catch {}
                                                }}>Stop</Button>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}

                            {examId && summary.length === 0 && (
                                <div className="md:col-span-5 text-center py-12 text-slate-400">
                                    <Users className="h-10 w-10 mx-auto mb-3 text-slate-600" />
                                    <p>No students/events yet.</p>
                                    <p className="text-xs text-slate-500 mt-1">This grid populates when proctoring events start arriving for the exam.</p>
                                </div>
                            )}
                        </div>
                    </Card>
                </div>
            )}
            </div>
        </motion.div>
    );
};

const ProctorDashboard = ({ open, onOpenChange, examId, user }: { open: boolean; onOpenChange: (b: boolean) => void; examId?: string | null; user?: UserProfile }) => {
    const [summary, setSummary] = useState<{ userId: string; name: string; count: number; lastEvent: any; countsByType?: any }[]>([]);
    const [selectedUser, setSelectedUser] = useState<string | null>(null);
    const [details, setDetails] = useState<any[]>([]);
    const lastTimestampRef = useRef<string | null>(null);
    const pollRef = useRef<number | null>(null);
    const [newCounts, setNewCounts] = useState<Record<string, number>>({});
    const clearTimersRef = useRef<Record<string, number>>({});

    useEffect(() => {
        if (!open || !examId) return;
        fetch(`${API_URL}/exams/${examId}/proctoring`, { headers: { 'Content-Type': 'application/json', 'X-User-Id': user?._id || '' } }).then(r => r.json()).then(data => {
            if (data && data.summary) setSummary(data.summary);
            // initialize lastTimestamp to now so polling fetches only new events
            lastTimestampRef.current = new Date().toISOString();
            // start polling for recent events
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = window.setInterval(async () => {
                try {
                    const since = lastTimestampRef.current || new Date(0).toISOString();
                    const res = await fetch(`${API_URL}/exams/${examId}/proctoring/recent?since=${encodeURIComponent(since)}`, { headers: { 'Content-Type': 'application/json', 'X-User-Id': user?._id || '' } });
                    if (!res.ok) return;
                    const recent = await res.json();
                    if (!recent || !Array.isArray(recent.events) || recent.events.length === 0) return;
                    // update lastTimestamp to newest event timestamp
                    const newest = recent.events.reduce((acc: string, ev: any) => ev.timestamp > acc ? ev.timestamp : acc, lastTimestampRef.current || '1970-01-01T00:00:00.000Z');
                    lastTimestampRef.current = newest;
                    // apply events to summary and details
                    setSummary(prev => {
                        const copy = [...prev];
                        for (const ev of recent.events) {
                            const userId = ev.userId || ev.user || 'unknown';
                            const idx = copy.findIndex(s => s.userId === userId);
                            if (idx >= 0) {
                                copy[idx] = { ...copy[idx], count: (copy[idx].count || 0) + 1, lastEvent: ev };
                            } else {
                                copy.push({ userId, name: ev.userName || userId, count: 1, lastEvent: ev });
                            }
                        }
                        return copy;
                    });
                    // compute per-user new event counts and set temporary badges/highlights
                    const perUserCounts: Record<string, number> = {};
                    for (const ev of recent.events) {
                        const userId = ev.userId || ev.user || 'unknown';
                        perUserCounts[userId] = (perUserCounts[userId] || 0) + 1;
                    }
                    setNewCounts(prev => {
                        const copy = { ...prev };
                        for (const uid of Object.keys(perUserCounts)) {
                            copy[uid] = (copy[uid] || 0) + perUserCounts[uid];
                            // reset any existing timer
                            if (clearTimersRef.current[uid]) { window.clearTimeout(clearTimersRef.current[uid]); }
                            // auto-clear the new badge after 6s
                            // capture uid for closure
                            const timerId = window.setTimeout(() => {
                                setNewCounts(curr => {
                                    const c2 = { ...curr };
                                    delete c2[uid];
                                    return c2;
                                });
                                delete clearTimersRef.current[uid];
                            }, 6000);
                            clearTimersRef.current[uid] = timerId;
                        }
                        return copy;
                    });
                    // if currently viewing one user's details, append new events for that user
                    if (selectedUser) {
                        const userEvents = recent.events.filter((e: any) => (e.userId || e.user) === selectedUser);
                        if (userEvents.length > 0) setDetails(prev => [...userEvents.map((ev: any) => ev), ...prev]);
                    }
                } catch (err) {
                    // ignore polling errors silently (optionally log)
                    // console.error('Polling proctor recent failed', err);
                }
            }, 1000); // Poll lecturer proctoring endpoint every 1s for near-real-time updates
        }).catch(err => console.error('Failed to load proctor summary', err));
        return () => {
            if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
            // clear any pending badge timers
            Object.values(clearTimersRef.current).forEach(tid => window.clearTimeout(tid));
            clearTimersRef.current = {};
            lastTimestampRef.current = null;
            setNewCounts({});
        };
    }, [open, examId]);

    useEffect(() => {
        if (!selectedUser || !examId) return;
        fetch(`${API_URL}/exams/${examId}/proctoring/${selectedUser}`, { headers: { 'Content-Type': 'application/json', 'X-User-Id': user?._id || '' } }).then(r => r.json()).then(data => {
            if (data && data.events) setDetails(data.events);
        }).catch(err => console.error('Failed to load proctor details', err));
    }, [selectedUser, examId]);

    return (
        <Dialog open={open} onOpenChange={onOpenChange} className="max-w-4xl">
            <h2 className="text-2xl font-bold mb-2">Proctoring Dashboard</h2>
            <div className="grid grid-cols-3 gap-4">
                <div className="col-span-1 bg-slate-900 p-3 rounded">
                    <h3 className="font-semibold mb-2">Students</h3>
                    <div className="space-y-2 max-h-80 overflow-y-auto">
                        {summary.map(s => (
                            <button key={s.userId} onClick={() => {
                                setSelectedUser(s.userId);
                                // clear new count for this user when lecturer opens details
                                if (newCounts[s.userId]) {
                                    setNewCounts(prev => { const c = { ...prev }; delete c[s.userId]; return c; });
                                    if (clearTimersRef.current[s.userId]) { window.clearTimeout(clearTimersRef.current[s.userId]); delete clearTimersRef.current[s.userId]; }
                                }
                            }} className={cn('w-full text-left p-2 rounded hover:bg-slate-800 flex justify-between items-center', newCounts[s.userId] ? 'ring-2 ring-yellow-400' : '')}>
                                <div>
                                    <div className="font-medium">{s.name}</div>
                                    <div className="text-xs text-slate-400">Events: {s.count}</div>
                                </div>
                                <div className="flex items-center space-x-2">
                                    {newCounts[s.userId] ? (
                                        <div className="bg-red-500 text-white text-xs px-2 py-0.5 rounded-full font-medium animate-pulse">{newCounts[s.userId]}</div>
                                    ) : null}
                                    <div className={cn('h-3 w-20 rounded-full', s.count > 0 ? 'bg-red-500' : 'bg-green-500')}></div>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>
                <div className="col-span-2 bg-slate-900 p-3 rounded">
                    <h3 className="font-semibold mb-2">Events</h3>
                    {selectedUser ? (
                        <div>
                            <div className="text-sm text-slate-400 mb-3">Showing events for <strong>{selectedUser}</strong></div>
                            <div className="space-y-2 max-h-96 overflow-y-auto">
                                {details.length === 0 && <div className="text-slate-400">No events recorded.</div>}
                                {details.map((ev: any) => (
                                    <div key={ev._id} className="p-2 bg-slate-800 rounded">
                                        <div className="text-sm font-medium">{ev.eventType}</div>
                                        <div className="text-xs text-slate-400">{new Date(ev.timestamp).toLocaleString()}</div>
                                        {ev.details?.snapshot ? (
                                            <div className="mt-2">
                                                <img src={ev.details.snapshot} alt="snapshot" className="w-40 h-28 object-cover rounded border border-slate-700" />
                                            </div>
                                        ) : null}
                                        <pre className="text-xs mt-2 text-slate-300 bg-black/10 p-2 rounded overflow-x-auto">{JSON.stringify(ev.details)}</pre>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="text-slate-400">Select a student to view detailed proctoring events.</div>
                    )}
                </div>
            </div>
        </Dialog>
    );
};

const HelpPage = ({ onBack }: { onBack: () => void }) => {
    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="min-h-screen p-8">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Help & Documentation</h2>
                <Button variant="outline" onClick={onBack}>Back</Button>
            </div>
            <Card className="p-6">
                <h3 className="font-semibold mb-2">Getting Started</h3>
                <p className="text-slate-400">This is placeholder help content. Replace with your institution's help and FAQs. For now, here are some tips:</p>
                <ul className="list-disc list-inside mt-4 text-slate-300">
                    <li>Allow camera and microphone permissions before starting exams.</li>
                    <li>Ensure a stable internet connection.</li>
                    <li>Contact your instructor if you face issues.</li>
                </ul>
            </Card>
        </motion.div>
    );
};
const QuestionRenderer = ({ question, onAnswer, savedAnswer }: { question: Question, onAnswer: (answer: any) => void, savedAnswer: any }) => {
    return (
        <div>
            <p className="text-lg font-semibold text-slate-200 mb-6">{question.question}</p>
            
            {question.type === 'multiple-choice' && question.options && (
                <div className="space-y-3">
                    {question.options.map((option, index) => (
                        <label key={index} className="flex items-center space-x-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700 hover:bg-slate-700/50 cursor-pointer">
                            <input type="radio" name={question._id} value={index + 1} onChange={(e: ChangeEvent<HTMLInputElement>) => onAnswer(Number(e.target.value))} checked={savedAnswer === (index + 1)} className="form-radio h-5 w-5 text-indigo-500 bg-slate-700 border-slate-600 focus:ring-indigo-500" />
                            <span>{option}</span>
                        </label>
                    ))}
                </div>
            )}

            {question.type === 'true-false' && (
                <div className="space-y-3">
                     <label className="flex items-center space-x-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700 hover:bg-slate-700/50 cursor-pointer">
                        <input type="radio" name={question._id} value="true" onChange={() => onAnswer(true)} checked={savedAnswer === true} className="form-radio h-5 w-5 text-indigo-500 bg-slate-700 border-slate-600 focus:ring-indigo-500" />
                        <span>True</span>
                    </label>
                     <label className="flex items-center space-x-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700 hover:bg-slate-700/50 cursor-pointer">
                        <input type="radio" name={question._id} value="false" onChange={() => onAnswer(false)} checked={savedAnswer === false} className="form-radio h-5 w-5 text-indigo-500 bg-slate-700 border-slate-600 focus:ring-indigo-500" />
                        <span>False</span>
                    </label>
                </div>
            )}

            {question.type === 'short-answer' && (
                <Input placeholder="Type your answer here..." value={savedAnswer || ''} onChange={(e: ChangeEvent<HTMLInputElement>) => onAnswer(e.target.value)} />
            )}

            {question.type === 'essay' && (
                 <textarea rows={8} className="w-full rounded-md border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm" placeholder="Type your essay here..." value={savedAnswer || ''} onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onAnswer(e.target.value)}></textarea>
            )}
        </div>
    );
};
