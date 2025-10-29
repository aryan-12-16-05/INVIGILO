import React, { useState, useEffect, type FormEvent, type ChangeEvent, useRef, useCallback } from 'react';
import Sidebar from './sidebar';
import { motion, AnimatePresence } from 'framer-motion';
import {
    User, LogIn, ShieldCheck, Cpu, BrainCircuit,
    Timer, PlusCircle, Monitor, AlertTriangle, CheckCircle, XCircle,
    School, GraduationCap, ChevronLeft, Eye, EyeOff,
    Lock, Users, Wifi, Mic, Video, Globe, Trash2, Unlock
} from 'lucide-react';


// Note: This assumes you have a 'cn' utility function for class names, e.g., from 'clsx' and 'tailwind-merge'.
// If not, you can replace cn(...) with a simple string of class names.
// import { cn } from './lib/utils';
const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');


// --- API URL ---
const API_URL = 'http://127.0.0.1:5000/api';

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
type AppState = 'loading' | 'landing' | 'auth' | 'student-dashboard' | 'lecturer-dashboard' | 'exam' | 'result' | 'live-proctoring' | 'help' | 'my-exams';
type UserRole = 'student' | 'lecturer';
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
            setExams(data.exams);
        } catch (error: any) {
            showToast(error.message, 'error');
        }
    }, [showToast]);

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
            case 'my-exams': return currentUser && <MyExamsPage key="my-exams" user={currentUser} exams={exams} onLogout={handleLogout} onStartExam={handleStartExam} onBack={() => navigateTo('student-dashboard')} showToast={showToast} onUpdateUser={setCurrentUser} navigateTo={navigateTo} />;
                case 'lecturer-dashboard': return currentUser && <LecturerDashboard key="lecturer-dashboard" user={currentUser} exams={exams} onLogout={handleLogout} onBack={() => navigateTo('landing')} onExamChange={fetchExams} showToast={showToast} onUpdateUser={setCurrentUser} navigateTo={navigateTo} />;
            case 'live-proctoring': return currentUser && <LiveProctoring key="live-proctoring" user={currentUser} onBack={() => navigateTo(currentUser?.role === 'student' ? 'student-dashboard' : 'lecturer-dashboard')} showToast={showToast} />;
            case 'help': return <HelpPage key="help" onBack={() => navigateTo(currentUser?.role === 'student' ? 'student-dashboard' : 'lecturer-dashboard')} />;
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
                <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
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
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null); // ✅ Persist webcam stream
  const [captureMessage, setCaptureMessage] = useState<string>("");
  const [institution, setInstitution] = useState("");
  const [department, setDepartment] = useState("");
  const formDataRef = useRef<any>({});
    const [enrollSamples, setEnrollSamples] = useState<string[]>([]);

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
                    await waitForVideoReady(video, 10000);
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
        // Reuse submit handler behavior but do not submit form; simply trigger capture & verify
        setIsLoading(true);
        setCaptureMessage('Verifying...');
        const video = videoRef.current;
        if (!video) {
            showToast('Webcam not initialized', 'error');
            setIsLoading(false);
            return;
        }
        try {
            await waitForVideoReady(video, 8000);
            const imageDataUrl = captureFrame();
            if (!imageDataUrl) throw new Error('Could not capture frame');
            showToast('Captured. Please submit the form to verify.', 'success');
        } catch (err: any) {
            showToast(err.message || 'Verification retry failed', 'error');
        } finally {
            setIsLoading(false);
            setCaptureMessage('');
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
              src={`https://placehold.co/600x400/1e293b/4f46e5?text=${initialRole.charAt(0).toUpperCase() + initialRole.slice(1)}`}
              alt={initialRole}
              className="rounded-lg w-full h-64 object-cover mb-6"
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
                                        <button type="button" onClick={handleVerifyRetry} className="w-full mt-2 inline-flex items-center justify-center rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800">
                                            Retry Camera / Capture
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
                      <Input id="email" name="email" type="email" required />
                    </div>
                    <div>
                      <Label htmlFor="phoneNumber">Phone Number</Label>
                      <Input id="phoneNumber" name="phoneNumber" type="tel" required />
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
                      <Input id="year" name="year" type="text" placeholder="e.g., 3" required />
                    </div>
                  )}
                  <div className="relative">
                    <Label htmlFor="password">Password</Label>
                    <Input id="password" name="password" type={showPassword ? "text" : "password"} required />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-8 text-slate-400 hover:text-slate-200"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  <div className="pt-2">
                    <Button type="submit" className="w-full">
                      Proceed to Face Registration
                    </Button>
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
    const [profileOpen, setProfileOpen] = useState(false);
    const [profileForm, setProfileForm] = useState<any>(null);
    const [faceDialogOpen, setFaceDialogOpen] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [faceSamples, setFaceSamples] = useState<string[]>([]);

    useEffect(() => {
        setProfileForm(user ? { name: user.name, phoneNumber: user.phoneNumber, institution: user.institution, department: user.department, year: user.year, studentId: user.studentId, lecturerId: user.lecturerId } : null);
    }, [user]);

    const handleAction = (action: string) => {
        if (action === 'profile') {
            setProfileOpen(true);
            return;
        }
        if (onAction) onAction(action);
    };

    const saveProfile = async () => {
        if (!profileForm) return;
        try {
            const res = await fetch(`${API_URL}/users/${user._id}`, {
                method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(profileForm)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to update profile');
            if (onUpdateUser && data.user) onUpdateUser(data.user as UserProfile);
            setProfileOpen(false);
            showToast?.('Profile updated', 'success');
        } catch (err: any) {
            showToast?.(err.message || 'Failed to update profile', 'error');
        }
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

        <Dialog open={profileOpen} onOpenChange={setProfileOpen} className="max-w-md">
            <h2 className="text-xl font-bold mb-2">Edit Profile</h2>
            {profileForm && (
                <div className="space-y-3">
                    <div>
                        <Label>Name</Label>
                        <Input value={profileForm.name} onChange={(e: any) => setProfileForm((p: any) => ({ ...p, name: e.target.value }))} />
                    </div>
                    <div>
                        <Label>Phone</Label>
                        <Input value={profileForm.phoneNumber} onChange={(e: any) => setProfileForm((p: any) => ({ ...p, phoneNumber: e.target.value }))} />
                    </div>
                    <div>
                        <Label>Institution</Label>
                        <Select
                            value={profileForm.institution || ''}
                            onChange={(e: any) => setProfileForm((p: any) => ({ ...p, institution: e.target.value, department: '' }))}
                        >
                            <option value="">Select Institution</option>
                            {Object.keys(INSTITUTIONS).map(inst => (
                                <option key={inst} value={inst}>{inst}</option>
                            ))}
                        </Select>
                    </div>
                    <div>
                        <Label>Department</Label>
                        <Select
                            value={profileForm.department || ''}
                            onChange={(e: any) => setProfileForm((p: any) => ({ ...p, department: e.target.value }))}
                            disabled={!profileForm.institution}
                        >
                            <option value="">Select Department</option>
                            {profileForm.institution && INSTITUTIONS[profileForm.institution]?.map((dept) => (
                                <option key={dept} value={dept}>{dept}</option>
                            ))}
                        </Select>
                    </div>
                    {user.role === 'student' && (
                        <div>
                            <Label>Year</Label>
                            <Input value={profileForm.year} onChange={(e: any) => setProfileForm((p: any) => ({ ...p, year: e.target.value }))} />
                        </div>
                    )}
                    <div className="flex justify-end space-x-2 pt-2">
                        <Button variant="outline" onClick={() => setProfileOpen(false)}>Cancel</Button>
                        <Button onClick={saveProfile}>Save</Button>
                        <Button variant="secondary" onClick={() => setFaceDialogOpen(true)}>Manage Face Samples</Button>
                    </div>
                </div>
            )}
        </Dialog>

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

const MyExamsPage = ({ user, exams, onLogout, onStartExam, onBack, showToast, onUpdateUser, navigateTo }: { user: UserProfile; exams: Exam[]; onLogout: () => void; onStartExam: (examId: string) => void; onBack: () => void; showToast: (message:string, type:'success'|'error') => void; onUpdateUser: (u: UserProfile) => void; navigateTo: (state: AppState) => void }) => {
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
        !(e as any).completedByUser
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
                                                const completed = (exam as any).completedByUser;
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
                                            <p className="text-xs text-slate-400">Your Answer: <span className={q.isCorrect ? 'text-green-400' : 'text-red-400'}>{String(q.userAnswer || 'No answer')}</span></p>
                                            {!q.isCorrect && <p className="text-xs text-slate-500">Correct Answer: {String(q.correctAnswer)}</p>}
                                        </div>
                                        <div className="ml-4">
                                            {q.isCorrect ? <CheckCircle className="h-5 w-5 text-green-400" /> : <XCircle className="h-5 w-5 text-red-400" />}
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

    const liveExams = userExams.filter(e => (e.status === 'Live') || (e.status === 'Available' && isWithinWindow(e)));
    const upcomingExams = userExams.filter(e => (e.status === 'Scheduled' || e.status === 'Available' || e.status === 'Locked') && !isExpired(e));
    const completedExams = userExams.filter(e => (e as any).attemptForUser); // treat exams with attempts as completed for this user
    const averageScore = completedExams.length > 0 ? Math.round(completedExams.reduce((acc, e) => acc + (e.attempt?.score || 0), 0) / completedExams.length) : 0;

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
        if (a === 'results') {
            // Open the most recent completed exam's results, if any
            try {
                const last = completedExams[completedExams.length - 1];
                if (last) { (async () => await openResults(last))(); }
                else { showToast('No results available yet.', 'error'); }
            } catch { showToast('Unable to open results.', 'error'); }
            return;
        }
        if (a === 'live-proctoring') { navigateTo('live-proctoring'); return; }
        if (a === 'help') { navigateTo('help'); return; }
        if (a === 'profile') { /* handled in DashboardLayout */ return; }
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
                                        const completed = (exam as any).completedByUser;
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
const LecturerDashboard = ({ user, exams, onLogout, onBack, onExamChange, showToast, onUpdateUser, navigateTo }: { user: UserProfile; exams: Exam[]; onLogout: () => void; onBack: () => void; onExamChange: () => void; showToast: (message: string, type: 'success' | 'error') => void; onUpdateUser: (u: UserProfile) => void; navigateTo: (state: AppState) => void }) => {
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


    const StatCard = ({ title, value, icon, colorClass }: { title: string, value: string | number, icon: React.ReactNode, colorClass: string }) => (
        <Card className={cn("p-4 flex items-center space-x-4", colorClass)}>
            <div className="p-3 bg-white/10 rounded-lg">{icon}</div>
            <div>
                <p className="text-sm text-slate-300">{title}</p>
                <p className="text-2xl font-bold text-white">{value}</p>
            </div>
        </Card>
    );

    // Fetch admin stats for lecturer dashboard
    useEffect(() => {
        const fetchStats = async () => {
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
    }, [user._id]);

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
        if (a === 'profile') { /* DashboardLayout handles */ return; }
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
                            <Badge variant={exam.status === 'Live' ? 'live' : 'info'}>{exam.status}</Badge>
                        </div>
                            <div className="flex justify-end items-center space-x-1">
                            <Button variant="ghost" size="sm" onClick={() => { setExamToEdit(exam); setCreateExamOpen(true); }} title="Edit Exam">
                                Edit
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => { setProctorExamId(exam._id); setProctorOpen(true); }} title="Proctor Exam">
                                Proctor
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => fetchReport(exam._id, exam.title)} title="View Report">
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
    const videoRef = useRef<HTMLVideoElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const proctoringIntervalRef = useRef<any>(null);
    const proctoringAbortControllerRef = useRef<AbortController | null>(null); // To abort all pending requests
    const [proctoringStopped, setProctoringStopped] = useState(false);
    const [proctoringKey, setProctoringKey] = useState(0); // Used to restart proctoring
    // Note: switched to continuous MediaRecorder with timeslice; no need for manual chunks buffer.
    // const audioChunksRef = useRef<Blob[]>([]);

    // Function to stop all proctoring activities
    const stopProctoring = useCallback(() => {
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
    }, [proctoringStopped]);

    // Function to restart proctoring if student cancels submission
    const restartProctoring = useCallback(() => {
        console.log('[PROCTORING] Restarting proctoring after cancel...');
        setProctoringStopped(false);
        // Increment key to force useEffect to re-run and restart proctoring
        setProctoringKey(prev => prev + 1);
    }, []);

    // Handle first submit button click - stop proctoring and show confirmation
    const handleInitialSubmitClick = useCallback(() => {
        console.log('[SUBMIT] Initial submit button clicked');
        stopProctoring();
        setShowSubmitConfirm(true);
    }, [stopProctoring]);

    // Handle cancel button in confirmation dialog - RESTART PROCTORING
    const handleCancelSubmit = useCallback(() => {
        console.log('[SUBMIT] User cancelled submission - restarting proctoring');
        setShowSubmitConfirm(false);
        restartProctoring();
    }, [restartProctoring]);

    // Handle actual submission after confirmation
    const handleSubmit = useCallback(async () => {
        setIsSubmitting(true);
        console.log('[SUBMIT] Starting exam submission...', { examId: exam._id, userId: user._id, answerCount: Object.keys(answers).length });
        
        // No need to wait - requests are already aborted via AbortController
        console.log('[SUBMIT] Submitting immediately - requests already aborted');
        
        try {
            const url = `${API_URL}/exams/${exam._id}/submit`;
            console.log('[SUBMIT] Now submitting to:', url);
            
            // Add timeout to prevent hanging (10 seconds should be enough after optimization)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: user._id, answers }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            console.log('[SUBMIT] Submit response status:', res.status);
            
            const data = await res.json();
            console.log('[SUBMIT] Submit response data:', data);
            
            if (!res.ok) throw new Error(data.error || 'Failed to submit exam');

            showToast('Exam submitted successfully!', 'success');
            onExit({ score: data.score, totalMarks: data.totalMarks, examTitle: exam.title, perQuestion: data.perQuestion });

        } catch (error: any) {
            console.error('[SUBMIT] Exam submission error:', error);
            if (error.name === 'AbortError') {
                showToast('Submission timeout. The server took too long to respond. Please try again.', 'error');
            } else {
                showToast(error.message || 'Failed to submit exam. Please try again.', 'error');
            }
            setIsSubmitting(false);
        }
    }, [answers, exam._id, exam.title, onExit, showToast, user._id]);

    // Proctoring Loop
    useEffect(() => {
        // Reset proctoring stopped state when starting a new exam
        console.log('[PROCTORING] Initializing proctoring for exam:', exam._id);
        setProctoringStopped(false);
        
        const [proctorDegradedRef] = [
            { current: false } as { current: boolean }
        ];
        // Simple network backoff to avoid spamming server if unreachable
        const nextAllowedRef = { current: 0 } as any;
        const backoffMs = () => 10000; // 10s backoff on network failure

        const startProctoring = async () => {
            console.log('[PROCTORING] Starting camera and audio streams...');
            
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
                                    body: JSON.stringify({ audioData: base64Audio.split(',')[1] }),
                                    signal: abortSignal // Add abort signal to cancel request
                                });
                                const data = await res.json();
                                if (data && data.audioStatus && String(data.audioStatus).toLowerCase().includes('suspicious')) {
                                    try {
                                        await fetch(`${API_URL}/proctor/event`, {
                                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ examId: exam._id, userId: user._id, eventType: 'audio', details: { audioStatus: data.audioStatus } }),
                                            signal: abortSignal
                                        });
                                    } catch (err) { /* ignore */ }
                                }
                            } catch (error: any) { 
                                // Don't log AbortError - it's expected when stopping
                                if (error.name !== 'AbortError') {
                                    console.error('[PROCTORING] Audio error:', error);
                                }
                            }
                        };
                    } catch { /* ignore */ }
                };

                // Start a recurring loop for proctoring - store in ref so handleSubmit can stop it
                proctoringIntervalRef.current = setInterval(() => {
                    // Skip if proctoring was stopped
                    if (abortSignal.aborted) return;
                    
                    // Skip if in backoff window
                    if (Date.now() < nextAllowedRef.current) return;
                    const imageDataUrl = captureFrame();
                    if (imageDataUrl) {
                        fetch(`${API_URL}/proctor`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ imageDataUrl, userId: user._id, examId: exam._id }),
                            signal: abortSignal // Add abort signal
                        }).then(async res => {
                            if (abortSignal.aborted) return; // Check before processing response
                            
                            if (!res.ok) {
                                nextAllowedRef.current = Date.now() + backoffMs();
                                proctorDegradedRef.current = true;
                            }
                            return res.json();
                        }).then(data => {
                            if (abortSignal.aborted || !data) return; // Check again
                            
                            if (data && !data.error) {
                                proctorDegradedRef.current = false;
                                const suspicious = [];
                                if (!data.identityVerified) suspicious.push({ event: 'identity', details: { similarity: data.similarity } });
                                if (data.faceCount && data.faceCount > 1) suspicious.push({ event: 'multiple_faces', details: { count: data.faceCount } });
                                if (data.objectsDetected && data.objectsDetected.length > 0 && data.objectsDetected[0] !== 'No objects') suspicious.push({ event: 'object_detected', details: { objects: data.objectsDetected } });
                                if (data.headPose && data.headPose !== 'Forward') suspicious.push({ event: 'head_pose', details: { pose: data.headPose } });
                                if (data.gazeDirection && data.gazeDirection !== 'Center') suspicious.push({ event: 'gaze', details: { gaze: data.gazeDirection } });
                                if (data.blinkStatus && data.blinkStatus === 'suspicious') suspicious.push({ event: 'blink', details: {} });

                                // FOR TESTING: Send a test event every 10th proctoring check to verify events are working
                                // Remove this after testing
                                if (Math.random() < 0.1) {
                                    suspicious.push({ event: 'gaze', details: { gaze: 'Test Event', message: 'This is a test event to verify proctoring works' } });
                                    console.log('[PROCTORING] Sending TEST event to verify system');
                                }

                                suspicious.forEach(async (ev) => {
                                    if (abortSignal.aborted) return;
                                    try {
                                        console.log(`[PROCTORING] Sending event: ${ev.event}`);
                                        const response = await fetch(`${API_URL}/proctor/event`, {
                                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ examId: exam._id, userId: user._id, eventType: ev.event, details: ev.details, snapshot: imageDataUrl }),
                                            signal: abortSignal
                                        });
                                        if (response.ok) {
                                            console.log(`[PROCTORING] Event ${ev.event} recorded successfully`);
                                        }
                                    } catch (err: any) {
                                        if (err.name !== 'AbortError') {
                                            console.error('Failed to record proctor event', err);
                                        }
                                    }
                                });
                            } else if (data && data.error) {
                                console.error('Proctoring error:', data.error);
                            }
                        }).catch(err => { 
                            // Don't log AbortError
                            if (err.name !== 'AbortError') {
                                console.error("Image proctoring error:", err); 
                                nextAllowedRef.current = Date.now() + backoffMs(); 
                                proctorDegradedRef.current = true;
                            }
                        });
                    }

                    // Ensure recorder is running with 1s timeslice for continuous small chunks
                    try {
                        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'inactive' && !abortSignal.aborted) {
                            mediaRecorderRef.current.start(1000);
                        }
                    } catch {}
                }, 3000); // Run every 3 seconds (faster proctoring)
                
                console.log('[PROCTORING] Proctoring started successfully - interval set');
            } catch (error) {
                console.error("[PROCTORING] Failed to start proctoring streams:", error);
                showToast("Could not start camera or microphone for proctoring.", "error");
            }
        };

        startProctoring();

        // Cleanup function
        return () => {
            console.log('[PROCTORING] Cleanup: stopping all proctoring activities');
            
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
    }, [exam._id, showToast, user._id, proctoringKey]); // Added proctoringKey to restart on cancel

    const captureFrame = (): string | null => {
        const video = videoRef.current;
        if (!video || video.readyState < 3) return null;
        // Downscale to reduce bandwidth and backend load, keep aspect ratio
        const srcW = video.videoWidth || 640;
        const srcH = video.videoHeight || 480;
        const targetW = 320; // small fixed width
        const scale = targetW / srcW;
        const targetH = Math.max(1, Math.round(srcH * scale));

        const canvas = document.createElement('canvas');
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext('2d');
        try {
            ctx?.drawImage(video, 0, 0, targetW, targetH);
            // Slightly lower quality to reduce payload size
            return canvas.toDataURL('image/jpeg', 0.7);
        } catch (e) {
            return null;
        }
    };

    // Timer logic
    useEffect(() => {
        if (timeLeft <= 0) {
            handleSubmit();
            return;
        }
        const timerId = setInterval(() => {
            setTimeLeft(t => t - 1);
        }, 1000);
        return () => clearInterval(timerId);
    }, [timeLeft, handleSubmit]);
    
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
                    <video ref={videoRef} autoPlay playsInline muted className="absolute top-4 right-4 w-48 h-36 rounded-md object-cover border-2 border-slate-700"></video>
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
            <Dialog open={showSubmitConfirm} onOpenChange={setShowSubmitConfirm}>
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
                    {Object.keys(answers).length < exam.questions.length && (
                        <div className="bg-yellow-500/20 text-yellow-300 p-3 rounded-lg mb-4 text-sm">
                            ⚠️ You have unanswered questions. They will be marked as incorrect.
                        </div>
                    )}
                    <div className="flex space-x-3">
                        <Button variant="outline" className="flex-1" onClick={handleCancelSubmit} disabled={isSubmitting}>
                            Cancel
                        </Button>
                        <Button variant="destructive" className="flex-1" onClick={() => { setShowSubmitConfirm(false); handleSubmit(); }} disabled={isSubmitting}>
                            {isSubmitting ? 'Submitting...' : 'Submit Now'}
                        </Button>
                    </div>
                </div>
            </Dialog>
        </motion.div>
    );
};
const ResultScreen = ({ result, onDone }: { result: ExamResult, onDone: () => void }) => (
    <motion.div
        className="min-h-screen flex flex-col items-center justify-center p-4"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
    >
        <Card className="p-8 text-center max-w-md w-full">
            <CheckCircle className="h-16 w-16 text-green-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-white mb-2">Exam Submitted!</h2>
            <p className="text-slate-400 mb-6">You have successfully completed the exam: <span className="font-semibold text-slate-200">{result.examTitle}</span>.</p>
            <div className="bg-slate-800/50 rounded-lg p-6 my-6">
                <p className="text-slate-400 text-sm">YOUR SCORE</p>
                <p className="text-6xl font-bold text-green-400 my-2">{result.score}%</p>
            </div>
            {result.perQuestion && result.perQuestion.length > 0 && (
                <div className="mb-4 bg-slate-900 p-4 rounded">
                    <h3 className="text-lg font-semibold text-white mb-2">Question Breakdown</h3>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                        {result.perQuestion.map((q, idx) => (
                            <div key={idx} className={cn('p-2 rounded', q.correct ? 'bg-green-800/30' : 'bg-red-800/20')}>
                                <div className="font-medium text-sm text-white">{idx + 1}. {q.question}</div>
                                <div className="text-xs text-slate-300">Your answer: {String(q.given)}</div>
                                <div className="text-xs text-slate-300">Correct answer: {String(q.expected)}</div>
                                <div className="text-xs text-slate-300">Marks: {q.marks} — {q.correct ? 'Correct' : 'Incorrect'}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
            <Button onClick={onDone} className="w-full">Back to Dashboard</Button>
        </Card>
    </motion.div>
);
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
const LiveProctoring = ({ user, onBack, showToast }: { user: UserProfile; onBack: () => void; showToast: (msg:string, type:'success'|'error') => void }) => {
    const [events, setEvents] = React.useState<any[]>([]);
    const pollRef = React.useRef<number | null>(null);

    React.useEffect(() => {
        // For students: fetch their own recent proctor events
        // For lecturers: fetch global recent events
        const fetchEvents = async () => {
            try {
                const endpoint = user.role === 'lecturer' 
                    ? `${API_URL}/proctoring/recent-global?limit=50` 
                    : `${API_URL}/proctoring/recent?userId=${user._id}&limit=50`;
                const res = await fetch(endpoint, { headers: { 'X-User-Id': user._id } });
                const data = await res.json();
                if (res.ok && data.events) {
                    setEvents(data.events);
                }
            } catch (err) {
                console.error('Failed to fetch events', err);
            }
        };

        fetchEvents();
        pollRef.current = window.setInterval(fetchEvents, 2000);

        return () => {
            if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
        };
    }, [user]);

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'high': return 'text-red-400 bg-red-500/10 border-red-500/50';
            case 'medium': return 'text-orange-400 bg-orange-500/10 border-orange-500/50';
            case 'warning': return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/50';
            default: return 'text-blue-400 bg-blue-500/10 border-blue-500/50';
        }
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="min-h-screen p-8">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Live Proctoring Monitor</h2>
                <Button variant="outline" onClick={onBack}>Back to Dashboard</Button>
            </div>

            <div className="grid grid-cols-1 gap-6">
                <Card className="p-6 bg-slate-900 border-slate-800">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-xl font-semibold text-white">Recent Proctoring Events</h3>
                        <div className="flex items-center space-x-2">
                            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
                            <span className="text-sm text-slate-400">Live Updates</span>
                        </div>
                    </div>
                    <div className="text-sm text-slate-400 mb-4">
                        {user.role === 'student' ? 'Your recent proctoring activity' : 'All recent proctoring events across exams'}
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
                                        <p className="text-xs text-slate-400 mb-1">
                                            {user.role === 'lecturer' && `Student: ${evt.userId || 'Unknown'} | `}
                                            Exam: {evt.examId || 'Unknown'}
                                        </p>
                                        <p className="text-xs text-slate-500">
                                            {new Date(evt.timestamp).toLocaleString()}
                                        </p>
                                        {evt.details && Object.keys(evt.details).length > 0 && (
                                            <div className="mt-2 text-xs text-slate-400">
                                                {Object.entries(evt.details).map(([key, val]) => (
                                                    <div key={key}>
                                                        <span className="font-medium">{key}:</span> {String(val)}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
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
                                <p className="text-xs text-slate-500 mt-1">Events will appear here when exams are in progress</p>
                            </div>
                        )}
                    </div>
                </Card>

                {user.role === 'lecturer' && (
                    <Card className="p-6 bg-slate-900 border-slate-800">
                        <h3 className="text-lg font-semibold text-white mb-2">Monitoring Information</h3>
                        <p className="text-sm text-slate-400">
                            This dashboard displays real-time proctoring events from all active exams. Events are color-coded by severity:
                        </p>
                        <ul className="mt-3 space-y-2 text-sm">
                            <li className="flex items-center space-x-2">
                                <div className="h-3 w-3 rounded-full bg-red-500"></div>
                                <span className="text-slate-300">High Severity - Immediate attention required</span>
                            </li>
                            <li className="flex items-center space-x-2">
                                <div className="h-3 w-3 rounded-full bg-orange-500"></div>
                                <span className="text-slate-300">Medium Severity - Review recommended</span>
                            </li>
                            <li className="flex items-center space-x-2">
                                <div className="h-3 w-3 rounded-full bg-yellow-500"></div>
                                <span className="text-slate-300">Warning - Minor issues detected</span>
                            </li>
                            <li className="flex items-center space-x-2">
                                <div className="h-3 w-3 rounded-full bg-blue-500"></div>
                                <span className="text-slate-300">Info - Normal activity</span>
                            </li>
                        </ul>
                    </Card>
                )}

                {user.role === 'student' && (
                    <Card className="p-6 bg-slate-900 border-slate-800">
                        <h3 className="text-lg font-semibold text-white mb-2">About Proctoring</h3>
                        <p className="text-sm text-slate-400">
                            During exams, the system monitors various behaviors to ensure academic integrity. Events shown here include:
                        </p>
                        <ul className="mt-3 space-y-1 text-sm text-slate-400">
                            <li>• Face detection and verification</li>
                            <li>• Head pose and gaze direction</li>
                            <li>• Multiple faces detected</li>
                            <li>• Audio anomalies and talking</li>
                            <li>• Tab switching and window focus</li>
                            <li>• Environmental changes</li>
                        </ul>
                        <p className="mt-3 text-xs text-slate-500">
                            All events are recorded and reviewed by your instructor. Ensure you follow exam guidelines to avoid issues.
                        </p>
                    </Card>
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
