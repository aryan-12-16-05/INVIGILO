import React, { useState, useEffect, type FormEvent, type ChangeEvent, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    User, LogIn, LogOut, Settings, HelpCircle, ShieldCheck, Cpu, BrainCircuit,
    FileText, Timer, BarChart, PlusCircle, Monitor, AlertTriangle, CheckCircle, XCircle,
    ClipboardList, School, GraduationCap, ChevronLeft, Eye, EyeOff,
    Clock, Lock, Users, Wifi, Mic, Video, Globe, Phone, Camera, Trash2, Unlock
} from 'lucide-react';

// Note: This assumes you have a 'cn' utility function for class names, e.g., from 'clsx' and 'tailwind-merge'.
// If not, you can replace cn(...) with a simple string of class names.
// import { cn } from './lib/utils';
const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');


// --- API URL ---
const API_URL = 'http://127.0.0.1:5000/api';

// --- MOCK INSTITUTION DATA ---
const INSTITUTIONS: { [key: string]: string[] } = {
    "Quantum University": ["Computer Science", "Physics", "Mathematics"],
    "Starlight College": ["Information Technology", "Business Administration"],
    "Apex Institute": ["Mechanical Engineering", "Civil Engineering"]
};

// --- TYPE DEFINITIONS ---
type AppState = 'loading' | 'landing' | 'auth' | 'student-dashboard' | 'lecturer-dashboard' | 'exam' | 'result';
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
            const res = await fetch(`${API_URL}/exams`);
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
            case 'student-dashboard': return currentUser && <StudentDashboard key="student-dashboard" user={currentUser} exams={exams} onLogout={handleLogout} onStartExam={handleStartExam} onBack={() => navigateTo('landing')} />;
            case 'lecturer-dashboard': return currentUser && <LecturerDashboard key="lecturer-dashboard" user={currentUser} exams={exams} onLogout={handleLogout} onBack={() => navigateTo('landing')} onExamChange={fetchExams} showToast={showToast} />;
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
  const [captureMessage, setCaptureMessage] = useState<string>("");
  const [institution, setInstitution] = useState("");
  const [department, setDepartment] = useState("");
  const formDataRef = useRef<any>({});

  useEffect(() => {
    if (currentStep === "face" || authMode === "signin") {
      const videoElement = videoRef.current;
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then((stream) => {
            if (videoElement) videoElement.srcObject = stream;
          })
          .catch((err) => {
            console.error("Error accessing webcam: ", err);
            showToast("Could not access webcam.", "error");
          });
      }
      return () => {
        if (videoElement && videoElement.srcObject) {
          const stream = videoElement.srcObject as MediaStream;
          stream.getTracks().forEach((track) => track.stop());
        }
      };
    }
  }, [currentStep, authMode, showToast]);

console.log('videoRef.current:', videoRef.current);
console.log('readyState:', videoRef.current?.readyState);
console.log('videoWidth, videoHeight:', videoRef.current?.videoWidth, videoRef.current?.videoHeight);
const waitForVideoReady = (video: HTMLVideoElement, timeout = 10000): Promise<void> => {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      if (video.readyState >= 3 && video.videoWidth > 0 && video.videoHeight > 0) {
        resolve();
      } else if (Date.now() - start > timeout) {
        reject(new Error('Video not ready in time'));
      } else {
        requestAnimationFrame(check);
      }
    };
    check();
  });
};


const captureFrame = (): string | null => {
  const video = videoRef.current;
  if (!video || video.readyState < 3 || video.videoWidth === 0 || video.videoHeight === 0) return null;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg');
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
  setCaptureMessage('');

  const video = videoRef.current;
  if (!video) {
    showToast("Webcam not initialized", 'error');
    setIsLoading(false);
    return;
  }

  try {
    await waitForVideoReady(video,10000); // ✅ wait for webcam

    // Countdown
    for (let i = 3; i > 0; i--) {
      setCaptureMessage(i.toString());
      await new Promise(res => setTimeout(res, 1000));
    }

    setCaptureMessage('Capturing...');
    const imageDataUrl = captureFrame();
    if (!imageDataUrl) {
      throw new Error("Could not capture image. Please ensure your camera is ready.");
    }

    const finalData = { ...formDataRef.current, imageDataUrl, role: initialRole };

    const res = await fetch(`${API_URL}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(finalData)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Registration failed');

    showToast(data.message, 'success');
    setAuthMode('signin');
    setCurrentStep('details');

  } catch (error: any) {
    showToast(error.message, 'error');
  } finally {
    setIsLoading(false);
    setCaptureMessage('');
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

    setTimeout(async () => {
      const imageDataUrl = captureFrame();
      if (!imageDataUrl) {
        showToast("Could not capture image for verification.", "error");
        setIsLoading(false);
        setCaptureMessage("");
        return;
      }

      try {
        const faceRes = await fetch(`${API_URL}/verify-face`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ identifier: currentIdentifier, role: initialRole, imageDataUrl }),
        });
        const faceData = await faceRes.json();
        if (!faceRes.ok) throw new Error(faceData.error || faceData.message);
        showToast("Face verified. Logging in...", "success");

        const loginRes = await fetch(`${API_URL}/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ identifier: currentIdentifier, password: currentPassword, role: initialRole }),
        });
        const loginData = await loginRes.json();
        if (!loginRes.ok) throw new Error(loginData.error || "Login failed");

        showToast(`Welcome back, ${loginData.user.name}!`, "success");
        onAuthSuccess(loginData.user);
      } catch (error: any) {
        showToast(error.message, "error");
      } finally {
        setIsLoading(false);
        setCaptureMessage("");
      }
    }, 1500);
  };

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
                    {captureMessage && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                        <p className="text-white text-3xl font-bold">{captureMessage}</p>
                      </div>
                    )}
                  </div>
                  <Button onClick={handleFullSignUp} className="w-full" isLoading={isLoading}>
                    Capture & Complete Signup
                  </Button>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </motion.div>
      </div>
    </motion.div>
  );
};




const DashboardLayout = ({ children, user, onLogout, onBack }: { children: React.ReactNode, user: UserProfile, onLogout: () => void, onBack: () => void }) => {
    const navItems = user.role === 'student' ? [
        { icon: <ClipboardList className="h-5 w-5" />, label: 'Dashboard' },
        { icon: <FileText className="h-5 w-5" />, label: 'My Exams' },
        { icon: <BarChart className="h-5 w-5" />, label: 'Results' },
        { icon: <User className="h-5 w-5" />, label: 'Profile' },
        { icon: <HelpCircle className="h-5 w-5" />, label: 'Help' },
    ] : [
        { icon: <Monitor className="h-5 w-5" />, label: 'Overview' },
        { icon: <PlusCircle className="h-5 w-5" />, label: 'Create Exam' },
        { icon: <BrainCircuit className="h-5 w-5" />, label: 'AI Questions' },
        { icon: <Users className="h-5 w-5" />, label: 'Live Proctoring' },
        { icon: <Settings className="h-5 w-5" />, label: 'Settings' },
    ];

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex min-h-screen">
            <div className="w-64 bg-slate-900 p-4 flex flex-col border-r border-slate-800">
                <div className="flex items-center space-x-2 mb-10">
                    <BrainCircuit className="h-8 w-8 text-indigo-400" />
                    <span className="text-xl font-bold">Invigilo</span>
                </div>
                <nav className="flex-1 space-y-2">
                    {navItems.map((item, index) => (
                        <a key={item.label} href="#" className={cn(
                            "flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors duration-200",
                            index === 0 ? "bg-indigo-600/50 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-white"
                        )}>
                            {item.icon}
                            <span>{item.label}</span>
                        </a>
                    ))}
                </nav>
                <div className="mt-auto">
                    <div className="flex items-center space-x-3 mb-4 p-2">
                        <div className="w-10 h-10 rounded-full bg-indigo-500 flex items-center justify-center font-bold">{user.name.charAt(0)}</div>
                        <div>
                            <p className="font-semibold text-sm text-white">{user.name}</p>
                            <p className="text-xs text-slate-400">{user.institution}</p>
                        </div>
                    </div>
                    <Button variant="secondary" className="w-full" onClick={onLogout}><LogOut className="h-4 w-4 mr-2" /> Logout</Button>
                </div>
            </div>
            <main className="flex-1 p-8 bg-slate-950/50 overflow-y-auto">
                <div className="flex justify-between items-center mb-8">
                    <h1 className="text-3xl font-bold text-white">Welcome back, {user.name.split(' ')[0]}!</h1>
                    <Button variant="outline" onClick={onBack}><ChevronLeft className="h-4 w-4 mr-2" /> Back to Home</Button>
                </div>
                {children}
            </main>
        </motion.div>
    );
};
const StudentDashboard = ({ user, exams, onLogout, onStartExam, onBack }: { user: UserProfile; exams: Exam[], onLogout: () => void; onStartExam: (examId: string) => void; onBack: () => void; }) => {
    const [systemCheckOpen, setSystemCheckOpen] = useState(false);
    
    const userExams = exams.filter(exam => 
        exam.institution.toLowerCase() === user.institution.toLowerCase() && 
        exam.department.toLowerCase() === user.department.toLowerCase() && 
        exam.targetYear === user.year
    );

    const liveExams = userExams.filter(e => e.status === 'Live');
    const upcomingExams = userExams.filter(e => e.status === 'Scheduled' || e.status === 'Available' || e.status === 'Locked');
    const completedExams = userExams.filter(e => e.status === 'Completed');
    const averageScore = completedExams.length > 0 ? Math.round(completedExams.reduce((acc, e) => acc + (e.attempt?.score || 0), 0) / completedExams.length) : 0;

    return (
        <DashboardLayout user={user} onLogout={onLogout} onBack={onBack}>
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
                                    <Button onClick={() => onStartExam(exam._id)} disabled={exam.status !== 'Available'}>
                                        {exam.status === 'Locked' ? <Lock className="h-4 w-4 mr-2"/> : null}
                                        {exam.status === 'Available' ? 'Start Exam' : 'View Details'}
                                    </Button>
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
                                        <p className="text-xs text-slate-500">{exam.attempt?.completedAt}</p>
                                    </div>
                                    <Badge variant={(exam.attempt?.score || 0) >= 80 ? 'success' : 'warning'}>{exam.attempt?.score}%</Badge>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>
            </div>
            
            <SystemCheckDialog open={systemCheckOpen} onOpenChange={setSystemCheckOpen} />
        </DashboardLayout>
    );
};
const LecturerDashboard = ({ user, exams, onLogout, onBack, onExamChange, showToast }: { user: UserProfile; exams: Exam[]; onLogout: () => void; onBack: () => void; onExamChange: () => void; showToast: (message: string, type: 'success' | 'error') => void; }) => {
    const lecturerExams = exams.filter((exam) => exam.lecturerId === user._id);
    const [createExamOpen, setCreateExamOpen] = useState(false);
    const [examToDelete, setExamToDelete] = useState<Exam | null>(null);


    const liveExamsCount = lecturerExams.filter(e => e.status === 'Live').length;

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

    return (
        <DashboardLayout user={user} onLogout={onLogout} onBack={onBack}>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard title="Total Students" value="127" icon={<Users className="h-6 w-6 text-indigo-300"/>} colorClass="border-indigo-500/50" />
                <StatCard title="Live Exams" value={liveExamsCount} icon={<Monitor className="h-6 w-6 text-green-300"/>} colorClass="border-green-500/50" />
                <StatCard title="Active Alerts" value="15" icon={<AlertTriangle className="h-6 w-6 text-yellow-300"/>} colorClass="border-yellow-500/50" />
                <StatCard title="System Uptime" value="99.9%" icon={<ShieldCheck className="h-6 w-6 text-sky-300"/>} colorClass="border-sky-500/50" />
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
            <CreateExamDialog open={createExamOpen} onOpenChange={setCreateExamOpen} lecturer={user} onExamCreated={onExamChange} showToast={showToast} />
            
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
    const videoRef = useRef<HTMLVideoElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);

    const handleSubmit = useCallback(async () => {
        setIsSubmitting(true);
        try {
            const res = await fetch(`${API_URL}/exams/${exam._id}/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: user._id, answers })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to submit exam');

            onExit({ score: data.score, totalMarks: data.totalMarks, examTitle: exam.title });

        } catch (error: any) {
            showToast(error.message, 'error');
            setIsSubmitting(false);
        }
    }, [answers, exam._id, exam.title, onExit, showToast, user._id]);

    // Proctoring Loop
    useEffect(() => {
        let proctoringInterval: any; // Using 'any' to avoid NodeJS/browser type conflicts for setInterval

        const startProctoring = async () => {
            try {
                // Start video stream
                const videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (videoRef.current) {
                    videoRef.current.srcObject = videoStream;
                }

                // Start audio stream and recorder
                const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorderRef.current = new MediaRecorder(audioStream);
                mediaRecorderRef.current.ondataavailable = (event) => {
                    audioChunksRef.current.push(event.data);
                };
                mediaRecorderRef.current.onstop = async () => {
                    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = async () => {
                        const base64Audio = reader.result as string;
                        try {
                            const res = await fetch(`${API_URL}/proctor/audio`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ audioData: base64Audio.split(',')[1] }), // Send only base64 part
                            });
                            const data = await res.json();
                            if (data.audioStatus === "Suspicious audio detected") {
                                showToast("Suspicious audio detected!", 'error');
                            }
                        } catch (error) {
                            console.error("Audio proctoring error:", error);
                        }
                    };
                    audioChunksRef.current = [];
                };

                // Start a recurring loop for proctoring
                proctoringInterval = setInterval(() => {
                    const imageDataUrl = captureFrame();
                    if (imageDataUrl) {
                        fetch(`${API_URL}/proctor`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ imageDataUrl, userId: user._id }),
                        }).then(res => res.json()).then(data => {
                            if (data.error) {
                                showToast(data.error, 'error');
                            } else {
                                if (!data.identityVerified) {
                                    showToast("Identity verification failed. Please ensure you are the registered user.", 'error');
                                }
                                if (data.objectsDetected && data.objectsDetected.length > 0 && data.objectsDetected[0] !== "No objects") {
                                    showToast(`Suspicious object detected: ${data.objectsDetected.join(', ')}`, 'error');
                                }
                                if (data.headPose !== 'Forward') {
                                    showToast(`Suspicious head pose: ${data.headPose}`, 'error');
                                }
                                if (data.gazeDirection !== 'Center') {
                                    showToast(`Suspicious eye gaze: Looking ${data.gazeDirection}`, 'error');
                                }
                            }
                        }).catch(err => console.error("Image proctoring error:", err));
                    }

                    if (mediaRecorderRef.current?.state === 'inactive') {
                        mediaRecorderRef.current.start();
                        setTimeout(() => {
                            if (mediaRecorderRef.current?.state === 'recording') {
                                mediaRecorderRef.current.stop();
                            }
                        }, 2000); // Record for 2 seconds
                    }
                }, 8000); // Run every 8 seconds
            } catch (error) {
                console.error("Failed to start proctoring streams:", error);
                showToast("Could not start camera or microphone for proctoring.", "error");
            }
        };

        startProctoring();

        // Cleanup function
        return () => {
            clearInterval(proctoringInterval);
            if (videoRef.current && videoRef.current.srcObject) {
                (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
            }
            if (mediaRecorderRef.current && mediaRecorderRef.current.stream) {
                mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [showToast, user._id]);

    const captureFrame = (): string | null => {
        const video = videoRef.current;
        if (video && video.readyState >= 3) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg');
        }
        return null;
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
                <div className="mt-auto">
                    <Button variant="destructive" className="w-full" isLoading={isSubmitting} onClick={handleSubmit}>Submit Exam</Button>
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
const CreateExamDialog = ({ open, onOpenChange, lecturer, onExamCreated, showToast }: { open: boolean, onOpenChange: (open: boolean) => void; lecturer: UserProfile, onExamCreated: () => void, showToast: (message: string, type: 'success' | 'error') => void; }) => {
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
            const res = await fetch(`${API_URL}/exams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(examDetails)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to create exam');

            showToast('Exam created successfully!', 'success');
            onExamCreated();
            onOpenChange(false);
            e.currentTarget.reset();
            setQuestions([]);

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
                    <div><Label htmlFor="exam-title">Exam Title</Label><Input id="exam-title" name="title" required/></div>
                    <div><Label htmlFor="course-code">Course Code</Label><Input id="course-code" name="courseCode" required/></div>
                </div>
                 <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                        <Label>Institution</Label>
                        <Input value={lecturer.institution} disabled />
                    </div>
                    <div className="space-y-1">
                        <Label htmlFor="department-create">Department</Label>
                        <Select id="department-create" name="department" value={department} onChange={(e: ChangeEvent<HTMLSelectElement>) => setDepartment(e.target.value)} required>
                            <option value="">Select Department</option>
                            {INSTITUTIONS[lecturer.institution]?.map(dept => <option key={dept} value={dept}>{dept}</option>)}
                        </Select>
                    </div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                    <div><Label htmlFor="exam-date">Exam Date</Label><Input id="exam-date" name="scheduledDate" type="date" required/></div>
                    <div><Label htmlFor="start-time">Start Time</Label><Input id="start-time" name="startTime" type="time" required/></div>
                    <div><Label htmlFor="duration">Duration (mins)</Label><Input id="duration" name="duration" type="number" required/></div>
                </div>
                 <div className="grid grid-cols-2 gap-4">
                    <div><Label htmlFor="targetYear">Target Year</Label><Input id="targetYear" name="targetYear" placeholder="e.g., 3" required /></div>
                </div>
                <div><Label htmlFor="description">Description</Label><textarea id="description" name="description" rows={2} className="w-full rounded-md border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm" required></textarea></div>
                
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

