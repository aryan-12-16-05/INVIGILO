import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Camera, Eye, Activity, ScanFace, AlertTriangle, XCircle, CheckCircle,
    Radio, ChevronLeft, ChevronRight, TrendingUp, Settings, Download,
    Filter, Bell, Plus, FileText, Users as UsersIcon, ArrowLeft, Play, Pause, X, Monitor
} from 'lucide-react';
import { io, type Socket } from 'socket.io-client';

const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

const API_URL = (() => {
    const RAW_API_URL = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:5000/api';
    const trimmed = String(RAW_API_URL).trim().replace(/\/+$/, '');
    return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
})();

interface LiveStudent {
    userId: string;
    studentId: string;
    name: string;
    faceDetected: boolean;
    gaze: 'forward' | 'away' | 'down';
    headPose: 'normal' | 'tilted' | 'down';
    violations: number;
    status: 'normal' | 'warning' | 'suspicious' | 'critical';
    timeRemaining: number;
    latestViolation?: string;
    violationTime?: string;
    videoStreamUrl?: string;
    screenStreamUrl?: string;
}

const Button = ({ children, variant = 'default', size = 'default', className = '', onClick, disabled }: any) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={cn(
            'inline-flex items-center justify-center rounded-lg font-medium transition-colors gap-2',
            variant === 'default' && 'bg-indigo-600 hover:bg-indigo-700 text-white',
            variant === 'outline' && 'border border-white/20 hover:bg-white/10 text-white',
            variant === 'ghost' && 'hover:bg-white/5 text-white',
            size === 'sm' && 'px-3 py-1.5 text-sm',
            size === 'default' && 'px-4 py-2',
            disabled && 'opacity-50 cursor-not-allowed',
            className
        )}
    >
        {children}
    </button>
);

const Progress = ({ value, className = '' }: { value: number; className?: string }) => (
    <div className={cn('w-full bg-white/10 rounded-full overflow-hidden h-2', className)}>
        <div
            className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-300"
            style={{ width: `${value}%` }}
        />
    </div>
);

// Violation Clip Player - paused with play button for manual control
const ViolationClipPlayer = ({ userId, clips, liveFrame }: { userId: string; clips: string[]; liveFrame?: string }) => {
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        if (!clips || clips.length === 0 || !isPlaying) return;
        
        // Loop through clips at ~10 FPS when playing
        const interval = setInterval(() => {
            setCurrentFrame(prev => (prev + 1) % clips.length);
        }, 100);

        return () => clearInterval(interval);
    }, [clips, isPlaying]);

    // If we have violation clips, show with play button; otherwise show live frame
    if (clips && clips.length > 0) {
        return (
            <div className="absolute inset-0 w-full h-full">
                <img
                    src={clips[currentFrame]}
                    alt="Violation clip"
                    className="w-full h-full object-cover"
                />
                <div className="absolute bottom-3 left-3 bg-black/80 backdrop-blur-sm px-3 py-1 rounded-md z-10 text-xs text-red-400 font-bold">
                    VIOLATION RECORDED
                </div>
                {/* Play/Pause Button */}
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-black/70 hover:bg-black/90 backdrop-blur-sm rounded-full p-6 transition-all hover:scale-110 z-20"
                >
                    {isPlaying ? (
                        <Pause className="h-10 w-10 text-white" />
                    ) : (
                        <Play className="h-10 w-10 text-white ml-1" />
                    )}
                </button>
            </div>
        );
    }

    // Fallback to live frame or placeholder
    if (liveFrame) {
        return (
            <img
                src={liveFrame}
                alt="Live student"
                className="absolute inset-0 w-full h-full object-cover"
            />
        );
    }

    return (
        <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
            <Camera className="h-12 w-12 text-slate-600" />
        </div>
    );
};

export default function LiveMonitoringDashboard({
    examId,
    examTitle,
    onBack
}: {
    examId: string;
    examTitle: string;
    onBack: () => void;
}) {
    const [students, setStudents] = useState<LiveStudent[]>([]);
    const [isMonitoring, setIsMonitoring] = useState(true);
    const [currentPage, setCurrentPage] = useState(0);
    const [currentViolationIndex, setCurrentViolationIndex] = useState(0);
    const [selectedStudent, setSelectedStudent] = useState<LiveStudent | null>(null);
    const socketRef = useRef<Socket | null>(null);    const videoFramesRef = useRef<Record<string, string>>({});
    const screenFramesRef = useRef<Record<string, string>>({});
    const violationClipsRef = useRef<Record<string, string[]>>({});    const [stats, setStats] = useState({
        activeExams: 1,
        studentsOnline: 0,
        activeViolations: 0
    });
    const [, forceUpdate] = useState(0);

    const studentsPerPage = 15;
    const totalPages = Math.ceil(students.length / studentsPerPage);
    const currentStudents = students.slice(
        currentPage * studentsPerPage,
        (currentPage + 1) * studentsPerPage
    );

    // Socket.io connection for real-time video/screen streams
    useEffect(() => {
        const baseUrl = API_URL.replace('/api', '');
        const newSocket = io(`${baseUrl}/proctor`, {
            query: { examId, role: 'lecturer' }
        });

        newSocket.on('connect', () => {
            console.log('[DASHBOARD] Connected to proctoring server');
            // Join the exam room to receive frames from students
            newSocket.emit('join_exam', { examId });
        });

        newSocket.on('status', (data: any) => {
            console.log('[DASHBOARD] Status:', data.message);
        });

        newSocket.on('error', (data: any) => {
            console.error('[DASHBOARD] Error:', data.message);
        });

        // Listen for video frames from students
        newSocket.on('video-frame', (data: { userId: string; frame: string; timestamp: number }) => {
            videoFramesRef.current[data.userId] = data.frame;
            // Force UI update for real-time display
            forceUpdate(prev => prev + 1);
        });

        // Listen for screen frames from students
        newSocket.on('screen-frame', (data: { userId: string; frame: string; timestamp: number }) => {
            screenFramesRef.current[data.userId] = data.frame;
            // Force UI update for real-time display
            forceUpdate(prev => prev + 1);
        });

        // Listen for students joining the exam in real-time
        newSocket.on('student-joined', (data: { examId: string; userId: string; studentId: string; name: string; timestamp: string }) => {
            console.log('[DASHBOARD] Student joined:', data);
            // Trigger immediate data refresh when a new student joins
            setIsMonitoring(prev => {
                // Force a re-fetch by toggling and immediately setting back
                setTimeout(() => setIsMonitoring(true), 0);
                return prev;
            });
        });

        socketRef.current = newSocket;

        return () => {
            newSocket.disconnect();
        };
    }, [examId]);

    // Get violations for review panel - only critical and suspicious cases
    const violationReviews = students
        .filter(s => {
            // Only show in review panel if status is critical or suspicious (high-risk violations)
            // This filters out minor warnings and keeps the review panel focused on serious issues
            return (s.status === 'critical' || s.status === 'suspicious') && s.violations > 0;
        })
        .sort((a, b) => {
            const severityOrder = { critical: 4, suspicious: 3, warning: 2, normal: 1 };
            return severityOrder[b.status] - severityOrder[a.status];
        });

    const currentViolation = violationReviews[currentViolationIndex];

    // Polling for real-time data
    useEffect(() => {
        if (!isMonitoring) return;

        const fetchData = async () => {
            try {
                const res = await fetch(`${API_URL}/lecturer/exams/${examId}/monitor`);

                if (res.ok) {
                    const data = await res.json();
                    if (data.students) {
                        const prevStudents = students;
                        setStudents(data.students);
                        
                        // Capture violation clips when violations increase
                        data.students.forEach((student: LiveStudent) => {
                            const prevStudent = prevStudents.find(s => s.userId === student.userId);
                            if (student.violations > 0 && (!prevStudent || student.violations > prevStudent.violations)) {
                                // New violation detected - capture current frame as clip
                                if (videoFramesRef.current[student.userId]) {
                                    if (!violationClipsRef.current[student.userId]) {
                                        violationClipsRef.current[student.userId] = [];
                                    }
                                    // Store the frame (limit to last 30 frames for looping)
                                    violationClipsRef.current[student.userId] = [
                                        ...violationClipsRef.current[student.userId].slice(-29),
                                        videoFramesRef.current[student.userId]
                                    ];
                                }
                            }
                        });
                        
                        setStats(data.stats || {
                            activeExams: 1,
                            studentsOnline: data.students.length,
                            activeViolations: data.students.filter((s: LiveStudent) => s.violations > 0).length
                        });
                    }
                }
            } catch (err) {
                console.error('Monitoring poll error:', err);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 3000);
        return () => clearInterval(interval);
    }, [examId, isMonitoring, students]);

    const formatTime = (seconds: number) => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = seconds % 60;
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'normal': return 'text-green-400';
            case 'warning': return 'text-yellow-400';
            case 'suspicious': return 'text-orange-400';
            case 'critical': return 'text-red-400';
            default: return 'text-gray-400';
        }
    };

    const getSeverityBg = (severity: string) => {
        switch (severity) {
            case 'normal': return 'bg-green-500/10 border-green-500/20';
            case 'warning': return 'bg-yellow-500/10 border-yellow-500/20';
            case 'suspicious': return 'bg-orange-500/10 border-orange-500/20';
            case 'critical': return 'bg-red-500/10 border-red-500/20';
            default: return 'bg-gray-500/10 border-gray-500/20';
        }
    };

    const getBorderColor = (severity: string) => {
        switch (severity) {
            case 'normal': return 'border-green-500/30 hover:border-green-500/60';
            case 'warning': return 'border-yellow-500/30 hover:border-yellow-500/60';
            case 'suspicious': return 'border-orange-500/30 hover:border-orange-500/60';
            case 'critical': return 'border-red-500/50 hover:border-red-500/80 animate-pulse';
            default: return 'border-gray-500/30';
        }
    };

    const handleNext = () => {
        setCurrentViolationIndex((prev) => (prev + 1) % violationReviews.length);
    };

    const handlePrevious = () => {
        setCurrentViolationIndex((prev) => (prev - 1 + violationReviews.length) % violationReviews.length);
    };

    const complianceRate = students.length > 0
        ? Math.round(((students.length - violationReviews.length) / students.length) * 100)
        : 100;

    const statusCounts = {
        normal: students.filter(s => s.status === 'normal').length,
        warning: students.filter(s => s.status === 'warning').length,
        suspicious: students.filter(s => s.status === 'suspicious').length,
        critical: students.filter(s => s.status === 'critical').length
    };

    // Calculate additional stats
    const averageRisk = students.length > 0
        ? Math.round(students.reduce((sum, s) => sum + (s.violations * 10), 0) / students.length)
        : 0;
    
    const idVerified = students.filter(s => s.faceDetected).length;
    const timeLeft = students.length > 0 ? students[0].timeRemaining : 0;

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
            {/* Top Navigation & Global Stats */}
            <div className="border-b border-white/10 bg-slate-900/80 backdrop-blur-xl sticky top-0 z-50 shadow-2xl">
                <div className="px-6 py-4">
                    {/* Lecturer Info Bar */}
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-4">
                            <button onClick={onBack} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                                <ArrowLeft className="h-5 w-5" />
                            </button>
                            <div className="flex items-center gap-3 border-r border-white/20 pr-4">
                                <Camera className="h-8 w-8 text-indigo-400" />
                                <div>
                                    <h1 className="text-lg font-bold">INVIGILO Live Proctoring</h1>
                                    <p className="text-xs text-gray-400">Computer Science Department</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2 bg-green-500/20 text-green-400 border border-green-500/30 px-3 py-1.5 rounded-lg animate-pulse">
                                <Radio className="h-4 w-4" />
                                <span className="text-sm font-bold">LIVE</span>
                            </div>
                        </div>

                        <div className="flex items-center gap-3">
                            <Button variant="outline" size="sm">
                                <Filter className="h-4 w-4" />
                            </Button>
                            <Button variant="outline" size="sm">
                                <Download className="h-4 w-4" />
                                Export
                            </Button>
                        </div>
                    </div>

                    {/* Metric Cards - 7 Cards */}
                    <div className="grid grid-cols-7 gap-3">
                        <div className="bg-gradient-to-br from-green-500/20 to-green-600/10 border border-green-500/30 rounded-xl p-3">
                            <div className="flex items-center gap-2 mb-1">
                                <CheckCircle className="h-4 w-4 text-green-400" />
                                <span className="text-xs text-green-300 font-medium">Online</span>
                            </div>
                            <p className="text-2xl font-bold text-white">{stats.studentsOnline}/{students.length}</p>
                        </div>

                        <div className="bg-gradient-to-br from-green-500/20 to-green-600/10 border border-green-500/30 rounded-xl p-3">
                            <div className="flex items-center gap-2 mb-1">
                                <ScanFace className="h-4 w-4 text-green-400" />
                                <span className="text-xs text-green-300 font-medium">ID Verified</span>
                            </div>
                            <p className="text-2xl font-bold text-white">{idVerified}</p>
                        </div>

                        <div className="bg-gradient-to-br from-orange-500/20 to-orange-600/10 border border-orange-500/30 rounded-xl p-3">
                            <div className="flex items-center gap-2 mb-1">
                                <Activity className="h-4 w-4 text-orange-400" />
                                <span className="text-xs text-orange-300 font-medium">Time Left</span>
                            </div>
                            <p className="text-xl font-bold text-white font-mono">{formatTime(timeLeft)}</p>
                        </div>

                        <div className="bg-gradient-to-br from-red-500/20 to-red-600/10 border border-red-500/30 rounded-xl p-3">
                            <div className="flex items-center gap-2 mb-1">
                                <AlertTriangle className="h-4 w-4 text-red-400" />
                                <span className="text-xs text-red-300 font-medium">Violations</span>
                            </div>
                            <p className="text-2xl font-bold text-white">{stats.activeViolations}</p>
                        </div>

                        <div className="bg-gradient-to-br from-yellow-500/20 to-yellow-600/10 border border-yellow-500/30 rounded-xl p-3">
                            <div className="flex items-center gap-2 mb-1">
                                <TrendingUp className="h-4 w-4 text-yellow-400" />
                                <span className="text-xs text-yellow-300 font-medium">Avg Risk</span>
                            </div>
                            <p className="text-2xl font-bold text-white">{averageRisk}%</p>
                        </div>

                        <div className="bg-gradient-to-br from-cyan-500/20 to-cyan-600/10 border border-cyan-500/30 rounded-xl p-3">
                            <div className="flex items-center gap-2 mb-1">
                                <Eye className="h-4 w-4 text-cyan-400" />
                                <span className="text-xs text-cyan-300 font-medium">Compliance</span>
                            </div>
                            <p className="text-2xl font-bold text-white">{complianceRate}%</p>
                        </div>

                        <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/10 border border-purple-500/30 rounded-xl p-3">
                            <div className="flex items-center gap-2 mb-1">
                                <FileText className="h-4 w-4 text-purple-400" />
                                <span className="text-xs text-purple-300 font-medium">Exam</span>
                            </div>
                            <p className="text-sm font-bold text-white truncate">{examTitle.substring(0, 8)}</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content - Grid + Focus Panel */}
            <div className="p-6 flex gap-6">
                {/* Student Monitoring Grid - 75% width */}
                <div className="flex-[0.75]">
                    <div className="mb-4">
                        <h2 className="text-xl font-bold text-white mb-1">Student Monitoring Grid</h2>
                        <p className="text-sm text-gray-400">Real-time surveillance of {currentStudents.length} active students</p>
                    </div>

                    <div className="grid grid-cols-5 gap-4 mb-6">
                        {currentStudents.map((student, idx) => {
                            const riskScore = Math.min(100, student.violations * 15);
                            const getBorderClass = () => {
                                if (riskScore >= 70 || student.status === 'critical') return 'border-red-500/70 shadow-red-500/30';
                                if (riskScore >= 40 || student.status === 'suspicious') return 'border-orange-500/70 shadow-orange-500/30';
                                if (riskScore >= 20 || student.status === 'warning') return 'border-yellow-500/70 shadow-yellow-500/30';
                                return 'border-green-500/50 shadow-green-500/20';
                            };

                            const getStatusText = () => {
                                if (student.status === 'critical' || !student.faceDetected) return 'Critical';
                                if (student.status === 'suspicious') return 'Suspicious';
                                if (student.status === 'warning') return 'Warning';
                                return 'Compliant';
                            };

                            const getStatusBgClass = () => {
                                if (student.status === 'critical' || !student.faceDetected) return 'bg-red-500 text-white';
                                if (student.status === 'suspicious') return 'bg-orange-500 text-white';
                                if (student.status === 'warning') return 'bg-yellow-500 text-black';
                                return 'bg-green-500 text-white';
                            };

                            return (
                                <motion.div
                                    key={student.userId}
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: idx * 0.03 }}
                                    whileHover={{ scale: 1.03, zIndex: 10 }}
                                    onDoubleClick={() => setSelectedStudent(student)}
                                    className={cn(
                                        'bg-slate-900/60 backdrop-blur-sm border-3 rounded-xl p-3 transition-all duration-300 cursor-pointer shadow-xl',
                                        getBorderClass()
                                    )}
                                >
                                    {/* Video Feed Area */}
                                    <div className="aspect-[4/3] bg-black rounded-lg mb-3 relative overflow-hidden">
                                        {/* AI Badge - Top Left */}
                                        <div className="absolute top-2 left-2 bg-purple-600 text-white px-2 py-1 rounded-md flex items-center gap-1 z-20 text-xs font-bold">
                                            <Activity className="h-3 w-3" />
                                            AI
                                        </div>

                                        {/* Risk Score - Top Right */}
                                        <div className={cn(
                                            'absolute top-2 right-2 px-2 py-1 rounded-md z-20 text-sm font-bold',
                                            riskScore >= 70 && 'bg-red-500 text-white',
                                            riskScore >= 40 && riskScore < 70 && 'bg-orange-500 text-white',
                                            riskScore >= 20 && riskScore < 40 && 'bg-yellow-500 text-black',
                                            riskScore < 20 && 'bg-green-500 text-white'
                                        )}>
                                            {riskScore}%
                                        </div>

                                        {/* No Face Overlay */}
                                        {!student.faceDetected && (
                                            <div className="absolute inset-0 bg-red-500/30 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                                                <XCircle className="h-10 w-10 text-red-400 mb-2" />
                                                <span className="text-sm font-bold text-red-400">No Face</span>
                                            </div>
                                        )}

                                        {/* Video Stream */}
                                        {videoFramesRef.current[student.userId] ? (
                                            <img
                                                src={videoFramesRef.current[student.userId]}
                                                alt="Student video"
                                                className="absolute inset-0 w-full h-full object-cover"
                                            />
                                        ) : (
                                            <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
                                                <Camera className="h-8 w-8 text-slate-600" />
                                            </div>
                                        )}

                                        {/* Face Detection Border */}
                                        {student.faceDetected && (
                                            <div className={cn(
                                                'absolute inset-4 border-2 rounded-md transition-colors',
                                                riskScore >= 70 && 'border-red-500',
                                                riskScore >= 40 && riskScore < 70 && 'border-orange-500',
                                                riskScore >= 20 && riskScore < 40 && 'border-yellow-500',
                                                riskScore < 20 && 'border-green-500'
                                            )} />
                                        )}
                                    </div>

                                    {/* Student Info */}
                                    <div className="mb-2">
                                        <p className="text-sm text-white font-bold truncate">{student.name}</p>
                                        <p className="text-xs text-gray-400">{student.studentId}</p>
                                    </div>

                                    {/* Status Bar */}
                                    <div className={cn(
                                        'w-full py-1.5 rounded-md text-center text-xs font-bold',
                                        getStatusBgClass()
                                    )}>
                                        {getStatusText()}
                                    </div>
                                </motion.div>
                            );
                        })}
                    </div>

                    {/* Pagination */}
                    {totalPages > 1 && (
                        <div className="flex items-center justify-center gap-4">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                                disabled={currentPage === 0}
                            >
                                <ChevronLeft className="h-4 w-4" />
                            </Button>
                            <span className="text-sm text-gray-400">
                                Page {currentPage + 1} of {totalPages}
                            </span>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
                                disabled={currentPage === totalPages - 1}
                            >
                                <ChevronRight className="h-4 w-4" />
                            </Button>
                        </div>
                    )}
                </div>

                {/* Active Review Panel - 25% width - Focused Review */}
                <div className="flex-[0.25]">
                    <div className="bg-slate-900/80 backdrop-blur-xl border-2 border-white/20 rounded-2xl p-5 sticky top-32 shadow-2xl">
                        <div className="flex items-center gap-2 mb-5">
                            <AlertTriangle className="h-6 w-6 text-red-400" />
                            <h3 className="text-xl font-bold text-white">Active Review</h3>
                        </div>

                        {violationReviews.length === 0 ? (
                            <div className="text-center py-16">
                                <CheckCircle className="h-20 w-20 text-green-400 mx-auto mb-4" />
                                <p className="text-green-400 font-bold text-lg mb-2">All Clear!</p>
                                <p className="text-sm text-gray-400">No violations detected</p>
                            </div>
                        ) : (
                            <AnimatePresence mode="wait">
                                <motion.div
                                    key={currentViolationIndex}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    className="space-y-4"
                                >
                                    {/* Primary Feed with Thick Red Border */}
                                    <div className="relative">
                                        <div className="aspect-[4/3] bg-black rounded-xl border-4 border-red-500 relative overflow-hidden shadow-xl shadow-red-500/30">
                                            {/* REVIEWING Tag */}
                                            <div className="absolute top-3 left-3 bg-red-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 z-20 font-bold text-sm">
                                                <div className="w-2.5 h-2.5 rounded-full bg-white animate-pulse" />
                                                REVIEWING
                                            </div>

                                            {/* Violations Alert */}
                                            <div className="absolute top-3 right-3 bg-red-600 text-white px-4 py-2 rounded-lg z-20 font-bold text-sm">
                                                {currentViolation.violations} VIOLATIONS
                                            </div>

                                            {/* Looping Violation Clip */}
                                            <ViolationClipPlayer 
                                                userId={currentViolation.userId}
                                                clips={violationClipsRef.current[currentViolation.userId] || []}
                                                liveFrame={videoFramesRef.current[currentViolation.userId]}
                                            />
                                        </div>
                                    </div>

                                    {/* Risk Breakdown */}
                                    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4">
                                        <div className="flex items-center justify-between mb-3">
                                            <span className="text-sm font-semibold text-gray-300">Risk Score</span>
                                            <span className="text-2xl font-bold text-red-400">
                                                {Math.min(100, currentViolation.violations * 15)}%
                                            </span>
                                        </div>
                                        <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-red-600 to-red-400 transition-all duration-500"
                                                style={{ width: `${Math.min(100, currentViolation.violations * 15)}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Student Details */}
                                    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4">
                                        <div className="flex items-start justify-between mb-2">
                                            <div>
                                                <p className="text-white font-bold text-lg">{currentViolation.name}</p>
                                                <p className="text-sm text-gray-400">{currentViolation.studentId}</p>
                                            </div>
                                            <span className={cn(
                                                'px-3 py-1 rounded-lg text-xs font-bold uppercase',
                                                currentViolation.status === 'critical' && 'bg-red-500 text-white',
                                                currentViolation.status === 'suspicious' && 'bg-orange-500 text-white',
                                                currentViolation.status === 'warning' && 'bg-yellow-500 text-black'
                                            )}>
                                                {currentViolation.status}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Specific Triggers */}
                                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                                        <p className="text-xs font-semibold text-red-300 mb-3 uppercase">Critical Violations Detected</p>
                                        <div className="space-y-2">
                                            {currentViolation.latestViolation && (
                                                <div className="flex items-start gap-2 text-sm">
                                                    <AlertTriangle className="h-4 w-4 text-red-400 mt-0.5 flex-shrink-0" />
                                                    <span className="text-gray-300">{currentViolation.latestViolation}</span>
                                                </div>
                                            )}
                                            {!currentViolation.faceDetected && (
                                                <div className="flex items-start gap-2 text-sm">
                                                    <XCircle className="h-4 w-4 text-red-400 mt-0.5 flex-shrink-0" />
                                                    <span className="text-gray-300">Face not detected</span>
                                                </div>
                                            )}
                                            {currentViolation.gaze !== 'forward' && (
                                                <div className="flex items-start gap-2 text-sm">
                                                    <Eye className="h-4 w-4 text-orange-400 mt-0.5 flex-shrink-0" />
                                                    <span className="text-gray-300">Suspicious gaze direction</span>
                                                </div>
                                            )}
                                            {currentViolation.headPose !== 'normal' && (
                                                <div className="flex items-start gap-2 text-sm">
                                                    <Activity className="h-4 w-4 text-orange-400 mt-0.5 flex-shrink-0" />
                                                    <span className="text-gray-300">Abnormal head pose</span>
                                                </div>
                                            )}
                                        </div>
                                        {currentViolation.violationTime && (
                                            <p className="text-xs text-gray-500 mt-3">{currentViolation.violationTime}</p>
                                        )}
                                    </div>

                                    {/* Review Navigation - Prominent */}
                                    <div className="flex items-center justify-between bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-3 border border-white/20">
                                        <button
                                            onClick={handlePrevious}
                                            className="p-2 hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                            disabled={violationReviews.length <= 1}
                                        >
                                            <ChevronLeft className="h-6 w-6 text-white" />
                                        </button>
                                        <div className="text-center">
                                            <p className="text-sm font-bold text-white">
                                                Review {currentViolationIndex + 1} of {violationReviews.length}
                                            </p>
                                            <p className="text-xs text-gray-400">Navigate between flagged students</p>
                                        </div>
                                        <button
                                            onClick={handleNext}
                                            className="p-2 hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                            disabled={violationReviews.length <= 1}
                                        >
                                            <ChevronRight className="h-6 w-6 text-white" />
                                        </button>
                                    </div>

                                    {/* Action Buttons - 3 Large Buttons */}
                                    <div className="grid grid-cols-3 gap-2">
                                        <button
                                            onClick={() => {
                                                alert(`Flagged ${currentViolation.name} for review`);
                                            }}
                                            className="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-4 rounded-xl transition-all duration-200 hover:scale-105 flex flex-col items-center justify-center gap-1 shadow-lg"
                                        >
                                            <AlertTriangle className="h-5 w-5" />
                                            <span className="text-xs">Flag</span>
                                        </button>

                                        <button
                                            onClick={() => {
                                                if (confirm(`Dismiss ${currentViolation.name}'s session?`)) {
                                                    alert(`Dismissed ${currentViolation.name}`);
                                                }
                                            }}
                                            className="bg-red-600 hover:bg-red-700 text-white font-bold py-4 rounded-xl transition-all duration-200 hover:scale-105 flex flex-col items-center justify-center gap-1 shadow-lg"
                                        >
                                            <XCircle className="h-5 w-5" />
                                            <span className="text-xs">Dismiss</span>
                                        </button>

                                        <button
                                            onClick={() => {
                                                alert(`Approved ${currentViolation.name} to continue`);
                                            }}
                                            className="bg-green-600 hover:bg-green-700 text-white font-bold py-4 rounded-xl transition-all duration-200 hover:scale-105 flex flex-col items-center justify-center gap-1 shadow-lg"
                                        >
                                            <CheckCircle className="h-5 w-5" />
                                            <span className="text-xs">Approve</span>
                                        </button>
                                    </div>
                                </motion.div>
                            </AnimatePresence>
                        )}
                    </div>
                </div>
            </div>

            {/* Student Detail Modal */}
            {selectedStudent && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 bg-black/80 backdrop-blur-md z-[100] flex items-center justify-center p-6"
                    onClick={() => setSelectedStudent(null)}
                >
                    <motion.div
                        initial={{ scale: 0.9, y: 20 }}
                        animate={{ scale: 1, y: 0 }}
                        exit={{ scale: 0.9, y: 20 }}
                        onClick={(e) => e.stopPropagation()}
                        className="bg-gradient-to-br from-slate-900 to-slate-800 border border-white/20 rounded-3xl p-6 w-full max-w-6xl max-h-[90vh] overflow-y-auto"
                    >
                        {/* Modal Header */}
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h2 className="text-2xl font-bold text-white mb-1">{selectedStudent.name}</h2>
                                <p className="text-gray-400">{selectedStudent.studentId}</p>
                            </div>

                            <div className="flex items-center gap-3">
                                {/* Status Badge */}
                                <div className={cn(
                                    'px-4 py-2 rounded-xl font-semibold text-sm border',
                                    getSeverityBg(selectedStudent.status),
                                    selectedStudent.status === 'critical' && 'animate-pulse'
                                )}>
                                    {selectedStudent.status.toUpperCase()}
                                </div>

                                {/* Close Button */}
                                <button
                                    onClick={() => setSelectedStudent(null)}
                                    className="p-2 hover:bg-white/10 rounded-xl transition-colors"
                                >
                                    <X className="h-6 w-6 text-white" />
                                </button>
                            </div>
                        </div>

                        {/* Video Streams Grid */}
                        <div className="grid grid-cols-2 gap-6 mb-6">
                            {/* Camera Feed */}
                            <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                    <Camera className="h-5 w-5 text-indigo-400" />
                                    <h3 className="text-lg font-semibold text-white">Camera Feed</h3>
                                    <div className="flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-md">
                                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                        <span className="text-xs font-semibold text-white">LIVE</span>
                                    </div>
                                </div>

                                <div className="aspect-video bg-black rounded-2xl relative overflow-hidden border-2 border-white/10">
                                    {/* Face Detection Box */}
                                    {selectedStudent.faceDetected && (
                                        <div className={cn(
                                            'absolute inset-8 border-4 rounded-lg transition-colors z-10',
                                            selectedStudent.status === 'normal' && 'border-green-500',
                                            selectedStudent.status === 'warning' && 'border-yellow-500',
                                            selectedStudent.status === 'suspicious' && 'border-orange-500',
                                            selectedStudent.status === 'critical' && 'border-red-500'
                                        )} />
                                    )}

                                    {/* No Face Warning */}
                                    {!selectedStudent.faceDetected && (
                                        <div className="absolute inset-0 bg-red-500/20 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                                            <XCircle className="h-16 w-16 text-red-400 mb-3" />
                                            <span className="text-xl font-semibold text-red-400">No Face Detected</span>
                                        </div>
                                    )}

                                    {/* Video Stream or Placeholder */}
                                    {videoFramesRef.current[selectedStudent.userId] ? (
                                        <img
                                            src={videoFramesRef.current[selectedStudent.userId]}
                                            alt="Student camera"
                                            className="absolute inset-0 w-full h-full object-cover"
                                        />
                                    ) : (
                                        <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
                                            <Camera className="h-16 w-16 text-slate-600" />
                                        </div>
                                    )}
                                </div>

                                {/* Camera Status Indicators */}
                                <div className="grid grid-cols-3 gap-3">
                                    <div className={cn(
                                        'flex items-center gap-2 px-4 py-3 rounded-xl',
                                        selectedStudent.faceDetected ? 'bg-green-500/20 border border-green-500/30' : 'bg-red-500/20 border border-red-500/30'
                                    )}>
                                        <ScanFace className="h-5 w-5" style={{ color: selectedStudent.faceDetected ? '#10b981' : '#ef4444' }} />
                                        <div>
                                            <p className="text-xs text-gray-400">Face</p>
                                            <p className="text-sm font-semibold" style={{ color: selectedStudent.faceDetected ? '#10b981' : '#ef4444' }}>
                                                {selectedStudent.faceDetected ? 'Detected' : 'Missing'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className={cn(
                                        'flex items-center gap-2 px-4 py-3 rounded-xl',
                                        selectedStudent.gaze === 'forward' ? 'bg-green-500/20 border border-green-500/30' : 'bg-yellow-500/20 border border-yellow-500/30'
                                    )}>
                                        <Eye className="h-5 w-5" style={{ color: selectedStudent.gaze === 'forward' ? '#10b981' : '#eab308' }} />
                                        <div>
                                            <p className="text-xs text-gray-400">Gaze</p>
                                            <p className="text-sm font-semibold capitalize" style={{ color: selectedStudent.gaze === 'forward' ? '#10b981' : '#eab308' }}>
                                                {selectedStudent.gaze}
                                            </p>
                                        </div>
                                    </div>

                                    <div className={cn(
                                        'flex items-center gap-2 px-4 py-3 rounded-xl',
                                        selectedStudent.headPose === 'normal' ? 'bg-green-500/20 border border-green-500/30' : 'bg-orange-500/20 border border-orange-500/30'
                                    )}>
                                        <Activity className="h-5 w-5" style={{ color: selectedStudent.headPose === 'normal' ? '#10b981' : '#f97316' }} />
                                        <div>
                                            <p className="text-xs text-gray-400">Pose</p>
                                            <p className="text-sm font-semibold capitalize" style={{ color: selectedStudent.headPose === 'normal' ? '#10b981' : '#f97316' }}>
                                                {selectedStudent.headPose}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Screen Share */}
                            <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                    <Monitor className="h-5 w-5 text-violet-400" />
                                    <h3 className="text-lg font-semibold text-white">Screen Share</h3>
                                    <div className="flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-md">
                                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                                        <span className="text-xs font-semibold text-white">ACTIVE</span>
                                    </div>
                                </div>

                                <div className="aspect-video bg-black rounded-2xl relative overflow-hidden border-2 border-white/10">
                                    {/* Screen Stream or Placeholder */}
                                    {screenFramesRef.current[selectedStudent.userId] ? (
                                        <img
                                            src={screenFramesRef.current[selectedStudent.userId]}
                                            alt="Student screen"
                                            className="absolute inset-0 w-full h-full object-contain"
                                        />
                                    ) : (
                                        <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
                                            <Monitor className="h-16 w-16 text-slate-600" />
                                        </div>
                                    )}
                                </div>

                                {/* Screen Activity Info */}
                                <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <p className="text-xs text-gray-400 mb-1">Time Remaining</p>
                                            <p className="text-2xl font-mono font-bold text-white">{formatTime(selectedStudent.timeRemaining)}</p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-gray-400 mb-1">Violations</p>
                                            <p className="text-2xl font-mono font-bold text-red-400">{selectedStudent.violations}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Violation History */}
                        {selectedStudent.latestViolation && (
                            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                                <div className="flex items-start gap-3">
                                    <AlertTriangle className="h-5 w-5 text-red-400 mt-0.5" />
                                    <div className="flex-1">
                                        <p className="font-semibold text-red-400 mb-1">Latest Violation</p>
                                        <p className="text-sm text-gray-300">{selectedStudent.latestViolation}</p>
                                        <p className="text-xs text-gray-500 mt-1">{selectedStudent.violationTime}</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex gap-3 mt-6">
                            <Button
                                variant="outline"
                                className="flex-1 bg-yellow-600 hover:bg-yellow-700"
                                onClick={() => {
                                    socketRef.current?.emit('pause_student', {
                                        examId,
                                        userId: selectedStudent.userId
                                    });
                                    alert(`Paused exam for ${selectedStudent.name}`);
                                }}
                            >
                                <Pause className="h-4 w-4" />
                                Pause Exam
                            </Button>

                            <Button
                                variant="destructive"
                                className="flex-1 bg-red-600 hover:bg-red-700"
                                onClick={() => {
                                    if (confirm(`Are you sure you want to stop the exam for ${selectedStudent.name}? This cannot be undone.`)) {
                                        socketRef.current?.emit('stop_student', {
                                            examId,
                                            userId: selectedStudent.userId
                                        });
                                        alert(`Stopped exam for ${selectedStudent.name}`);
                                    }
                                }}
                            >
                                <X className="h-4 w-4" />
                                Stop Exam
                            </Button>

                            <Button
                                variant="outline"
                                className="flex-1 bg-green-600 hover:bg-green-700"
                                onClick={() => {
                                    socketRef.current?.emit('allow_student', {
                                        examId,
                                        userId: selectedStudent.userId
                                    });
                                    alert(`Allowed ${selectedStudent.name} to continue`);
                                }}
                            >
                                <CheckCircle className="h-4 w-4" />
                                Allow Continue
                            </Button>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </div>
    );
}
