import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Camera, Eye, Activity, ScanFace, AlertTriangle, XCircle, CheckCircle,
    Radio, ChevronLeft, ChevronRight, TrendingUp, Settings, Download,
    Filter, Bell, Plus, FileText, Users as UsersIcon, ArrowLeft, Play, Pause
} from 'lucide-react';

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
    const [stats, setStats] = useState({
        activeExams: 1,
        studentsOnline: 0,
        activeViolations: 0
    });

    const studentsPerPage = 15;
    const totalPages = Math.ceil(students.length / studentsPerPage);
    const currentStudents = students.slice(
        currentPage * studentsPerPage,
        (currentPage + 1) * studentsPerPage
    );

    // Get violations for review panel
    const violationReviews = students
        .filter(s => s.violations > 0)
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
                const res = await fetch(`${API_URL}/lecturer/exams/${examId}/monitor`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                if (res.ok) {
                    const data = await res.json();
                    if (data.students) {
                        setStudents(data.students);
                        setStats({
                            activeExams: 1,
                            studentsOnline: data.students.filter((s: LiveStudent) => s.faceDetected).length,
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
    }, [examId, isMonitoring]);

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

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
            {/* Header Bar */}
            <div className="border-b border-white/10 bg-white/5 backdrop-blur-xl sticky top-0 z-50">
                <div className="p-4">
                    <div className="flex items-center justify-between mb-4">
                        {/* Branding */}
                        <div className="flex items-center gap-4">
                            <button onClick={onBack} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                                <ArrowLeft className="h-5 w-5" />
                            </button>
                            <div className="flex items-center gap-3">
                                <Camera className="h-8 w-8 text-indigo-400" />
                                <div>
                                    <h1 className="text-xl font-bold">INVIGILO</h1>
                                    <span className="text-xs text-gray-400">Lecturer</span>
                                </div>
                            </div>
                        </div>

                        {/* Stats Panel */}
                        <div className="flex items-center gap-4">
                            <div className="bg-white/5 rounded-xl border border-white/10 px-4 py-2 flex items-center gap-2">
                                <FileText className="h-4 w-4 text-indigo-400" />
                                <div>
                                    <p className="text-xs text-gray-400">Active Exams</p>
                                    <p className="text-white font-semibold">{stats.activeExams}</p>
                                </div>
                            </div>

                            <div className="bg-white/5 rounded-xl border border-white/10 px-4 py-2 flex items-center gap-2">
                                <UsersIcon className="h-4 w-4 text-green-400" />
                                <div>
                                    <p className="text-xs text-gray-400">Students Online</p>
                                    <p className="text-white font-semibold">{stats.studentsOnline}</p>
                                </div>
                            </div>

                            <div className="bg-white/5 rounded-xl border border-white/10 px-4 py-2 flex items-center gap-2">
                                <AlertTriangle className="h-4 w-4 text-red-400" />
                                <div>
                                    <p className="text-xs text-gray-400">Active Violations</p>
                                    <p className="text-white font-semibold">{stats.activeViolations}</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Title & Actions */}
                    <div className="flex items-center justify-between">
                        <div>
                            <h2 className="text-2xl font-bold text-white">Live Monitoring Dashboard</h2>
                            <p className="text-gray-400 text-sm mt-1">{examTitle} • Real-time Student Surveillance</p>
                        </div>

                        <div className="flex items-center gap-3">
                            <button className="bg-green-500/20 text-green-400 border border-green-500/30 px-4 py-2 rounded-lg flex items-center gap-2 animate-pulse">
                                <Radio className="h-4 w-4" />
                                <span className="font-semibold">LIVE PROCTORING</span>
                            </button>

                            <Button variant="outline" size="sm">
                                <Filter className="h-4 w-4" />
                                Filter
                            </Button>

                            <Button variant="outline" size="sm">
                                <Download className="h-4 w-4" />
                                Export Report
                            </Button>

                            <Button
                                variant={isMonitoring ? 'ghost' : 'default'}
                                size="sm"
                                onClick={() => setIsMonitoring(!isMonitoring)}
                            >
                                {isMonitoring ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                                {isMonitoring ? 'Pause' : 'Resume'}
                            </Button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="p-6 flex gap-6">
                {/* Student Grid - 80% */}
                <div className="flex-[0.78]">
                    <div className="grid grid-cols-5 gap-4 mb-6">
                        {currentStudents.map((student, idx) => (
                            <motion.div
                                key={student.userId}
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: idx * 0.05 }}
                                whileHover={{ scale: 1.05, zIndex: 50 }}
                                className={cn(
                                    'bg-white/5 backdrop-blur-xl border-2 rounded-2xl p-3 transition-all duration-300',
                                    getBorderColor(student.status)
                                )}
                            >
                                {/* Video Feed */}
                                <div className="aspect-[4/3] bg-black rounded-xl mb-3 relative overflow-hidden">
                                    {/* LIVE Indicator */}
                                    <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-md flex items-center gap-1.5 z-10">
                                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                        <span className="text-xs font-semibold text-white">LIVE</span>
                                    </div>

                                    {/* Violation Count Badge */}
                                    {student.violations > 0 && (
                                        <div className={cn(
                                            'absolute top-2 right-2 px-2 py-1 rounded-md backdrop-blur-sm z-10 text-xs font-bold',
                                            student.status === 'critical' && 'bg-red-500/90 text-white',
                                            student.status === 'suspicious' && 'bg-orange-500/80 text-white',
                                            student.status === 'warning' && 'bg-yellow-500/80 text-black'
                                        )}>
                                            {student.violations}
                                        </div>
                                    )}

                                    {/* Face Detection Box */}
                                    {student.faceDetected && (
                                        <div className={cn(
                                            'absolute inset-3 border-2 rounded-lg transition-colors',
                                            student.status === 'normal' && 'border-green-500',
                                            student.status === 'warning' && 'border-yellow-500',
                                            student.status === 'suspicious' && 'border-orange-500',
                                            student.status === 'critical' && 'border-red-500'
                                        )} />
                                    )}

                                    {/* No Face Warning */}
                                    {!student.faceDetected && (
                                        <div className="absolute inset-0 bg-red-500/20 backdrop-blur-sm flex flex-col items-center justify-center">
                                            <XCircle className="h-8 w-8 text-red-400 mb-2" />
                                            <span className="text-xs font-semibold text-red-400">No Face</span>
                                        </div>
                                    )}

                                    {/* Timer */}
                                    <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-md z-10">
                                        <span className="text-xs font-mono text-white">{formatTime(student.timeRemaining)}</span>
                                    </div>

                                    {/* Placeholder for actual video */}
                                    <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900" />
                                </div>

                                {/* Student Info */}
                                <div className="mb-2">
                                    <p className="text-sm text-white truncate font-semibold">{student.name}</p>
                                    <p className="text-xs text-gray-400">{student.studentId}</p>
                                </div>

                                {/* Status Indicators Grid */}
                                <div className="grid grid-cols-3 gap-1 text-xs mb-2">
                                    <div className={cn(
                                        'flex items-center gap-1 px-2 py-1 rounded',
                                        student.faceDetected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                                    )}>
                                        <ScanFace className="h-3 w-3" />
                                        <span className="text-[10px]">Face</span>
                                    </div>

                                    <div className={cn(
                                        'flex items-center gap-1 px-2 py-1 rounded',
                                        student.gaze === 'forward' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
                                    )}>
                                        <Eye className="h-3 w-3" />
                                        <span className="text-[10px]">Gaze</span>
                                    </div>

                                    <div className={cn(
                                        'flex items-center gap-1 px-2 py-1 rounded',
                                        student.headPose === 'normal' ? 'bg-green-500/20 text-green-400' : 'bg-orange-500/20 text-orange-400'
                                    )}>
                                        <Activity className="h-3 w-3" />
                                        <span className="text-[10px]">Pose</span>
                                    </div>
                                </div>

                                {/* Overall Status Badge */}
                                <div className={cn(
                                    'mt-2 px-2 py-1 rounded-lg text-center text-xs font-semibold border',
                                    getSeverityBg(student.status),
                                    getSeverityColor(student.status)
                                )}>
                                    {student.status === 'normal' && '✓ Normal'}
                                    {student.status === 'warning' && '⚠ Warning'}
                                    {student.status === 'suspicious' && '⚠ Suspicious'}
                                    {student.status === 'critical' && '✕ Critical'}
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    {/* Pagination */}
                    {totalPages > 1 && (
                        <div className="flex items-center justify-center gap-4 mt-6">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                                disabled={currentPage === 0}
                            >
                                <ChevronLeft className="h-4 w-4" />
                                Previous
                            </Button>
                            <span className="text-sm text-gray-400">
                                Page {currentPage + 1} of {totalPages} • Showing {currentStudents.length} of {students.length} students
                            </span>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
                                disabled={currentPage === totalPages - 1}
                            >
                                Next
                                <ChevronRight className="h-4 w-4" />
                            </Button>
                        </div>
                    )}
                </div>

                {/* Violation Review Panel - 20% */}
                <div className="flex-[0.22]">
                    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 sticky top-24">
                        <div className="flex items-center gap-2 mb-4">
                            <AlertTriangle className="h-5 w-5 text-red-400" />
                            <h3 className="text-lg font-bold text-white">Violation Review</h3>
                        </div>

                        {violationReviews.length === 0 ? (
                            <div className="text-center py-12">
                                <CheckCircle className="h-16 w-16 text-green-400 mx-auto mb-4" />
                                <p className="text-green-400 font-semibold mb-1">No violations detected</p>
                                <p className="text-xs text-gray-400">All students are compliant</p>
                            </div>
                        ) : (
                            <AnimatePresence mode="wait">
                                <motion.div
                                    key={currentViolationIndex}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                >
                                    {/* Large Video Feed */}
                                    <div className="aspect-[4/3] bg-black rounded-xl border-2 border-red-500/50 mb-4 relative overflow-hidden">
                                        <div className="absolute top-2 left-2 bg-red-500/80 backdrop-blur-sm px-3 py-2 rounded-lg flex items-center gap-2 z-10">
                                            <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                                            <span className="text-xs font-bold text-white">REVIEWING</span>
                                        </div>

                                        <div className="absolute top-2 right-2 bg-red-500/90 text-white px-3 py-2 rounded-lg z-10">
                                            <span className="text-xs font-bold">{currentViolation.violations} VIOLATIONS</span>
                                        </div>

                                        <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900" />
                                    </div>

                                    {/* Student Details */}
                                    <div className="bg-white/5 rounded-xl p-4 mb-4">
                                        <div className="flex items-start justify-between mb-2">
                                            <div>
                                                <p className="text-white font-semibold">{currentViolation.name}</p>
                                                <p className="text-xs text-gray-400">{currentViolation.studentId}</p>
                                            </div>
                                            <span className={cn(
                                                'px-2 py-1 rounded text-xs font-semibold',
                                                getSeverityBg(currentViolation.status),
                                                getSeverityColor(currentViolation.status)
                                            )}>
                                                {currentViolation.status.toUpperCase()}
                                            </span>
                                        </div>

                                        <p className="text-sm text-gray-300 mb-2">{currentViolation.latestViolation || 'Multiple violations detected'}</p>
                                        <p className="text-xs text-gray-500">{currentViolation.violationTime || 'Just now'}</p>
                                    </div>

                                    {/* Action Buttons */}
                                    <div className="grid grid-cols-2 gap-3 mb-4">
                                        <Button variant="outline" size="sm" className="border-red-500/30 hover:bg-red-500/10 text-red-400">
                                            <XCircle className="h-4 w-4" />
                                            Remove
                                        </Button>
                                        <Button size="sm" className="bg-gradient-to-r from-green-600 to-emerald-600">
                                            <CheckCircle className="h-4 w-4" />
                                            Keep
                                        </Button>
                                    </div>

                                    {/* Navigation */}
                                    <div className="flex items-center justify-between">
                                        <Button variant="ghost" size="sm" onClick={handlePrevious}>
                                            <ChevronLeft className="h-4 w-4" />
                                        </Button>
                                        <span className="text-sm text-gray-400">
                                            {currentViolationIndex + 1} / {violationReviews.length}
                                        </span>
                                        <Button variant="ghost" size="sm" onClick={handleNext}>
                                            <ChevronRight className="h-4 w-4" />
                                        </Button>
                                    </div>
                                </motion.div>
                            </AnimatePresence>
                        )}
                    </div>
                </div>
            </div>

            {/* Bottom Summary */}
            <div className="p-6 pt-0">
                <div className="grid grid-cols-3 gap-6">
                    {/* Session Statistics */}
                    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                        <div className="flex items-center gap-2 mb-4">
                            <TrendingUp className="h-5 w-5 text-cyan-400" />
                            <h3 className="text-lg font-bold text-white">Session Statistics</h3>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-sm mb-2">
                                    <span className="text-gray-400">Students Monitored</span>
                                    <span className="text-white font-semibold">{students.length}/{students.length}</span>
                                </div>
                                <Progress value={100} />
                            </div>

                            <div>
                                <div className="flex justify-between text-sm mb-2">
                                    <span className="text-gray-400">Compliance Rate</span>
                                    <span className="text-white font-semibold">{complianceRate}%</span>
                                </div>
                                <Progress value={complianceRate} />
                            </div>
                        </div>
                    </div>

                    {/* Violation Breakdown */}
                    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                        <div className="flex items-center gap-2 mb-4">
                            <AlertTriangle className="h-5 w-5 text-orange-400" />
                            <h3 className="text-lg font-bold text-white">Violation Breakdown</h3>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                                <p className="text-2xl font-bold text-green-400">{statusCounts.normal}</p>
                                <p className="text-xs text-gray-400">Normal</p>
                            </div>

                            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3">
                                <p className="text-2xl font-bold text-yellow-400">{statusCounts.warning}</p>
                                <p className="text-xs text-gray-400">Warnings</p>
                            </div>

                            <div className="bg-orange-500/10 border border-orange-500/20 rounded-lg p-3">
                                <p className="text-2xl font-bold text-orange-400">{statusCounts.suspicious}</p>
                                <p className="text-xs text-gray-400">Suspicious</p>
                            </div>

                            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                                <p className="text-2xl font-bold text-red-400">{statusCounts.critical}</p>
                                <p className="text-xs text-gray-400">Critical</p>
                            </div>
                        </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                        <div className="flex items-center gap-2 mb-4">
                            <Settings className="h-5 w-5 text-indigo-400" />
                            <h3 className="text-lg font-bold text-white">Quick Actions</h3>
                        </div>

                        <div className="space-y-3">
                            <Button className="w-full bg-gradient-to-r from-indigo-600 to-violet-600">
                                <Plus className="h-4 w-4" />
                                Create New Exam
                            </Button>

                            <Button variant="outline" className="w-full">
                                <Bell className="h-4 w-4" />
                                Send Announcement
                            </Button>

                            <Button variant="outline" className="w-full">
                                <Download className="h-4 w-4" />
                                Export Report
                            </Button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
