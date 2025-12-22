import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Users, Clock, Calendar, Activity, Eye } from 'lucide-react';

const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

const API_URL = (() => {
    const RAW_API_URL = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:5000/api';
    const trimmed = String(RAW_API_URL).trim().replace(/\/+$/, '');
    return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
})();

interface Exam {
    _id: string;
    title: string;
    courseCode: string;
    scheduledDate: string;
    startTime: string;
    endTime: string;
    duration: number;
    status: string;
    lecturerId: string;
    questions: any[];
    attempts?: any[];
}

export default function LecturerLiveExamsList({
    lecturerId,
    onBack,
    onSelectExam
}: {
    lecturerId: string;
    onBack: () => void;
    onSelectExam: (examId: string, examTitle: string) => void;
}) {
    const [liveExams, setLiveExams] = useState<Exam[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const fetchLiveExams = async () => {
            try {
                const res = await fetch(`${API_URL}/exams`);
                const data = await res.json();
                if (res.ok) {
                    // Filter for exams that are Live or Available and belong to this lecturer
                    const live = data.exams.filter((exam: Exam) => 
                        exam.lecturerId === lecturerId && 
                        (exam.status === 'Live' || exam.status === 'Available')
                    );
                    setLiveExams(live);
                }
                setIsLoading(false);
            } catch (err) {
                console.error('Failed to fetch live exams:', err);
                setIsLoading(false);
            }
        };

        fetchLiveExams();
        // Refresh every 30 seconds
        const interval = setInterval(fetchLiveExams, 30000);
        return () => clearInterval(interval);
    }, [lecturerId]);

    const getActiveStudents = (exam: Exam) => {
        // Count unique students who have attempts
        if (!exam.attempts || exam.attempts.length === 0) return 0;
        const uniqueUsers = new Set(exam.attempts.map((a: any) => a.userId));
        return uniqueUsers.size;
    };

    if (isLoading) {
        return (
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto mb-4"></div>
                    <p className="text-slate-400">Loading live exams...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 text-white p-6">
            {/* Header */}
            <div className="mb-6">
                <button 
                    onClick={onBack}
                    className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors mb-4"
                >
                    <ArrowLeft className="h-5 w-5" />
                    Back to Dashboard
                </button>

                <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                    <div className="flex items-center gap-3 mb-2">
                        <Activity className="h-8 w-8 text-green-400" />
                        <h1 className="text-3xl font-bold text-white">Live Exam Monitoring</h1>
                    </div>
                    <p className="text-slate-400">Select an exam to view real-time student monitoring</p>
                </div>
            </div>

            {/* Live Exams Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {liveExams.length === 0 ? (
                    <div className="col-span-full text-center py-12">
                        <Activity className="h-16 w-16 text-slate-600 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-slate-400 mb-2">No Live Exams</h3>
                        <p className="text-slate-500">Set an exam status to "Live" or "Available" to monitor it here.</p>
                    </div>
                ) : (
                    liveExams.map((exam) => (
                        <motion.div
                            key={exam._id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-slate-900 border border-slate-800 rounded-lg p-6 hover:border-indigo-500 transition-all cursor-pointer"
                            onClick={() => onSelectExam(exam._id, exam.title)}
                        >
                            <div className="mb-4">
                                <div className="flex items-start justify-between mb-2">
                                    <h3 className="text-xl font-bold text-white">{exam.title}</h3>
                                    <span className={cn(
                                        'px-3 py-1 rounded-full text-xs font-semibold',
                                        exam.status === 'Live' ? 'bg-green-900/30 text-green-300 animate-pulse' : 'bg-blue-900/30 text-blue-300'
                                    )}>
                                        {exam.status}
                                    </span>
                                </div>
                                <p className="text-slate-400 text-sm">{exam.courseCode}</p>
                            </div>

                            <div className="space-y-3 mb-4">
                                <div className="flex items-center gap-2 text-slate-300">
                                    <Calendar className="h-4 w-4 text-slate-500" />
                                    <span className="text-sm">{new Date(exam.scheduledDate).toLocaleDateString()}</span>
                                </div>
                                <div className="flex items-center gap-2 text-slate-300">
                                    <Clock className="h-4 w-4 text-slate-500" />
                                    <span className="text-sm">{exam.startTime} - {exam.endTime} ({exam.duration} min)</span>
                                </div>
                                <div className="flex items-center gap-2 text-slate-300">
                                    <Users className="h-4 w-4 text-slate-500" />
                                    <span className="text-sm">{getActiveStudents(exam)} active students</span>
                                </div>
                            </div>

                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onSelectExam(exam._id, exam.title);
                                }}
                                className="w-full px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                            >
                                <Eye className="h-4 w-4" />
                                Monitor Exam
                            </button>
                        </motion.div>
                    ))
                )}
            </div>

            {liveExams.length > 0 && (
                <div className="mt-6 text-center text-slate-500 text-sm">
                    <p>Monitoring {liveExams.length} live exam{liveExams.length !== 1 ? 's' : ''}</p>
                    <p className="mt-1">Click any exam card to enter detailed monitoring view</p>
                </div>
            )}
        </div>
    );
}
