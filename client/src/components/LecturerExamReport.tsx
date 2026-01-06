import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Download, FileText, BarChart3, Users, AlertTriangle, CheckCircle, TrendingUp, Clock } from 'lucide-react';

const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

const API_URL = (() => {
    const RAW_API_URL = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:5000/api';
    const trimmed = String(RAW_API_URL).trim().replace(/\/+$/, '');
    return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
})();

interface ExamReport {
    examId: string;
    title: string;
    courseCode: string;
    date: string;
    duration: number;
    totalStudents: number;
    attendanceRate: number;
    averageScore: number;
    passRate: number;
    totalIncidents: number;
    highRiskStudents: number;
    students: StudentReportData[];
    incidentBreakdown: {
        identityMismatch: number;
        multipleFaces: number;
        phoneDetected: number;
        tabSwitch: number;
        gazeAway: number;
        audioViolation: number;
    };
}

interface StudentReportData {
    userId?: string;
    studentId: string;
    name: string;
    score: number;
    percentage: number;
    riskScore: number;
    incidentCount: number;
    duration: number;
    status: 'completed' | 'disqualified' | 'abandoned';
}

interface StudentAttemptDetails {
    attempt?: {
        userId: string;
        score?: number;
        totalMarks?: number;
        percentage?: number;
        completedAt?: string;
        perQuestion?: Array<{
            questionId: string;
            question: string;
            given: any;
            expected: any;
            marks: number;
            correct: boolean;
        }>;
    } | null;
}

interface StudentProctorEvent {
    _id: string;
    timestamp?: string;
    eventType?: string;
    severity?: string;
    details?: any;
    frameEvidence?: string;
}

export default function LecturerExamReport({ 
    user,
    examId, 
    onBack, 
    showToast 
}: { 
    user: { _id: string };
    examId: string; 
    onBack: () => void;
    showToast: (msg: string, type: 'success' | 'error') => void;
}) {
    const [report, setReport] = useState<ExamReport | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [sortBy, setSortBy] = useState<'name' | 'score' | 'risk'>('score');
    const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

    const [selectedStudent, setSelectedStudent] = useState<StudentReportData | null>(null);
    const [studentAttempt, setStudentAttempt] = useState<StudentAttemptDetails['attempt'] | null>(null);
    const [studentEvents, setStudentEvents] = useState<StudentProctorEvent[]>([]);
    const [studentViolations, setStudentViolations] = useState<any[]>([]);
    const [studentDetailLoading, setStudentDetailLoading] = useState(false);
    const [studentDetailError, setStudentDetailError] = useState<string | null>(null);

    useEffect(() => {
        const fetchReport = async () => {
            try {
                console.log(`[REPORT] Fetching report for exam: ${examId}`);
                const res = await fetch(`${API_URL}/exams/${examId}/report`);
                console.log(`[REPORT] Response status: ${res.status}`);
                
                const data = await res.json();
                console.log(`[REPORT] Response data:`, data);
                
                if (res.ok) {
                    setReport(data);
                    console.log(`[REPORT] Report loaded successfully:`, data.title);
                } else {
                    console.error(`[REPORT] Error response:`, data);
                    showToast(data.error || 'Failed to load report', 'error');
                }
            } catch (err) {
                console.error('[REPORT] Fetch error:', err);
                showToast('Failed to load report. Check your connection and try again.', 'error');
            } finally {
                setIsLoading(false);
            }
        };

        fetchReport();
    }, [examId, showToast]);

    useEffect(() => {
        const fetchStudentDetail = async () => {
            if (!selectedStudent) return;

            const userId = (selectedStudent as any).userId;
            if (!userId) {
                setStudentDetailError('Missing userId for this student.');
                return;
            }

            try {
                setStudentDetailLoading(true);
                setStudentDetailError(null);
                setStudentAttempt(null);
                setStudentEvents([]);
                setStudentViolations([]);

                const headers: Record<string, string> = {
                    'Content-Type': 'application/json',
                    'X-User-Id': String(user._id)
                };

                const [attemptRes, eventsRes, violationsRes] = await Promise.all([
                    fetch(`${API_URL}/exams/${examId}/attempt?userId=${encodeURIComponent(String(userId))}`, { headers }),
                    fetch(`${API_URL}/exams/${examId}/proctoring/${encodeURIComponent(String(userId))}`, { headers }),
                    fetch(`${API_URL}/exams/${examId}/students/${encodeURIComponent(String(userId))}/violations`, { headers })
                ]);

                if (attemptRes.ok) {
                    const attemptJson = await attemptRes.json();
                    setStudentAttempt(attemptJson?.attempt ?? null);
                }

                if (eventsRes.ok) {
                    const eventsJson = await eventsRes.json();
                    setStudentEvents(Array.isArray(eventsJson?.events) ? eventsJson.events : []);
                }

                if (violationsRes.ok) {
                    const vJson = await violationsRes.json();
                    setStudentViolations(Array.isArray(vJson?.violations) ? vJson.violations : []);
                }

                if (!attemptRes.ok && !eventsRes.ok && !violationsRes.ok) {
                    setStudentDetailError('Failed to load student details.');
                }
            } catch (e) {
                console.error('[REPORT] Student detail fetch error:', e);
                setStudentDetailError('Failed to load student details.');
            } finally {
                setStudentDetailLoading(false);
            }
        };

        fetchStudentDetail();
    }, [selectedStudent, examId, user._id]);

    const handleExport = (format: 'pdf' | 'csv') => {
        showToast(`Exporting report as ${format.toUpperCase()}...`, 'success');
        // Actual export logic would go here
    };

    const handlePrint = () => {
        window.print();
    };

    const sortedStudents = report?.students ? [...report.students].sort((a, b) => {
        let comparison = 0;
        if (sortBy === 'name') {
            comparison = a.name.localeCompare(b.name);
        } else if (sortBy === 'score') {
            comparison = a.score - b.score;
        } else if (sortBy === 'risk') {
            comparison = a.riskScore - b.riskScore;
        }
        return sortOrder === 'asc' ? comparison : -comparison;
    }) : [];

    const getRiskColor = (score: number) => {
        if (score >= 75) return 'text-red-400';
        if (score >= 50) return 'text-orange-400';
        if (score >= 20) return 'text-yellow-400';
        return 'text-green-400';
    };

    const getRiskBg = (score: number) => {
        if (score >= 75) return 'bg-red-900/20';
        if (score >= 50) return 'bg-orange-900/20';
        if (score >= 20) return 'bg-yellow-900/20';
        return 'bg-green-900/20';
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'bg-green-900/20 text-green-400';
            case 'disqualified': return 'bg-red-900/20 text-red-400';
            case 'abandoned': return 'bg-gray-700 text-gray-400';
            default: return 'bg-slate-700 text-slate-400';
        }
    };

    if (isLoading) {
        return (
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto mb-4"></div>
                    <p className="text-slate-400">Loading report...</p>
                </div>
            </div>
        );
    }

    if (!report) {
        return (
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <div className="text-center">
                    <p className="text-slate-400">Failed to load report</p>
                    <button onClick={onBack} className="mt-4 text-indigo-400 hover:text-indigo-300">
                        Go back
                    </button>
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
                    className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors mb-4 print:hidden"
                >
                    <ArrowLeft className="h-5 w-5" />
                    Back to Dashboard
                </button>

                {isLoading ? (
                    <div className="flex items-center justify-center h-96">
                        <div className="text-slate-400">Loading report...</div>
                    </div>
                ) : !report ? (
                    <div className="flex flex-col items-center justify-center h-96 space-y-4">
                        <AlertTriangle className="h-16 w-16 text-orange-400" />
                        <div className="text-xl text-slate-400">Failed to load report</div>
                        <p className="text-sm text-slate-500">Please check your connection and try again.</p>
                        <button 
                            onClick={() => window.location.reload()} 
                            className="mt-4 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                ) : (
                    <>
                        {/* Report Header */}
                        <div className="bg-slate-900 border border-slate-800 rounded-lg p-8 mb-6">
                            <div className="flex justify-between items-start mb-6">
                                <div>
                                    <div className="flex items-center gap-3 mb-2">
                                        <FileText className="h-8 w-8 text-indigo-400" />
                                        <h1 className="text-3xl font-bold text-white">Exam Report</h1>
                                    </div>
                                    <h2 className="text-2xl font-semibold text-slate-300 mb-2">{report.title}</h2>
                                    <p className="text-slate-400">
                                        {report.courseCode} • {new Date(report.date).toLocaleDateString()} • {report.duration} minutes
                                    </p>
                                    {report.totalStudents === 0 && (
                                        <p className="text-yellow-400 text-sm mt-2">⚠ No student submissions yet</p>
                                    )}
                                </div>

                                <div className="flex gap-2 print:hidden">
                                    <button 
                                        onClick={() => handleExport('pdf')}
                                        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors flex items-center gap-2"
                                    >
                                        <Download className="h-4 w-4" />
                                        PDF
                                    </button>
                                    <button 
                                        onClick={() => handleExport('csv')}
                                        className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-2"
                                    >
                                        <Download className="h-4 w-4" />
                                        CSV
                                    </button>
                                    <button 
                                        onClick={handlePrint}
                                        className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
                                    >
                                        Print
                                    </button>
                                </div>
                            </div>

                    {/* KPI Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <Users className="h-4 w-4" />
                                Attendance
                            </div>
                            <div className="text-2xl font-bold text-green-400">{report.attendanceRate}%</div>
                            <p className="text-sm text-slate-400 mt-1">{report.totalStudents} students</p>
                        </div>

                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <TrendingUp className="h-4 w-4" />
                                Average Score
                            </div>
                            <div className="text-2xl font-bold text-blue-400">{report.averageScore}%</div>
                            <p className="text-sm text-slate-400 mt-1">Class average</p>
                        </div>

                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <CheckCircle className="h-4 w-4" />
                                Pass Rate
                            </div>
                            <div className="text-2xl font-bold text-green-400">{report.passRate}%</div>
                            <p className="text-sm text-slate-400 mt-1">≥60% threshold</p>
                        </div>

                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <AlertTriangle className="h-4 w-4" />
                                Incidents
                            </div>
                            <div className="text-2xl font-bold text-orange-400">{report.totalIncidents}</div>
                            <p className="text-sm text-slate-400 mt-1">{report.highRiskStudents} high risk</p>
                        </div>
                    </div>
                </div>

                {/* Incident Breakdown */}
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 mb-6">
                    <div className="flex items-center gap-2 mb-4">
                        <BarChart3 className="h-5 w-5 text-indigo-400" />
                        <h3 className="text-xl font-bold">Violation Analytics</h3>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        {Object.entries(report.incidentBreakdown).map(([key, value]) => (
                            <div key={key} className="bg-slate-800 rounded-lg p-4">
                                <p className="text-slate-400 text-sm capitalize mb-1">
                                    {key.replace(/([A-Z])/g, ' $1').trim()}
                                </p>
                                <div className="flex items-baseline gap-2">
                                    <span className="text-2xl font-bold text-white">{value}</span>
                                    <span className="text-sm text-slate-400">incidents</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Student Performance Table */}
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-xl font-bold">Student Performance</h3>
                        
                        <div className="flex gap-2 print:hidden">
                            <select
                                value={sortBy}
                                onChange={(e) => setSortBy(e.target.value as any)}
                                className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                            >
                                <option value="name">Sort by Name</option>
                                <option value="score">Sort by Score</option>
                                <option value="risk">Sort by Risk</option>
                            </select>
                            <button
                                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                                className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm hover:bg-slate-700 transition-colors"
                            >
                                {sortOrder === 'asc' ? '↑' : '↓'}
                            </button>
                        </div>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-slate-700">
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium text-sm">Student ID</th>
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium text-sm">Name</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium text-sm">Score</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium text-sm">Risk Score</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium text-sm">Incidents</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium text-sm">Duration</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium text-sm">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {sortedStudents.length === 0 ? (
                                    <tr>
                                        <td colSpan={7} className="py-12 text-center text-slate-400">
                                            <Users className="h-12 w-12 mx-auto mb-2 text-slate-600" />
                                            <p>No student submissions yet</p>
                                            <p className="text-sm text-slate-500 mt-1">Students will appear here once they complete the exam</p>
                                        </td>
                                    </tr>
                                ) : (
                                    sortedStudents.map((student, idx) => (
                                    <motion.tr 
                                        key={student.studentId}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: idx * 0.05 }}
                                        className="border-b border-slate-800 hover:bg-slate-800/50 transition-colors cursor-pointer"
                                        onClick={() => setSelectedStudent(student)}
                                    >
                                        <td className="py-3 px-4 text-white font-mono text-sm">{student.studentId}</td>
                                        <td className="py-3 px-4 text-white">{student.name}</td>
                                        <td className="py-3 px-4 text-center">
                                            <span className={cn(
                                                'font-bold',
                                                student.percentage >= 80 ? 'text-green-400' :
                                                student.percentage >= 60 ? 'text-blue-400' :
                                                student.percentage >= 40 ? 'text-yellow-400' :
                                                'text-red-400'
                                            )}>
                                                {student.percentage}%
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-center">
                                            <span className={cn(
                                                'px-2 py-1 rounded text-sm font-semibold',
                                                getRiskBg(student.riskScore),
                                                getRiskColor(student.riskScore)
                                            )}>
                                                {student.riskScore}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-center text-white">{student.incidentCount}</td>
                                        <td className="py-3 px-4 text-center text-white">{student.duration} min</td>
                                        <td className="py-3 px-4 text-center">
                                            <span className={cn(
                                                'px-2 py-1 rounded text-xs font-semibold capitalize',
                                                getStatusColor(student.status)
                                            )}>
                                                {student.status}
                                            </span>
                                        </td>
                                    </motion.tr>
                                )))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Summary Statistics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                        <h4 className="text-slate-400 text-sm mb-2">Score Distribution</h4>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">90-100%:</span>
                                <span className="text-white font-semibold">
                                    {sortedStudents.filter(s => s.percentage >= 90).length} students
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">80-89%:</span>
                                <span className="text-white font-semibold">
                                    {sortedStudents.filter(s => s.percentage >= 80 && s.percentage < 90).length} students
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">70-79%:</span>
                                <span className="text-white font-semibold">
                                    {sortedStudents.filter(s => s.percentage >= 70 && s.percentage < 80).length} students
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">60-69%:</span>
                                <span className="text-white font-semibold">
                                    {sortedStudents.filter(s => s.percentage >= 60 && s.percentage < 70).length} students
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Below 60%:</span>
                                <span className="text-red-400 font-semibold">
                                    {sortedStudents.filter(s => s.percentage < 60).length} students
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                        <h4 className="text-slate-400 text-sm mb-2">Risk Analysis</h4>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Low Risk (0-19):</span>
                                <span className="text-green-400 font-semibold">
                                    {sortedStudents.filter(s => s.riskScore < 20).length}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Medium Risk (20-49):</span>
                                <span className="text-yellow-400 font-semibold">
                                    {sortedStudents.filter(s => s.riskScore >= 20 && s.riskScore < 50).length}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">High Risk (50-74):</span>
                                <span className="text-orange-400 font-semibold">
                                    {sortedStudents.filter(s => s.riskScore >= 50 && s.riskScore < 75).length}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Critical (75+):</span>
                                <span className="text-red-400 font-semibold">
                                    {sortedStudents.filter(s => s.riskScore >= 75).length}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                        <h4 className="text-slate-400 text-sm mb-2">Exam Status</h4>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Completed:</span>
                                <span className="text-green-400 font-semibold">
                                    {sortedStudents.filter(s => s.status === 'completed').length}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Disqualified:</span>
                                <span className="text-red-400 font-semibold">
                                    {sortedStudents.filter(s => s.status === 'disqualified').length}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Abandoned:</span>
                                <span className="text-gray-400 font-semibold">
                                    {sortedStudents.filter(s => s.status === 'abandoned').length}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-6 text-center text-slate-500 text-sm">
                    <p>Report generated on {new Date().toLocaleString()}</p>
                    <p className="mt-1">Invigilo Proctoring System © 2024</p>
                </div>
                </>
                )}
            </div>

            {/* Student Detail Modal */}
            {selectedStudent && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70">
                    <div className="w-full max-w-5xl bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
                            <div>
                                <h3 className="text-lg font-bold text-white">{selectedStudent.name}</h3>
                                <p className="text-sm text-slate-400">{selectedStudent.studentId}</p>
                            </div>
                            <button
                                onClick={() => {
                                    setSelectedStudent(null);
                                    setStudentAttempt(null);
                                    setStudentEvents([]);
                                    setStudentViolations([]);
                                    setStudentDetailError(null);
                                }}
                                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm"
                            >
                                Close
                            </button>
                        </div>

                        <div className="p-6 space-y-6 max-h-[80vh] overflow-y-auto">
                            {studentDetailLoading && (
                                <div className="text-slate-400">Loading student details…</div>
                            )}
                            {studentDetailError && (
                                <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-300 text-sm">
                                    {studentDetailError}
                                </div>
                            )}

                            {/* Question-wise results */}
                            <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-4">
                                <div className="flex items-center gap-2 mb-3">
                                    <Clock className="h-4 w-4 text-indigo-400" />
                                    <h4 className="font-semibold">Question-wise Results</h4>
                                </div>

                                {studentAttempt?.perQuestion && studentAttempt.perQuestion.length > 0 ? (
                                    <div className="space-y-3">
                                        <div className="text-sm text-slate-300">
                                            Score: <span className="font-semibold">{studentAttempt.score ?? selectedStudent.score}</span> / <span className="font-semibold">{studentAttempt.totalMarks ?? '-'}</span>
                                            {' '}• Percentage: <span className="font-semibold">{studentAttempt.percentage ?? selectedStudent.percentage}%</span>
                                        </div>
                                        <div className="space-y-2">
                                            {studentAttempt.perQuestion.map((q, i) => (
                                                <div key={q.questionId || i} className="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                                                    <div className="flex items-start justify-between gap-3">
                                                        <div>
                                                            <div className="text-sm text-white font-medium">Q{i + 1}. {q.question}</div>
                                                            <div className="text-xs text-slate-400 mt-1">Marks: {q.marks}</div>
                                                        </div>
                                                        <span className={cn(
                                                            'text-xs px-2 py-1 rounded border',
                                                            q.correct ? 'bg-green-500/10 text-green-300 border-green-500/20' : 'bg-red-500/10 text-red-300 border-red-500/20'
                                                        )}>
                                                            {q.correct ? 'Correct' : 'Wrong'}
                                                        </span>
                                                    </div>
                                                    <div className="grid md:grid-cols-2 gap-3 mt-3 text-sm">
                                                        <div className="bg-slate-950/40 border border-slate-800 rounded p-2">
                                                            <div className="text-xs text-slate-400">Your answer</div>
                                                            <div className="text-slate-200 break-words">{String(q.given ?? '—')}</div>
                                                        </div>
                                                        <div className="bg-slate-950/40 border border-slate-800 rounded p-2">
                                                            <div className="text-xs text-slate-400">Correct answer</div>
                                                            <div className="text-slate-200 break-words">{String(q.expected ?? '—')}</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-slate-400 text-sm">No per-question results found for this student.</div>
                                )}
                            </div>

                            {/* Proctoring events timeline */}
                            <div className="bg-slate-950/40 border border-slate-800 rounded-lg p-4">
                                <div className="flex items-center gap-2 mb-3">
                                    <AlertTriangle className="h-4 w-4 text-orange-400" />
                                    <h4 className="font-semibold">Proctoring Timeline</h4>
                                </div>

                                {(studentViolations?.length ?? 0) === 0 && (studentEvents?.length ?? 0) === 0 ? (
                                    <div className="text-slate-400 text-sm">No proctoring events recorded for this student.</div>
                                ) : (
                                    <div className="space-y-3">
                                        {studentViolations?.length > 0 && (
                                            <div>
                                                <div className="text-sm text-slate-300 mb-2">Captured Violations</div>
                                                <div className="space-y-2">
                                                    {studentViolations.map((v: any) => (
                                                        <div key={v._id} className="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                                                            <div className="flex items-start justify-between gap-3">
                                                                <div>
                                                                    <div className="text-sm text-white font-medium">{String(v.eventType || 'violation')}</div>
                                                                    <div className="text-xs text-slate-400 mt-1">{v.timestamp ? new Date(v.timestamp).toLocaleString() : ''}</div>
                                                                </div>
                                                                <span className="text-xs px-2 py-1 rounded border bg-orange-500/10 text-orange-300 border-orange-500/20">
                                                                    {String(v.severity || 'unknown').toUpperCase()}
                                                                </span>
                                                            </div>
                                                            {v.frameEvidence && (
                                                                <div className="mt-3 rounded-lg overflow-hidden border border-slate-800 bg-slate-950/40">
                                                                    <img src={v.frameEvidence} alt="Evidence" className="w-full max-h-72 object-cover" />
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {studentEvents?.length > 0 && (
                                            <div>
                                                <div className="text-sm text-slate-300 mb-2">All Proctor Events</div>
                                                <div className="space-y-2">
                                                    {studentEvents.slice(0, 100).map((ev) => (
                                                        <div key={ev._id} className="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                                                            <div className="flex items-start justify-between gap-3">
                                                                <div>
                                                                    <div className="text-sm text-white font-medium">{String(ev.eventType || 'event')}</div>
                                                                    <div className="text-xs text-slate-400 mt-1">{ev.timestamp ? new Date(ev.timestamp).toLocaleString() : ''}</div>
                                                                </div>
                                                                {ev.severity && (
                                                                    <span className="text-xs px-2 py-1 rounded border bg-slate-800 text-slate-200 border-slate-700">
                                                                        {String(ev.severity).toUpperCase()}
                                                                    </span>
                                                                )}
                                                            </div>

                                                            {ev.details?.message && (
                                                                <div className="text-xs text-slate-300 mt-2">{String(ev.details.message)}</div>
                                                            )}

                                                            {ev.frameEvidence && (
                                                                <div className="mt-3 rounded-lg overflow-hidden border border-slate-800 bg-slate-950/40">
                                                                    <img src={ev.frameEvidence} alt="Evidence" className="w-full max-h-72 object-cover" />
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
