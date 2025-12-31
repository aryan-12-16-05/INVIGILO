import React, { useState, useEffect } from 'react';
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
    studentId: string;
    name: string;
    score: number;
    percentage: number;
    riskScore: number;
    incidentCount: number;
    duration: number;
    status: 'completed' | 'disqualified' | 'abandoned';
}

export default function LecturerExamReport({ 
    examId, 
    onBack, 
    showToast 
}: { 
    examId: string; 
    onBack: () => void;
    showToast: (msg: string, type: 'success' | 'error') => void;
}) {
    const [report, setReport] = useState<ExamReport | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [sortBy, setSortBy] = useState<'name' | 'score' | 'risk'>('score');
    const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

    useEffect(() => {
        const fetchReport = async () => {
            try {
                const res = await fetch(`${API_URL}/exams/${examId}/report`);
                const data = await res.json();
                if (res.ok && data.examId) {
                    setReport(data);
                    setIsLoading(false);
                } else {
                    // Load mock data if API returns error
                    loadMockData();
                    setIsLoading(false);
                }
            } catch (err) {
                console.error('Failed to fetch report:', err);
                // Load mock data on error
                loadMockData();
                setIsLoading(false);
            }
        };

        fetchReport();
    }, [examId]);

    const loadMockData = () => {
        const mockReport: ExamReport = {
            examId: examId,
            title: 'Data Science Mid-Term Examination',
            courseCode: 'CSE401',
            date: '2024-01-15',
            duration: 90,
            totalStudents: 45,
            attendanceRate: 93.3,
            averageScore: 76.5,
            passRate: 84.4,
            totalIncidents: 23,
            highRiskStudents: 3,
            students: [
                {
                    studentId: 'S001',
                    name: 'John Doe',
                    score: 85,
                    percentage: 85,
                    riskScore: 5,
                    incidentCount: 0,
                    duration: 88,
                    status: 'completed'
                },
                {
                    studentId: 'S002',
                    name: 'Jane Smith',
                    score: 72,
                    percentage: 72,
                    riskScore: 65,
                    incidentCount: 5,
                    duration: 90,
                    status: 'completed'
                },
                {
                    studentId: 'S003',
                    name: 'Mike Johnson',
                    score: 0,
                    percentage: 0,
                    riskScore: 0,
                    incidentCount: 0,
                    duration: 0,
                    status: 'abandoned'
                },
                {
                    studentId: 'S004',
                    name: 'Sarah Williams',
                    score: 91,
                    percentage: 91,
                    riskScore: 10,
                    incidentCount: 1,
                    duration: 85,
                    status: 'completed'
                },
                {
                    studentId: 'S005',
                    name: 'Robert Brown',
                    score: 0,
                    percentage: 0,
                    riskScore: 95,
                    incidentCount: 12,
                    duration: 45,
                    status: 'disqualified'
                }
            ],
            incidentBreakdown: {
                identityMismatch: 2,
                multipleFaces: 8,
                phoneDetected: 4,
                tabSwitch: 5,
                gazeAway: 3,
                audioViolation: 1
            }
        };

        setReport(mockReport);
    };

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
                                {sortedStudents.map((student, idx) => (
                                    <motion.tr 
                                        key={student.studentId}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: idx * 0.05 }}
                                        className="border-b border-slate-800 hover:bg-slate-800/50 transition-colors"
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
                                ))}
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
            </div>
        </div>
    );
}
