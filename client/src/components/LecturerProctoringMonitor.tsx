import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, RefreshCw, Search, Filter, Download, AlertTriangle, CheckCircle, XCircle, Eye, Flag, Ban, MessageSquare, Users, Activity, Shield } from 'lucide-react';

const cn = (...classes: (string | undefined | null | false)[]) => classes.filter(Boolean).join(' ');

const API_URL = (() => {
    const RAW_API_URL = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:5000/api';
    const trimmed = String(RAW_API_URL).trim().replace(/\/+$/, '');
    return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
})();

interface StudentStatus {
    userId: string;
    studentId: string;
    name: string;
    status: 'online' | 'offline' | 'not-started' | 'submitted';
    identityVerified: boolean;
    faceCount: number;
    gazeDirection: 'center' | 'left' | 'right' | 'down' | 'unknown';
    headPose: 'normal' | 'suspicious';
    mouthStatus: 'normal' | 'talking';
    blinkStatus: 'normal' | 'drowsy';
    audioStatus: 'normal' | 'multiple-voices' | 'noise';
    lastSeen: string;
    riskScore: number;
    alertLevel: 'none' | 'low' | 'medium' | 'high' | 'critical';
    incidentCount: number;
}

interface Incident {
    _id: string;
    userId: string;
    studentId: string;
    studentName: string;
    timestamp: string;
    eventType: string;
    severity: 'low' | 'medium' | 'high';
    details: any;
    frameEvidence?: string;
    resolved: boolean;
    lecturerNotes?: string;
}

interface ExamInfo {
    _id: string;
    title: string;
    courseCode: string;
    scheduledDate: string;
    startTime: string;
    endTime: string;
    duration: number;
    status: string;
}

export default function LecturerProctoringMonitor({ 
    examId, 
    onBack, 
    showToast 
}: { 
    examId: string; 
    onBack: () => void;
    showToast: (msg: string, type: 'success' | 'error') => void;
}) {
    const [examInfo, setExamInfo] = useState<ExamInfo | null>(null);
    const [students, setStudents] = useState<StudentStatus[]>([]);
    const [incidents, setIncidents] = useState<Incident[]>([]);
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [severityFilter, setSeverityFilter] = useState<'all' | 'low' | 'medium' | 'high'>('all');
    const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
    const [selectedStudent, setSelectedStudent] = useState<StudentStatus | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Fetch exam info
    useEffect(() => {
        const fetchExamInfo = async () => {
            try {
                const res = await fetch(`${API_URL}/exams/${examId}`);
                const data = await res.json();
                if (res.ok) {
                    setExamInfo(data);
                }
            } catch (err) {
                console.error('Failed to fetch exam info:', err);
            }
        };
        fetchExamInfo();
    }, [examId]);

    // Polling for monitoring data
    useEffect(() => {
        if (!isMonitoring) return;

        const poll = async () => {
            try {
                const res = await fetch(`${API_URL}/lecturer/exams/${examId}/monitor`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                });
                const data = await res.json();
                if (res.ok) {
                    setStudents(data.students || []);
                    setIncidents(data.incidents || []);
                    setIsLoading(false);
                }
            } catch (err) {
                console.error('Monitoring poll error:', err);
                setIsLoading(false);
            }
        };

        poll(); // Initial fetch
        const interval = setInterval(poll, 3000); // Poll every 3 seconds

        return () => clearInterval(interval);
    }, [examId, isMonitoring]);

    // Load mock data initially
    useEffect(() => {
        const loadMockData = () => {
            // Mock students for testing
            const mockStudents: StudentStatus[] = [
                {
                    userId: '1',
                    studentId: 'S001',
                    name: 'John Doe',
                    status: 'online',
                    identityVerified: true,
                    faceCount: 1,
                    gazeDirection: 'center',
                    headPose: 'normal',
                    mouthStatus: 'normal',
                    blinkStatus: 'normal',
                    audioStatus: 'normal',
                    lastSeen: new Date().toISOString(),
                    riskScore: 5,
                    alertLevel: 'none',
                    incidentCount: 0
                },
                {
                    userId: '2',
                    studentId: 'S002',
                    name: 'Jane Smith',
                    status: 'online',
                    identityVerified: true,
                    faceCount: 2,
                    gazeDirection: 'left',
                    headPose: 'suspicious',
                    mouthStatus: 'talking',
                    blinkStatus: 'normal',
                    audioStatus: 'multiple-voices',
                    lastSeen: new Date().toISOString(),
                    riskScore: 65,
                    alertLevel: 'high',
                    incidentCount: 3
                },
                {
                    userId: '3',
                    studentId: 'S003',
                    name: 'Mike Johnson',
                    status: 'not-started',
                    identityVerified: false,
                    faceCount: 0,
                    gazeDirection: 'unknown',
                    headPose: 'normal',
                    mouthStatus: 'normal',
                    blinkStatus: 'normal',
                    audioStatus: 'normal',
                    lastSeen: new Date(Date.now() - 300000).toISOString(),
                    riskScore: 0,
                    alertLevel: 'none',
                    incidentCount: 0
                }
            ];

            const mockIncidents: Incident[] = [
                {
                    _id: '1',
                    userId: '2',
                    studentId: 'S002',
                    studentName: 'Jane Smith',
                    timestamp: new Date().toISOString(),
                    eventType: 'Multiple Faces Detected',
                    severity: 'high',
                    details: { faceCount: 2 },
                    resolved: false
                },
                {
                    _id: '2',
                    userId: '2',
                    studentId: 'S002',
                    studentName: 'Jane Smith',
                    timestamp: new Date(Date.now() - 60000).toISOString(),
                    eventType: 'Multiple Voices Detected',
                    severity: 'medium',
                    details: { audioStatus: 'multiple-voices' },
                    resolved: false
                }
            ];

            setStudents(mockStudents);
            setIncidents(mockIncidents);
            setIsLoading(false);
        };

        loadMockData();
    }, []);

    const filteredStudents = students.filter(s => 
        s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.studentId.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const filteredIncidents = incidents.filter(i => 
        severityFilter === 'all' || i.severity === severityFilter
    );

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'online': return 'bg-green-500';
            case 'offline': return 'bg-red-500';
            case 'not-started': return 'bg-gray-500';
            case 'submitted': return 'bg-blue-500';
            default: return 'bg-gray-500';
        }
    };

    const getAlertColor = (level: string) => {
        switch (level) {
            case 'critical': return 'border-red-600 bg-red-900/20';
            case 'high': return 'border-orange-600 bg-orange-900/20';
            case 'medium': return 'border-yellow-600 bg-yellow-900/20';
            case 'low': return 'border-blue-600 bg-blue-900/20';
            default: return 'border-slate-700 bg-slate-800';
        }
    };

    const getRiskColor = (score: number) => {
        if (score >= 75) return 'text-red-400';
        if (score >= 50) return 'text-orange-400';
        if (score >= 20) return 'text-yellow-400';
        return 'text-green-400';
    };

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
                    <div className="flex justify-between items-start">
                        <div>
                            <h1 className="text-3xl font-bold text-white mb-2">
                                {examInfo?.title || 'Loading...'}
                            </h1>
                            <p className="text-slate-400">
                                {examInfo?.courseCode} • {examInfo?.scheduledDate} • {examInfo?.duration} minutes
                            </p>
                            <div className="mt-2">
                                <span className={cn(
                                    'px-3 py-1 rounded-full text-sm font-semibold',
                                    examInfo?.status === 'Live' ? 'bg-green-900/30 text-green-300' : 'bg-slate-700 text-slate-300'
                                )}>
                                    {examInfo?.status}
                                </span>
                            </div>
                        </div>

                        <div className="flex gap-2">
                            <button
                                onClick={() => setIsMonitoring(!isMonitoring)}
                                className={cn(
                                    'px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2',
                                    isMonitoring 
                                        ? 'bg-red-600 hover:bg-red-700' 
                                        : 'bg-green-600 hover:bg-green-700'
                                )}
                            >
                                <Activity className="h-4 w-4" />
                                {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
                            </button>
                            <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-2">
                                <Download className="h-4 w-4" />
                                Export
                            </button>
                        </div>
                    </div>

                    {/* Stats Row */}
                    <div className="grid grid-cols-4 gap-4 mt-6">
                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <Users className="h-4 w-4" />
                                Total Students
                            </div>
                            <div className="text-2xl font-bold">{students.length}</div>
                        </div>
                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <CheckCircle className="h-4 w-4" />
                                Online
                            </div>
                            <div className="text-2xl font-bold text-green-400">
                                {students.filter(s => s.status === 'online').length}
                            </div>
                        </div>
                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <AlertTriangle className="h-4 w-4" />
                                Active Incidents
                            </div>
                            <div className="text-2xl font-bold text-orange-400">
                                {incidents.filter(i => !i.resolved).length}
                            </div>
                        </div>
                        <div className="bg-slate-800 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
                                <Shield className="h-4 w-4" />
                                High Risk
                            </div>
                            <div className="text-2xl font-bold text-red-400">
                                {students.filter(s => s.riskScore >= 50).length}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Controls */}
            <div className="flex gap-4 mb-6">
                <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search by name or student ID..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-indigo-500"
                    />
                </div>
                <select
                    value={severityFilter}
                    onChange={(e) => setSeverityFilter(e.target.value as any)}
                    className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                >
                    <option value="all">All Severity</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                </select>
            </div>

            {/* Main Content */}
            <div className="grid grid-cols-3 gap-6">
                {/* Students Grid - 2 columns */}
                <div className="col-span-2 space-y-4">
                    <h2 className="text-xl font-bold mb-4">Students ({filteredStudents.length})</h2>
                    
                    {isLoading ? (
                        <div className="text-center py-12 text-slate-400">Loading students...</div>
                    ) : filteredStudents.length === 0 ? (
                        <div className="text-center py-12 text-slate-400">No students found</div>
                    ) : (
                        <div className="space-y-3">
                            {filteredStudents.map(student => (
                                <motion.div
                                    key={student.userId}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className={cn(
                                        'p-4 rounded-lg border transition-all',
                                        getAlertColor(student.alertLevel)
                                    )}
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-3 mb-2">
                                                <div className={cn('h-3 w-3 rounded-full', getStatusColor(student.status))} />
                                                <h3 className="font-semibold text-white">{student.name}</h3>
                                                <span className="text-slate-400 text-sm">({student.studentId})</span>
                                                {student.identityVerified && (
                                                    <CheckCircle className="h-4 w-4 text-green-400" />
                                                )}
                                            </div>

                                            <div className="grid grid-cols-3 gap-4 text-sm">
                                                <div>
                                                    <span className="text-slate-400">Faces:</span>
                                                    <span className={cn('ml-2', student.faceCount !== 1 && 'text-red-400')}>
                                                        {student.faceCount}
                                                    </span>
                                                </div>
                                                <div>
                                                    <span className="text-slate-400">Gaze:</span>
                                                    <span className="ml-2">{student.gazeDirection}</span>
                                                </div>
                                                <div>
                                                    <span className="text-slate-400">Audio:</span>
                                                    <span className={cn('ml-2', student.audioStatus !== 'normal' && 'text-orange-400')}>
                                                        {student.audioStatus}
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="mt-2 flex items-center gap-4 text-sm">
                                                <div>
                                                    <span className="text-slate-400">Risk Score:</span>
                                                    <span className={cn('ml-2 font-bold', getRiskColor(student.riskScore))}>
                                                        {student.riskScore}/100
                                                    </span>
                                                </div>
                                                <div>
                                                    <span className="text-slate-400">Incidents:</span>
                                                    <span className="ml-2">{student.incidentCount}</span>
                                                </div>
                                                <div>
                                                    <span className="text-slate-400">Last seen:</span>
                                                    <span className="ml-2">{new Date(student.lastSeen).toLocaleTimeString()}</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="flex gap-2">
                                            <button 
                                                onClick={() => setSelectedStudent(student)}
                                                className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                                                title="View Details"
                                            >
                                                <Eye className="h-4 w-4" />
                                            </button>
                                            <button 
                                                className="p-2 bg-yellow-900/30 hover:bg-yellow-900/50 text-yellow-400 rounded-lg transition-colors"
                                                title="Flag"
                                            >
                                                <Flag className="h-4 w-4" />
                                            </button>
                                            <button 
                                                className="p-2 bg-red-900/30 hover:bg-red-900/50 text-red-400 rounded-lg transition-colors"
                                                title="Disqualify"
                                            >
                                                <Ban className="h-4 w-4" />
                                            </button>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Incidents Panel - 1 column */}
                <div>
                    <h2 className="text-xl font-bold mb-4">Recent Incidents</h2>
                    <div className="space-y-3">
                        {filteredIncidents.length === 0 ? (
                            <div className="bg-slate-800 rounded-lg p-6 text-center text-slate-400">
                                No incidents detected
                            </div>
                        ) : (
                            filteredIncidents.map(incident => (
                                <motion.div
                                    key={incident._id}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    onClick={() => setSelectedIncident(incident)}
                                    className={cn(
                                        'p-4 rounded-lg border cursor-pointer transition-all hover:border-indigo-500',
                                        incident.severity === 'high' && 'border-red-600 bg-red-900/10',
                                        incident.severity === 'medium' && 'border-yellow-600 bg-yellow-900/10',
                                        incident.severity === 'low' && 'border-blue-600 bg-blue-900/10'
                                    )}
                                >
                                    <div className="flex items-start justify-between mb-2">
                                        <div className="flex-1">
                                            <h4 className="font-semibold text-white text-sm">{incident.eventType}</h4>
                                            <p className="text-xs text-slate-400 mt-1">{incident.studentName}</p>
                                        </div>
                                        <span className={cn(
                                            'px-2 py-1 rounded text-xs font-semibold',
                                            incident.severity === 'high' && 'bg-red-600/20 text-red-400',
                                            incident.severity === 'medium' && 'bg-yellow-600/20 text-yellow-400',
                                            incident.severity === 'low' && 'bg-blue-600/20 text-blue-400'
                                        )}>
                                            {incident.severity.toUpperCase()}
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-400">
                                        {new Date(incident.timestamp).toLocaleTimeString()}
                                    </p>
                                </motion.div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Incident Modal */}
            {selectedIncident && (
                <div 
                    className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
                    onClick={() => setSelectedIncident(null)}
                >
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        onClick={(e) => e.stopPropagation()}
                        className="bg-slate-900 rounded-lg border border-slate-700 p-6 max-w-2xl w-full"
                    >
                        <h2 className="text-xl font-bold mb-4">Incident Details</h2>
                        
                        <div className="space-y-4">
                            <div>
                                <p className="text-slate-400 text-sm">Event Type</p>
                                <p className="text-white font-semibold">{selectedIncident.eventType}</p>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-slate-400 text-sm">Student</p>
                                    <p className="text-white">{selectedIncident.studentName}</p>
                                </div>
                                <div>
                                    <p className="text-slate-400 text-sm">Severity</p>
                                    <span className={cn(
                                        'px-2 py-1 rounded text-xs font-semibold',
                                        selectedIncident.severity === 'high' && 'bg-red-600/20 text-red-400',
                                        selectedIncident.severity === 'medium' && 'bg-yellow-600/20 text-yellow-400',
                                        selectedIncident.severity === 'low' && 'bg-blue-600/20 text-blue-400'
                                    )}>
                                        {selectedIncident.severity.toUpperCase()}
                                    </span>
                                </div>
                            </div>

                            <div>
                                <p className="text-slate-400 text-sm">Timestamp</p>
                                <p className="text-white">{new Date(selectedIncident.timestamp).toLocaleString()}</p>
                            </div>

                            {selectedIncident.frameEvidence && (
                                <div>
                                    <p className="text-slate-400 text-sm mb-2">Evidence</p>
                                    <img 
                                        src={selectedIncident.frameEvidence} 
                                        alt="Evidence" 
                                        className="rounded-lg border border-slate-700 w-full"
                                    />
                                </div>
                            )}

                            <div>
                                <p className="text-slate-400 text-sm mb-2">Lecturer Notes</p>
                                <textarea 
                                    className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-white resize-none focus:outline-none focus:border-indigo-500"
                                    rows={3}
                                    placeholder="Add notes about this incident..."
                                    defaultValue={selectedIncident.lecturerNotes || ''}
                                />
                            </div>
                        </div>

                        <div className="flex gap-3 mt-6">
                            <button 
                                onClick={() => {
                                    showToast('Incident confirmed', 'success');
                                    setSelectedIncident(null);
                                }}
                                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors"
                            >
                                Confirm Violation
                            </button>
                            <button 
                                onClick={() => {
                                    showToast('Incident dismissed', 'success');
                                    setSelectedIncident(null);
                                }}
                                className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors"
                            >
                                Dismiss
                            </button>
                        </div>
                    </motion.div>
                </div>
            )}

            {/* Student Detail Modal */}
            {selectedStudent && (
                <div 
                    className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
                    onClick={() => setSelectedStudent(null)}
                >
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        onClick={(e) => e.stopPropagation()}
                        className="bg-slate-900 rounded-lg border border-slate-700 p-6 max-w-2xl w-full"
                    >
                        <h2 className="text-xl font-bold mb-4">Student Details</h2>
                        
                        <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-slate-400 text-sm">Name</p>
                                    <p className="text-white font-semibold">{selectedStudent.name}</p>
                                </div>
                                <div>
                                    <p className="text-slate-400 text-sm">Student ID</p>
                                    <p className="text-white">{selectedStudent.studentId}</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-slate-400 text-sm">Status</p>
                                    <div className="flex items-center gap-2 mt-1">
                                        <div className={cn('h-3 w-3 rounded-full', getStatusColor(selectedStudent.status))} />
                                        <span className="text-white capitalize">{selectedStudent.status}</span>
                                    </div>
                                </div>
                                <div>
                                    <p className="text-slate-400 text-sm">Identity Verified</p>
                                    <p className="text-white mt-1">
                                        {selectedStudent.identityVerified ? (
                                            <span className="flex items-center gap-1 text-green-400">
                                                <CheckCircle className="h-4 w-4" /> Yes
                                            </span>
                                        ) : (
                                            <span className="flex items-center gap-1 text-red-400">
                                                <XCircle className="h-4 w-4" /> No
                                            </span>
                                        )}
                                    </p>
                                </div>
                            </div>

                            <div className="bg-slate-800 rounded-lg p-4">
                                <h3 className="font-semibold mb-3">Current Status</h3>
                                <div className="grid grid-cols-2 gap-3 text-sm">
                                    <div>
                                        <span className="text-slate-400">Face Count:</span>
                                        <span className={cn('ml-2', selectedStudent.faceCount !== 1 && 'text-red-400')}>
                                            {selectedStudent.faceCount}
                                        </span>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">Gaze Direction:</span>
                                        <span className="ml-2 capitalize">{selectedStudent.gazeDirection}</span>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">Head Pose:</span>
                                        <span className={cn('ml-2 capitalize', selectedStudent.headPose === 'suspicious' && 'text-orange-400')}>
                                            {selectedStudent.headPose}
                                        </span>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">Mouth Status:</span>
                                        <span className={cn('ml-2 capitalize', selectedStudent.mouthStatus === 'talking' && 'text-yellow-400')}>
                                            {selectedStudent.mouthStatus}
                                        </span>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">Blink Status:</span>
                                        <span className="ml-2 capitalize">{selectedStudent.blinkStatus}</span>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">Audio Status:</span>
                                        <span className={cn('ml-2 capitalize', selectedStudent.audioStatus !== 'normal' && 'text-orange-400')}>
                                            {selectedStudent.audioStatus}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-slate-800 rounded-lg p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-400 text-sm">Risk Score</p>
                                        <p className={cn('text-3xl font-bold', getRiskColor(selectedStudent.riskScore))}>
                                            {selectedStudent.riskScore}/100
                                        </p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-slate-400 text-sm">Total Incidents</p>
                                        <p className="text-3xl font-bold text-white">{selectedStudent.incidentCount}</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="flex gap-3 mt-6">
                            <button 
                                className="flex-1 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                            >
                                <MessageSquare className="h-4 w-4" />
                                Send Message
                            </button>
                            <button 
                                onClick={() => setSelectedStudent(null)}
                                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors"
                            >
                                Close
                            </button>
                        </div>
                    </motion.div>
                </div>
            )}
        </div>
    );
}
