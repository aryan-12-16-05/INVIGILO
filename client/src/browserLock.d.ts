declare module './browserLock' {
  export type BrowserLockViolationDetails = {
    title?: string;
    message?: string;
  };

  export default class BrowserLock {
    examId: string;
    userId: string;
    serverUrl: string;
    isEnabled: boolean;
    warningCount: number;
    maxWarnings: number;

    onViolation: null | ((title: string, message: string, warningCount: number) => void);
    onMaxWarnings: null | ((warningCount: number, details?: BrowserLockViolationDetails) => void);

    constructor(examId: string, userId: string, serverUrl?: string);

    enable(): Promise<boolean>;
    disable(): void;

    enterFullscreen(): Promise<boolean>;

    getWarningCount(): number;
    resetWarnings(): void;
  }
}
