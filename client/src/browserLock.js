/**
 * Browser Lock and Security Module for Exam Proctoring
 * 
 * Prevents students from:
 * - Exiting fullscreen
 * - Switching tabs/windows
 * - Opening developer tools
 * - Copy/paste
 * - Right-clicking
 * - Taking screenshots
 * 
 * Usage:
 *   const lock = new BrowserLock(examId, userId, serverUrl);
 *   await lock.enable();
 *   // ... during exam ...
 *   lock.disable();
 */

class BrowserLock {
  constructor(examId, userId, serverUrl = 'http://localhost:5000') {
    this.examId = examId;
    this.userId = userId;
    this.serverUrl = serverUrl;
    this.isEnabled = false;
    this.listeners = [];
    this.warningCount = 0;
  this.maxWarnings = 5;

  // Anti-spam: tab switch / window blur often fires multiple events for a single action.
  // We'll debounce counting so one switch counts as ONE violation.
  this._lastContextSwitchAt = 0;
  this._contextSwitchCooldownMs = 1500;

    // Anti-spam: fullscreen exit + escape can cascade into blur/visibility/fullscreenchange.
    // Debounce fullscreen-related warnings so it counts as ONE violation.
    this._lastFullscreenViolationAt = 0;
    this._fullscreenViolationCooldownMs = 2000;
    
    // Callbacks
    this.onViolation = null;
    this.onMaxWarnings = null;
  }
  
  /**
   * Enable all browser security restrictions
   */
  async enable() {
    if (this.isEnabled) {
      console.warn('BrowserLock already enabled');
      return false;
    }
    
    try {
      // 1. Request fullscreen
      await this.enterFullscreen();
      
      // 2. Attach all event listeners
      this.attachFullscreenListener();
      this.attachVisibilityListener();
      this.attachFocusListener();
      this.attachKeyboardListener();
      this.attachContextMenuListener();
      this.attachCopyPasteListener();
      this.attachPrintScreenListener();
      this.attachDevToolsDetector();
      
      this.isEnabled = true;
      console.log('✅ BrowserLock enabled');
      return true;
    } catch (error) {
      console.error('❌ Failed to enable BrowserLock:', error);
      return false;
    }
  }
  
  /**
   * Disable all browser security restrictions
   */
  disable() {
    if (!this.isEnabled) return;
    
    // Remove all event listeners
    this.listeners.forEach(({ element, event, handler }) => {
      element.removeEventListener(event, handler);
    });
    
    this.listeners = [];
    this.isEnabled = false;
    
    // Exit fullscreen
    if (document.fullscreenElement) {
      document.exitFullscreen();
    }
    
    console.log('BrowserLock disabled');
  }
  
  /**
   * Request fullscreen mode
   */
  async enterFullscreen() {
    try {
      const elem = document.documentElement;
      
      if (elem.requestFullscreen) {
        await elem.requestFullscreen();
      } else if (elem.webkitRequestFullscreen) {
        await elem.webkitRequestFullscreen();
      } else if (elem.msRequestFullscreen) {
        await elem.msRequestFullscreen();
      } else {
        throw new Error('Fullscreen API not supported');
      }
      
      console.log('✅ Entered fullscreen');
      return true;
    } catch (error) {
      console.error('Failed to enter fullscreen:', error);
      throw error;
    }
  }
  
  /**
   * Detect fullscreen exit
   */
  attachFullscreenListener() {
    const handler = () => {
      if (!document.fullscreenElement) {
        this.logActivity('fullscreen_exit', {
          message: 'User exited fullscreen mode'
        });

        // Count as a single fullscreen violation (debounced)
        this.handleFullscreenViolation('Fullscreen Exit', 'You must stay in fullscreen mode during the exam');
        
        // Try to re-enter fullscreen after 1 second
        setTimeout(() => {
          if (this.isEnabled) {
            this.enterFullscreen().catch(() => {
              console.error('Failed to re-enter fullscreen');
            });
          }
        }, 1000);
      }
    };
    
    this.addListener(document, 'fullscreenchange', handler);
    this.addListener(document, 'webkitfullscreenchange', handler);
    this.addListener(document, 'msfullscreenchange', handler);
  }
  
  /**
   * Detect tab visibility changes
   */
  attachVisibilityListener() {
    const handler = () => {
      if (document.hidden) {
        this.logActivity('tab_unfocused', {
          message: 'Tab lost visibility'
        });

        this.handleContextSwitchViolation('Tab/Window Switch', 'You switched away from the exam window');
      }
    };
    
    this.addListener(document, 'visibilitychange', handler);
  }
  
  /**
   * Detect window focus changes
   */
  attachFocusListener() {
    const blurHandler = () => {
      this.logActivity('window_blur', {
        message: 'Window lost focus'
      });

      this.handleContextSwitchViolation('Tab/Window Switch', 'Exam window lost focus');
    };
    
    this.addListener(window, 'blur', blurHandler);
  }
  
  /**
   * Block keyboard shortcuts (F12, Ctrl+Shift+I, etc.)
   */
  attachKeyboardListener() {
    const handler = (e) => {
      // Escape - prevent exiting fullscreen via Esc
      // Note: browsers may not allow fully blocking Esc in all cases.
      if (e.key === 'Escape') {
        e.preventDefault();
        this.logActivity('escape_pressed', {
          key: 'Escape',
          message: 'Attempted to exit fullscreen with Escape'
        });

        // Ensure Esc produces ONLY one warning (debounced)
        this.handleFullscreenViolation('Escape pressed', 'Do not press Escape. Stay in fullscreen mode during the exam');

        // Suppress context-switch counting that may occur during fullscreen exit cascade
        this._lastContextSwitchAt = Date.now();

        // Try to re-enter fullscreen quickly
        setTimeout(() => {
          if (this.isEnabled) {
            this.enterFullscreen().catch(() => {});
          }
        }, 200);
        return false;
      }

      // F12 - Developer Tools
      if (e.key === 'F12') {
        e.preventDefault();
        this.logActivity('dev_tools_attempt', {
          key: 'F12',
          message: 'Attempted to open developer tools (F12)'
        });
        this.handleViolation('Developer Tools', 'Developer tools are not allowed');
        return false;
      }
      
      // Ctrl+Shift+I - Developer Tools
      if (e.ctrlKey && e.shiftKey && e.key === 'I') {
        e.preventDefault();
        this.logActivity('dev_tools_attempt', {
          key: 'Ctrl+Shift+I',
          message: 'Attempted to open developer tools (Ctrl+Shift+I)'
        });
        this.handleViolation('Developer Tools', 'Developer tools are not allowed');
        return false;
      }
      
      // Ctrl+Shift+C - Element Inspector
      if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        this.logActivity('dev_tools_attempt', {
          key: 'Ctrl+Shift+C',
          message: 'Attempted to open element inspector'
        });
        return false;
      }
      
      // Ctrl+Shift+J - Console
      if (e.ctrlKey && e.shiftKey && e.key === 'J') {
        e.preventDefault();
        this.logActivity('dev_tools_attempt', {
          key: 'Ctrl+Shift+J',
          message: 'Attempted to open console'
        });
        return false;
      }
      
      // Ctrl+U - View Source
      if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        this.logActivity('dev_tools_attempt', {
          key: 'Ctrl+U',
          message: 'Attempted to view page source'
        });
        return false;
      }
      
      // F11 - Fullscreen toggle (allow but log)
      if (e.key === 'F11') {
        e.preventDefault();
        this.logActivity('fullscreen_toggle', {
          key: 'F11',
          message: 'Attempted to toggle fullscreen with F11'
        });
        return false;
      }
      
      // Print Screen
      if (e.key === 'PrintScreen') {
        this.logActivity('screenshot_attempt', {
          key: 'PrintScreen',
          message: 'Print Screen key pressed'
        });
        this.handleViolation('Screenshot', 'Screenshots are not allowed during exam');
      }
    };
    
    this.addListener(document, 'keydown', handler);
  }

  /**
   * Count tab/window switching as a single violation (debounced)
   */
  handleContextSwitchViolation(title, message) {
    const now = Date.now();
    if (now - (this._lastContextSwitchAt || 0) < this._contextSwitchCooldownMs) {
      // Still log but don't increment warnings again
      this.handleViolation(title, message, false);
      return;
    }
    this._lastContextSwitchAt = now;
    this.handleViolation(title, message, true);
  }

  /**
   * Fullscreen/Escape warnings debounce (prevents cascade triple-count)
   */
  handleFullscreenViolation(title, message) {
    const now = Date.now();
    if (now - (this._lastFullscreenViolationAt || 0) < this._fullscreenViolationCooldownMs) {
      // Log but don't increment warning count again
      this.handleViolation(title, message, false);
      return;
    }
    this._lastFullscreenViolationAt = now;
    // Also suppress context-switch counting right after fullscreen exit
    this._lastContextSwitchAt = now;
    this.handleViolation(title, message, true);
  }
  
  /**
   * Block right-click context menu
   */
  attachContextMenuListener() {
    const handler = (e) => {
      e.preventDefault();
      
      this.logActivity('right_click', {
        x: e.clientX,
        y: e.clientY,
        target: e.target.tagName,
        message: 'Right-click detected'
      });
      
      this.handleViolation('Right Click', 'Right-click is disabled during exam', false);
      return false;
    };
    
    this.addListener(document, 'contextmenu', handler);
  }
  
  /**
   * Block copy/paste
   */
  attachCopyPasteListener() {
    const copyHandler = (e) => {
      // Allow copy on input fields (for typing answers)
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }
      
      e.preventDefault();
      this.logActivity('copy_attempted', {
        target: e.target.tagName,
        message: 'Attempted to copy content'
      });
      this.handleViolation('Copy', 'Copying is not allowed', false);
    };
    
    const pasteHandler = (e) => {
      // Allow paste on input fields
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }
      
      e.preventDefault();
      this.logActivity('paste_attempted', {
        target: e.target.tagName,
        message: 'Attempted to paste content'
      });
      this.handleViolation('Paste', 'Pasting is not allowed', false);
    };
    
    this.addListener(document, 'copy', copyHandler);
    this.addListener(document, 'paste', pasteHandler);
    this.addListener(document, 'cut', copyHandler);
  }
  
  /**
   * Detect Print Screen key
   */
  attachPrintScreenListener() {
    // Print Screen is hard to detect, but we can try
    const handler = (e) => {
      if (e.key === 'PrintScreen') {
        this.logActivity('print_screen', {
          message: 'Print Screen key detected'
        });
      }
    };
    
    this.addListener(window, 'keyup', handler);
  }
  
  /**
   * Detect developer tools opening (advanced detection)
   */
  attachDevToolsDetector() {
    // Method 1: Window size change detection
    let devtoolsOpen = false;
    const threshold = 160;
    
    const checkDevTools = () => {
      const widthDiff = window.outerWidth - window.innerWidth;
      const heightDiff = window.outerHeight - window.innerHeight;
      
      const isOpen = widthDiff > threshold || heightDiff > threshold;
      
      if (isOpen && !devtoolsOpen) {
        devtoolsOpen = true;
        this.logActivity('dev_tools_opened', {
          method: 'size_detection',
          widthDiff,
          heightDiff,
          message: 'Developer tools detected via window size'
        });
        this.handleViolation('Developer Tools', 'Developer tools are not allowed');
      } else if (!isOpen && devtoolsOpen) {
        devtoolsOpen = false;
      }
    };
    
    // Check periodically
    const interval = setInterval(checkDevTools, 1000);
    this.listeners.push({
      cleanup: () => clearInterval(interval)
    });
    
    // Method 2: debugger statement detection (can be bypassed)
    const debuggerDetect = () => {
      const start = performance.now();
      debugger; // If dev tools open, this pauses execution
      const end = performance.now();
      
      if (end - start > 100) {
        this.logActivity('dev_tools_opened', {
          method: 'debugger_detection',
          delay: end - start,
          message: 'Developer tools detected via debugger'
        });
      }
    };
    
    // Run occasionally (not too often to avoid annoying developers in dev mode)
    const debugInterval = setInterval(debuggerDetect, 5000);
    this.listeners.push({
      cleanup: () => clearInterval(debugInterval)
    });
  }
  
  /**
   * Add event listener and track for cleanup
   */
  addListener(element, event, handler) {
    element.addEventListener(event, handler);
    this.listeners.push({ element, event, handler });
  }
  
  /**
   * Handle violation (increment warnings, show alert)
   */
  handleViolation(title, message, incrementWarning = true) {
    console.warn(`⚠️ Violation: ${title} - ${message}`);
    
    if (incrementWarning) {
      this.warningCount++;
      if (this.warningCount >= this.maxWarnings) {
        if (this.onMaxWarnings) {
          this.onMaxWarnings(this.warningCount, { title, message });
        }
      } else {
        // UI is handled by the host app via onViolation callback.
      }
    }
    
    if (this.onViolation) {
      this.onViolation(title, message, this.warningCount);
    }
  }
  
  /**
   * Log activity to backend
   */
  async logActivity(activityType, details = {}) {
    try {
      const response = await fetch(`${this.serverUrl}/api/log-activity`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          examId: this.examId,
          userId: this.userId,
          activityType: activityType,
          details: details,
          timestamp: new Date().toISOString()
        })
      });
      
      if (!response.ok) {
        console.error('Failed to log activity:', response.statusText);
      } else {
        const result = await response.json();
        console.log(`Logged activity: ${activityType} (severity: ${result.severity}, score: ${result.score})`);
      }
    } catch (error) {
      console.error('Error logging activity:', error);
    }
  }
  
  /**
   * Get current warning count
   */
  getWarningCount() {
    return this.warningCount;
  }
  
  /**
   * Reset warning count
   */
  resetWarnings() {
    this.warningCount = 0;
  }
}

// Export for use in modules
// ESM default export (Vite/Rollup)
export default BrowserLock;

// CommonJS fallback (only if this file is ever required from a CJS context)
// eslint-disable-next-line no-undef
if (typeof module !== 'undefined' && module.exports) {
  // eslint-disable-next-line no-undef
  module.exports = BrowserLock;
}
