// Clipboard Auto-Detection System
// Automatically detects schedule-related content in clipboard

class ClipboardMonitor {
    constructor() {
        this.isEnabled = true;
        this.lastCheckedText = '';
        this.scheduleKeywords = [
            // Korean
            '일정', '미팅', '회의', '약속', '만남', '모임', '행사',
            // English
            'meeting', 'appointment', 'schedule', 'event', 'conference',
            // Date patterns
            '월', '일', '시', '분',
            'am', 'pm', 'today', 'tomorrow', 'next week'
        ];
    }

    async checkClipboard() {
        if (!this.isEnabled) return;

        try {
            // Check if Clipboard API is supported
            if (!navigator.clipboard || !navigator.clipboard.readText) {
                console.log('[Clipboard] API not supported');
                return;
            }

            const text = await navigator.clipboard.readText();

            // Skip if empty or same as last check
            if (!text || text === this.lastCheckedText) return;

            this.lastCheckedText = text;

            // Check if text contains schedule-related keywords or date patterns
            if (this.containsScheduleInfo(text)) {
                this.showClipboardNotification(text);
            }
        } catch (error) {
            // Permission denied or other errors
            console.log('[Clipboard] Access denied or error:', error.message);
        }
    }

    containsScheduleInfo(text) {
        const lowerText = text.toLowerCase();

        // Check for keywords
        const hasKeyword = this.scheduleKeywords.some(keyword =>
            lowerText.includes(keyword.toLowerCase())
        );

        // Check for date patterns (YYYY-MM-DD, MM/DD, etc.)
        const datePatterns = [
            /\d{4}-\d{1,2}-\d{1,2}/,  // 2025-12-05
            /\d{1,2}\/\d{1,2}/,        // 12/5
            /\d{1,2}월\s*\d{1,2}일/,  // 12월 5일
            /\d{1,2}:\d{2}/,           // 14:30
        ];

        const hasDatePattern = datePatterns.some(pattern => pattern.test(text));

        return hasKeyword || hasDatePattern;
    }

    showClipboardNotification(text) {
        // Check if notification already exists
        if (document.getElementById('clipboard-notification')) return;

        const notification = document.createElement('div');
        notification.id = 'clipboard-notification';
        notification.className = 'clipboard-notification';
        notification.innerHTML = `
            <div class="clipboard-content">
                <span class="clipboard-icon">📋</span>
                <span class="clipboard-text">Detected schedule in clipboard. Analyze it?</span>
                <div class="clipboard-actions">
                    <button onclick="clipboardMonitor.useClipboard()" class="btn-use">Analyze</button>
                    <button onclick="clipboardMonitor.dismissNotification()" class="btn-dismiss">Dismiss</button>
                </div>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-dismiss after 10 seconds
        setTimeout(() => {
            this.dismissNotification();
        }, 10000);
    }

    useClipboard() {
        const textarea = document.getElementById('input-text');
        if (textarea && this.lastCheckedText) {
            textarea.value = this.lastCheckedText;
            textarea.focus();

            // Trigger analysis automatically
            const analyzeBtn = document.getElementById('extract-button');
            if (analyzeBtn) {
                analyzeBtn.click();
            }
        }
        this.dismissNotification();
    }

    dismissNotification() {
        const notification = document.getElementById('clipboard-notification');
        if (notification) {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }
    }

    enable() {
        this.isEnabled = true;
    }

    disable() {
        this.isEnabled = false;
    }
}

// Initialize clipboard monitor
const clipboardMonitor = new ClipboardMonitor();

// Check clipboard when page becomes visible
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        setTimeout(() => clipboardMonitor.checkClipboard(), 500);
    }
});

// Check clipboard when window gains focus
window.addEventListener('focus', () => {
    setTimeout(() => clipboardMonitor.checkClipboard(), 500);
});

// Initial check after page load
window.addEventListener('load', () => {
    setTimeout(() => clipboardMonitor.checkClipboard(), 1000);
});
