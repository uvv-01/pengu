/**
 * Pengu Desktop UI — JavaScript controller.
 *
 * Connects to the FastAPI backend via REST API and WebSocket.
 * Architecture:
 *   UI → FastAPI → Router → Tool/Model → Response → UI
 */

const API_BASE = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws`;

// State
let ws = null;
let isConnected = false;
let isProcessing = false;

// DOM elements
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('command-input');
const sendBtn = document.getElementById('send-btn');
const cancelBtn = document.getElementById('cancel-btn');
const statusBadge = document.getElementById('status-badge');
const providerBadge = document.getElementById('provider-badge');

// --- WebSocket ---

function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            isConnected = true;
            console.log('WebSocket connected');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'state_update') {
                    updateStatus(msg.state);
                }
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        ws.onclose = () => {
            isConnected = false;
            console.log('WebSocket disconnected, reconnecting in 5s...');
            setTimeout(connectWebSocket, 5000);
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    } catch (e) {
        console.error('WebSocket connection failed:', e);
        setTimeout(connectWebSocket, 5000);
    }
}

// --- Status ---

function updateStatus(state) {
    const map = {
        'STANDBY': { class: 'badge-standby', text: 'STANDBY' },
        'WAKE_DETECTED': { class: 'badge-listening', text: 'WAKE DETECTED' },
        'ACTIVE': { class: 'badge-listening', text: 'ACTIVE' },
        'LISTENING': { class: 'badge-listening', text: 'LISTENING' },
        'THINKING': { class: 'badge-thinking', text: 'THINKING' },
        'PLANNING': { class: 'badge-thinking', text: 'PLANNING' },
        'EXECUTING': { class: 'badge-executing', text: 'EXECUTING' },
        'SPEAKING': { class: 'badge-executing', text: 'SPEAKING' },
        'COMPLETE': { class: 'badge-standby', text: 'COMPLETE' },
        'ERROR': { class: 'badge-error', text: 'ERROR' },
    };

    const info = map[state] || { class: 'badge-standby', text: state };
    statusBadge.className = `badge ${info.class}`;
    statusBadge.textContent = info.text;
}

// --- Messages ---

function addMessage(role, content, extra = {}) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    let label = '';
    if (role === 'user') label = 'YOU';
    else if (role === 'assistant') label = 'PENGU';
    else if (role === 'system') label = '';

    let meta = '';
    if (extra.tool_used) meta += ` <span class="hint">[${extra.tool_used}]</span>`;
    if (extra.provider && role === 'assistant') {
        providerBadge.textContent = extra.provider;
        if (extra.model) providerBadge.textContent += ` / ${extra.model}`;
    }

    div.innerHTML = `
        ${label ? `<div class="message-label">${label}</div>` : ''}
        <div class="message-content">
            <p>${escapeHtml(content)}${meta}</p>
        </div>
    `;

    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
}

function addToolActivity(text) {
    const div = document.createElement('div');
    div.className = 'tool-activity';
    div.innerHTML = `<div class="spinner"></div><span>${escapeHtml(text)}</span>`;
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
}

function removeElement(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

function scrollToBottom() {
    const chatArea = document.getElementById('chat-area');
    chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// --- Send command ---

async function sendCommand(text) {
    if (!text.trim() || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    inputEl.disabled = true;
    updateStatus('THINKING');

    // Show user message
    addMessage('user', text);
    inputEl.value = '';
    autoResize();

    // Show activity indicator
    const activity = addToolActivity('Processing...');

    try {
        const response = await fetch(`${API_BASE}/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        removeElement(activity);

        if (!response.ok) {
            addMessage('assistant', `Error: HTTP ${response.status}`);
            updateStatus('ERROR');
        } else {
            const data = await response.json();

            if (data.error) {
                addMessage('assistant', `Error: ${data.error}`);
                updateStatus('ERROR');
            } else {
                addMessage('assistant', data.response, {
                    tool_used: data.tool_used,
                    provider: data.provider,
                    model: data.model,
                });
                updateStatus('COMPLETE');
            }
        }
    } catch (e) {
        removeElement(activity);
        addMessage('assistant', `Connection error: ${e.message}. Is the Pengu backend running?`);
        updateStatus('ERROR');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        inputEl.disabled = false;
        inputEl.focus();

        // Return to standby after a delay
        setTimeout(() => updateStatus('STANDBY'), 2000);
    }
}

// --- Cancel ---

async function cancelCommand() {
    try {
        await fetch(`${API_BASE}/cancel`, { method: 'POST' });
        updateStatus('STANDBY');
    } catch (e) {
        console.error('Cancel failed:', e);
    }
}

// --- Auto-resize textarea ---

function autoResize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

// --- Event listeners ---

sendBtn.addEventListener('click', () => {
    sendCommand(inputEl.value);
});

cancelBtn.addEventListener('click', cancelCommand);

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendCommand(inputEl.value);
    }
    if (e.key === 'Escape') {
        cancelCommand();
    }
});

inputEl.addEventListener('input', autoResize);

// --- Init ---

async function init() {
    // Check backend health
    try {
        const resp = await fetch(`${API_BASE}/health`);
        if (resp.ok) {
            const data = await resp.json();
            providerBadge.textContent = data.provider || 'none';
            updateStatus('STANDBY');
        }
    } catch (e) {
        addMessage('system', 'Cannot connect to Pengu backend. Make sure it is running on port 8420.');
        providerBadge.textContent = 'offline';
    }

    // Connect WebSocket for live updates
    connectWebSocket();

    // Focus input
    inputEl.focus();
}

init();
