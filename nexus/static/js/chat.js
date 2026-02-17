/**
 * Nexus Dashboard — Conductor Chat
 * ==================================
 * WebSocket-powered chat interface for code generation.
 */

const NexusChat = (() => {

    let ws = null;
    let reconnectTimer = null;

    // ── DOM ──
    const $ = (sel) => document.querySelector(sel);

    function init() {
        // Toggle chat panel
        $('#chat-header').addEventListener('click', toggleChat);

        // Send on button click
        $('#chat-send').addEventListener('click', sendMessage);

        // Send on Enter key
        $('#chat-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    function toggleChat() {
        const panel = $('#chat-panel');
        panel.classList.toggle('minimized');
    }

    function sendMessage() {
        const input = $('#chat-input');
        const text = input.value.trim();
        if (!text) return;

        input.value = '';
        addMessage(text, 'user');

        // Determine target module
        const selected = NexusApp.state.selectedModule;

        // Connect WebSocket and send
        connectAndSend({
            request: text,
            module: selected || null,
            provider: 'gemini',  // TODO: configurable
            api_key: '',         // Uses server env var
        });
    }

    function connectAndSend(payload) {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws/generate`;

        setStatus('Connecting...');
        setGenerating(true);

        ws = new WebSocket(url);

        ws.onopen = () => {
            setStatus('Planning...');
            ws.send(JSON.stringify(payload));
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        };

        ws.onerror = () => {
            addMessage('Connection error. Is the server running?', 'error');
            setStatus('Error');
            setGenerating(false);
        };

        ws.onclose = () => {
            setGenerating(false);
        };
    }

    function handleServerMessage(msg) {
        switch (msg.type) {
            case 'plan':
                setStatus('Planned');
                addMessage(
                    `Plan: ${msg.data.count} module(s) → ${msg.data.modules.join(', ')}`,
                    'system'
                );
                break;

            case 'executing':
                setStatus(`Generating: ${msg.module}`);
                addMessage(`⚡ Generating ${msg.module}...`, 'system');
                if (typeof NexusGraph !== 'undefined') {
                    NexusGraph.setModuleGenerating(msg.module, true);
                }
                break;

            case 'code':
                if (typeof NexusGraph !== 'undefined') {
                    NexusGraph.setModuleGenerating(msg.module, false);
                }
                if (msg.success && msg.code) {
                    // Show a truncated preview
                    const preview = msg.code.length > 400
                        ? msg.code.substring(0, 400) + '\n...'
                        : msg.code;
                    addMessage(preview, 'code');
                    if (typeof NexusGraph !== 'undefined') {
                        NexusGraph.updateModuleStatus(msg.module, 'hasCode');
                    }
                } else if (msg.error) {
                    addMessage(`✘ ${msg.module}: ${msg.error}`, 'error');
                }
                break;

            case 'verified':
                const icon = msg.passed ? '✔' : '✘';
                const cls = msg.passed ? 'system' : 'error';
                addMessage(
                    `${icon} Verification: ${msg.module} — ${msg.passed ? 'PASSED' : 'FAILED'}` +
                    (msg.violations.length ? `\n  ${msg.violations.join('\n  ')}` : ''),
                    cls
                );
                if (msg.passed && typeof NexusGraph !== 'undefined') {
                    NexusGraph.updateModuleStatus(msg.module, 'verified');
                }
                break;

            case 'health':
                setStatus('Done');
                const h = msg.data;
                addMessage(
                    `Session: ${h.total_edits || 0} edits, ` +
                    `${h.regression_rate || '0%'} regression, ` +
                    `streak: ${h.clean_streak || 0}`,
                    'system'
                );
                break;

            case 'done':
                setStatus('Ready');
                setGenerating(false);
                addMessage('✔ Generation complete.', 'system');
                // Refresh architecture
                NexusApp.refreshArchitecture();
                if (ws) { ws.close(); ws = null; }
                break;

            case 'error':
                setStatus('Error');
                setGenerating(false);
                addMessage(`✘ ${msg.message}`, 'error');
                if (ws) { ws.close(); ws = null; }
                break;
        }
    }

    // ── UI Helpers ──

    function addMessage(text, type) {
        const container = $('#chat-messages');
        const div = document.createElement('div');
        div.className = `chat__msg chat__msg--${type}`;

        if (type === 'code') {
            div.textContent = text;
        } else {
            div.textContent = text;
        }

        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    function setStatus(text) {
        $('#chat-status').textContent = text;
    }

    function setGenerating(active) {
        NexusApp.state.generating = active;
        const btn = $('#chat-send');
        const input = $('#chat-input');

        if (active) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>';
            input.disabled = true;
        } else {
            btn.disabled = false;
            btn.textContent = 'Generate';
            input.disabled = false;
            input.focus();
        }
    }

    // Boot
    document.addEventListener('DOMContentLoaded', init);

    return { addMessage, setStatus };

})();
