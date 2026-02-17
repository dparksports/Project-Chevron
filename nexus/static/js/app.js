/**
 * Nexus Dashboard — Main Application Controller
 * ================================================
 * View switching, API client, state management.
 */

const NexusApp = (() => {

    // ── State ──
    const state = {
        projectLoaded: false,
        architecture: null,
        selectedModule: null,
        generating: false,
        currentView: 'welcome',
    };

    // ── API Client ──
    const api = {
        async get(path) {
            const res = await fetch(path);
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || 'Request failed');
            }
            return res.json();
        },
        async post(path, body) {
            const res = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || 'Request failed');
            }
            return res.json();
        },
    };

    // ── Template Icons ──
    const TEMPLATE_ICONS = {
        'todo-app': '📝',
        'web-api': '🌐',
        'cli-tool': '⌨️',
        'data-pipeline': '🔄',
        'blank': '📦',
    };

    // ── DOM Refs ──
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ── View Switching ──
    function showView(name) {
        state.currentView = name;
        $$('.view').forEach(v => v.classList.remove('active'));
        const el = $(`#view-${name}`);
        if (el) el.classList.add('active');

        $$('.header__nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === name);
        });
    }

    // ── Initialize ──
    async function init() {
        // Wire up nav buttons
        $$('.header__nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                showView(btn.dataset.view);
                if (btn.dataset.view === 'health') loadHealth();
                if (btn.dataset.view === 'providers') loadProviders();
            });
        });

        // Try to load existing project
        try {
            const arch = await api.get('/api/architecture');
            onProjectLoaded(arch);
        } catch {
            // No project — show welcome
            showView('welcome');
            loadTemplates();
        }
    }

    // ── Load Templates ──
    async function loadTemplates() {
        try {
            const templates = await api.get('/api/templates');
            const gallery = $('#template-gallery');
            gallery.innerHTML = '';

            templates.forEach(t => {
                const card = document.createElement('div');
                card.className = 'card template-card';
                card.innerHTML = `
                    <div class="template-card__icon">${TEMPLATE_ICONS[t.name] || '📦'}</div>
                    <div class="template-card__name">${t.display}</div>
                    <div class="template-card__desc">${t.description}</div>
                `;
                card.addEventListener('click', () => initProject(t.name));
                gallery.appendChild(card);
            });
        } catch (e) {
            console.error('Failed to load templates:', e);
        }
    }

    // ── Init Project ──
    async function initProject(template) {
        const name = prompt('Project name:', 'myapp');
        if (!name) return;

        try {
            const result = await api.post('/api/init', { name, template });
            if (result.success) {
                onProjectLoaded(result.spec);
            }
        } catch (e) {
            alert('Failed to create project: ' + e.message);
        }
    }

    // ── Project Loaded ──
    function onProjectLoaded(arch) {
        state.projectLoaded = true;
        state.architecture = arch;

        // Update header
        $('#project-name').textContent = arch.name || 'Project';
        $('#nav').style.display = 'flex';
        $('#sidebar').style.display = 'flex';

        // Populate module list
        renderModuleList(arch.modules || []);

        // Switch to dashboard view
        showView('dashboard');

        // Init graph
        if (typeof NexusGraph !== 'undefined') {
            NexusGraph.init(arch);
        }

        // Load health
        loadHealthSidebar();
    }

    // ── Render Module List ──
    function renderModuleList(modules) {
        const list = $('#module-list');
        list.innerHTML = '';

        modules.forEach(mod => {
            const li = document.createElement('li');
            li.className = 'sidebar__module-item';
            li.dataset.module = mod.name;

            let statusClass = 'status-dot--stub';
            if (mod.is_frozen) statusClass = 'status-dot--verified';
            else if (mod.has_code) statusClass = 'status-dot--active';

            li.innerHTML = `
                <span class="status-dot ${statusClass}"></span>
                <span>${mod.name}</span>
            `;

            li.addEventListener('click', () => selectModule(mod.name));
            list.appendChild(li);
        });
    }

    // ── Select Module ──
    function selectModule(moduleName) {
        state.selectedModule = moduleName;

        // Highlight in sidebar
        $$('.sidebar__module-item').forEach(item => {
            item.classList.toggle('active', item.dataset.module === moduleName);
        });

        // Open inspector
        const mod = (state.architecture.modules || []).find(m => m.name === moduleName);
        if (mod) renderInspector(mod);

        // Highlight in graph
        if (typeof NexusGraph !== 'undefined') {
            NexusGraph.selectModule(moduleName);
        }
    }

    // ── Render Inspector ──
    function renderInspector(mod) {
        const panel = $('#inspector');
        panel.classList.remove('collapsed');

        $('#inspector-name').textContent = mod.name;
        $('#inspector-glyph').textContent = '◬';

        const body = $('#inspector-body');
        let html = '';

        // Description
        if (mod.description) {
            html += `
                <div class="inspector__section">
                    <div class="inspector__section-title">Description</div>
                    <p style="font-size: 0.8125rem; color: var(--text-secondary);">${mod.description}</p>
                </div>
            `;
        }

        // Status badge
        let badgeClass = 'badge--amber';
        let badgeText = 'Stub';
        if (mod.is_frozen) { badgeClass = 'badge--green'; badgeText = 'Verified & Frozen'; }
        else if (mod.has_code) { badgeClass = 'badge--blue'; badgeText = 'Has Code'; }

        html += `
            <div class="inspector__section">
                <div class="inspector__section-title">Status</div>
                <span class="badge ${badgeClass}">${badgeText}</span>
                ${mod.edit_count ? `<span class="badge badge--blue" style="margin-left: 6px;">${mod.edit_count} edits</span>` : ''}
            </div>
        `;

        // Methods
        const methods = mod.methods || [];
        if (methods.length) {
            html += `<div class="inspector__section"><div class="inspector__section-title">Methods</div>`;
            methods.forEach(m => {
                const glyph = m.glyph || '';
                const inputs = (m.inputs || []).join(', ');
                const output = m.output || 'None';
                html += `
                    <div class="inspector__method">
                        <div class="inspector__method-name">${glyph} ${m.name}</div>
                        <div class="inspector__method-sig">(${inputs}) → ${output}</div>
                        ${m.constraint ? `<div style="font-size: 0.6875rem; color: var(--text-muted); margin-top: 4px;">${m.constraint}</div>` : ''}
                    </div>
                `;
            });
            html += '</div>';
        }

        // Constraints
        const constraints = mod.constraints || [];
        if (constraints.length) {
            html += `<div class="inspector__section"><div class="inspector__section-title">Constraints</div>`;
            constraints.forEach(c => {
                html += `<div class="inspector__constraint">${c}</div>`;
            });
            html += '</div>';
        }

        // Dependencies
        const deps = mod.allowed_dependencies || [];
        if (deps.length) {
            html += `<div class="inspector__section"><div class="inspector__section-title">Dependencies</div>`;
            deps.forEach(d => {
                html += `<span class="inspector__dep">${d}</span>`;
            });
            html += '</div>';
        }

        // Forbidden
        const forbidden = mod.forbidden || [];
        if (forbidden.length) {
            html += `<div class="inspector__section"><div class="inspector__section-title">Forbidden (RAG Denied)</div>`;
            forbidden.forEach(f => {
                html += `<span class="inspector__forbidden">${f}</span>`;
            });
            html += '</div>';
        }

        body.innerHTML = html;

        // Close button
        $('#inspector-close').onclick = () => {
            panel.classList.add('collapsed');
            state.selectedModule = null;
            $$('.sidebar__module-item').forEach(i => i.classList.remove('active'));
            if (typeof NexusGraph !== 'undefined') NexusGraph.selectModule(null);
        };
    }

    // ── Health ──
    async function loadHealthSidebar() {
        try {
            const health = await api.get('/api/health');
            $('#health-streak').textContent = health.clean_streak || 0;
            $('#health-regression').textContent = health.regression_rate || '0%';

            // Compute bar width from streak / total
            const total = health.total_edits || 0;
            const streak = health.clean_streak || 0;
            const pct = total > 0 ? Math.round((streak / Math.max(total, 1)) * 100) : 0;
            $('#health-bar').style.width = pct + '%';
        } catch {
            // No session yet — that's fine
        }
    }

    async function loadHealth() {
        try {
            const health = await api.get('/api/health');
            const el = $('#health-details');

            let html = '';
            for (const [key, val] of Object.entries(health)) {
                html += `
                    <div class="card" style="padding: 16px; margin-bottom: 12px; cursor: default;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.875rem; color: var(--text-secondary);">${key.replace(/_/g, ' ')}</span>
                            <span class="mono" style="font-size: 1rem; color: var(--cyan);">${val}</span>
                        </div>
                    </div>
                `;
            }
            el.innerHTML = html || '<div class="empty-state"><div class="empty-state__icon">𓂀</div><div class="empty-state__text">No session data yet. Generate some code first.</div></div>';
        } catch (e) {
            $('#health-details').innerHTML = `<div class="empty-state"><div class="empty-state__icon">𓂀</div><div class="empty-state__text">${e.message}</div></div>`;
        }
    }

    // ── Providers ──
    async function loadProviders() {
        try {
            const data = await api.get('/api/providers');
            const el = $('#provider-list');

            const PROVIDER_INFO = {
                'gemini': { icon: '🔷', install: 'pip install google-genai', models: 'gemini-2.5-pro, gemini-2.0-flash' },
                'openai': { icon: '🟢', install: 'pip install openai', models: 'gpt-4o, gpt-4o-mini' },
                'anthropic': { icon: '🟠', install: 'pip install anthropic', models: 'claude-sonnet-4-20250514' },
                'ollama': { icon: '🦙', install: 'ollama.com', models: 'llama3.1, codellama' },
            };

            let html = '';
            const all = ['gemini', 'openai', 'anthropic', 'ollama'];
            all.forEach(name => {
                const available = (data.providers || []).includes(name);
                const info = PROVIDER_INFO[name] || {};
                html += `
                    <div class="card" style="padding: 16px; margin-bottom: 12px; cursor: default;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 1.5rem;">${info.icon || '⚡'}</span>
                            <div style="flex: 1;">
                                <div style="font-weight: 600; text-transform: capitalize;">${name}</div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">${info.models || ''}</div>
                            </div>
                            <span class="badge ${available ? 'badge--green' : 'badge--red'}">
                                ${available ? 'Available' : 'Not Installed'}
                            </span>
                        </div>
                        ${!available ? `<div style="margin-top: 8px; font-size: 0.75rem; color: var(--text-muted);">Install: <code>${info.install || ''}</code></div>` : ''}
                    </div>
                `;
            });
            el.innerHTML = html;
        } catch (e) {
            $('#provider-list').innerHTML = `<div class="empty-state"><div class="empty-state__text">${e.message}</div></div>`;
        }
    }

    // ── Refresh Architecture ──
    async function refreshArchitecture() {
        try {
            const arch = await api.get('/api/architecture');
            state.architecture = arch;
            renderModuleList(arch.modules || []);
            if (typeof NexusGraph !== 'undefined') NexusGraph.init(arch);
            loadHealthSidebar();
        } catch (e) {
            console.error('Failed to refresh architecture:', e);
        }
    }

    // ── Public API ──
    return {
        init,
        state,
        api,
        showView,
        selectModule,
        refreshArchitecture,
        onProjectLoaded,
    };

})();

// Boot on load
document.addEventListener('DOMContentLoaded', NexusApp.init);
