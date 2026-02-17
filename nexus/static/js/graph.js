/**
 * Nexus Dashboard — Architecture Graph Visualization
 * ====================================================
 * Force-directed graph of SCP modules using HTML5 Canvas.
 * Modules are rounded-rect nodes, dependencies are curved arrows.
 */

const NexusGraph = (() => {

    let canvas, ctx;
    let nodes = [];
    let edges = [];
    let selectedNode = null;
    let hoveredNode = null;
    let dragging = null;
    let dragOffset = { x: 0, y: 0 };
    let animFrame = null;
    let time = 0;

    // ── Colors ──
    const COLORS = {
        stub: { bg: '#1f2a42', border: '#334155', text: '#94a3b8' },
        hasCode: { bg: '#0c3b2e', border: '#06d6a0', text: '#06d6a0' },
        verified: { bg: '#042f2e', border: '#06d6a0', text: '#06d6a0' },
        generating: { bg: '#451a03', border: '#f59e0b', text: '#f59e0b' },
        failed: { bg: '#450a0a', border: '#ef4444', text: '#ef4444' },
        selected: { bg: '#0e2a3d', border: '#4cc9f0', text: '#4cc9f0' },
        edge: 'rgba(148, 163, 184, 0.18)',
        edgeArrow: 'rgba(148, 163, 184, 0.35)',
    };

    // ── Glyph Map ──
    const GLYPHS = {
        'EntropyScorer': '◬', 'ContextPruner': 'Ө', 'SCPRetriever': '◬',
        'EditLedger': '◬', 'ContractCache': '☾', 'SessionManager': '☤',
        'Planner': '◬', 'Executor': '☾', 'Verifier': '𓂀',
        'Conductor': '☤', 'AIProvider': '☤',
    };

    function init(arch) {
        canvas = document.getElementById('graph-canvas');
        if (!canvas) return;
        ctx = canvas.getContext('2d');

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        buildGraph(arch);
        setupInteraction();

        if (animFrame) cancelAnimationFrame(animFrame);
        animate();
    }

    function resizeCanvas() {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }

    function buildGraph(arch) {
        const modules = arch.modules || [];
        nodes = [];
        edges = [];

        const cw = canvas.width / window.devicePixelRatio;
        const ch = canvas.height / window.devicePixelRatio;
        const cx = cw / 2;
        const cy = ch / 2;
        const radius = Math.min(cw, ch) * 0.32;

        // Create nodes in a circle
        modules.forEach((mod, i) => {
            const angle = (i / modules.length) * Math.PI * 2 - Math.PI / 2;
            nodes.push({
                name: mod.name,
                x: cx + Math.cos(angle) * radius,
                y: cy + Math.sin(angle) * radius,
                vx: 0, vy: 0,
                w: Math.max(100, mod.name.length * 9 + 40),
                h: 52,
                mod: mod,
                glyph: GLYPHS[mod.name] || '◬',
            });
        });

        // Create edges from dependencies
        const nameMap = {};
        nodes.forEach(n => nameMap[n.name] = n);

        modules.forEach(mod => {
            const deps = mod.allowed_dependencies || [];
            deps.forEach(dep => {
                if (nameMap[dep]) {
                    edges.push({
                        from: nameMap[mod.name],
                        to: nameMap[dep],
                    });
                }
            });
        });

        // Run simple force layout
        for (let iter = 0; iter < 120; iter++) {
            forceStep(0.3);
        }
    }

    function forceStep(dt) {
        const cw = canvas.width / window.devicePixelRatio;
        const ch = canvas.height / window.devicePixelRatio;
        const cx = cw / 2;
        const cy = ch / 2;

        // Repulsion between all nodes
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let dist = Math.sqrt(dx * dx + dy * dy) || 1;
                let force = 8000 / (dist * dist);
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;
                a.vx -= fx; a.vy -= fy;
                b.vx += fx; b.vy += fy;
            }
        }

        // Attraction along edges
        edges.forEach(e => {
            let dx = e.to.x - e.from.x;
            let dy = e.to.y - e.from.y;
            let dist = Math.sqrt(dx * dx + dy * dy) || 1;
            let force = (dist - 180) * 0.02;
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            e.from.vx += fx; e.from.vy += fy;
            e.to.vx -= fx; e.to.vy -= fy;
        });

        // Center gravity
        nodes.forEach(n => {
            n.vx += (cx - n.x) * 0.003;
            n.vy += (cy - n.y) * 0.003;
        });

        // Apply velocity
        nodes.forEach(n => {
            if (n === dragging) return;
            n.vx *= 0.85;
            n.vy *= 0.85;
            n.x += n.vx * dt;
            n.y += n.vy * dt;
            // Keep in bounds
            n.x = Math.max(n.w / 2 + 10, Math.min(cw - n.w / 2 - 10, n.x));
            n.y = Math.max(n.h / 2 + 10, Math.min(ch - n.h / 2 - 10, n.y));
        });
    }

    function animate() {
        time++;
        draw();
        animFrame = requestAnimationFrame(animate);
    }

    function draw() {
        const cw = canvas.width / window.devicePixelRatio;
        const ch = canvas.height / window.devicePixelRatio;

        ctx.clearRect(0, 0, cw, ch);

        // Draw subtle grid
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.03)';
        ctx.lineWidth = 1;
        for (let x = 0; x < cw; x += 40) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
        }
        for (let y = 0; y < ch; y += 40) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
        }

        // Draw edges
        edges.forEach(e => drawEdge(e));

        // Draw nodes
        nodes.forEach(n => drawNode(n));
    }

    function drawEdge(e) {
        const fromX = e.from.x;
        const fromY = e.from.y;
        const toX = e.to.x;
        const toY = e.to.y;

        // Curved line
        const midX = (fromX + toX) / 2;
        const midY = (fromY + toY) / 2;
        const dx = toX - fromX;
        const dy = toY - fromY;
        const perpX = -dy * 0.15;
        const perpY = dx * 0.15;
        const cpX = midX + perpX;
        const cpY = midY + perpY;

        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.quadraticCurveTo(cpX, cpY, toX, toY);
        ctx.strokeStyle = COLORS.edge;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Arrow head
        const t = 0.85;
        const arrowX = (1 - t) * (1 - t) * fromX + 2 * (1 - t) * t * cpX + t * t * toX;
        const arrowY = (1 - t) * (1 - t) * fromY + 2 * (1 - t) * t * cpY + t * t * toY;
        const tangentX = 2 * (1 - t) * (cpX - fromX) + 2 * t * (toX - cpX);
        const tangentY = 2 * (1 - t) * (cpY - fromY) + 2 * t * (toY - cpY);
        const angle = Math.atan2(tangentY, tangentX);

        ctx.save();
        ctx.translate(arrowX, arrowY);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-8, -4);
        ctx.lineTo(-8, 4);
        ctx.closePath();
        ctx.fillStyle = COLORS.edgeArrow;
        ctx.fill();
        ctx.restore();
    }

    function drawNode(n) {
        const isSelected = selectedNode === n.name;
        const isHovered = hoveredNode === n;
        const mod = n.mod;

        // Determine color scheme
        let colors = COLORS.stub;
        if (isSelected) colors = COLORS.selected;
        else if (mod.is_frozen) colors = COLORS.verified;
        else if (mod.has_code) colors = COLORS.hasCode;

        const x = n.x - n.w / 2;
        const y = n.y - n.h / 2;
        const r = 10;

        // Glow effect for selected / hovered
        if (isSelected || isHovered) {
            ctx.shadowColor = colors.border;
            ctx.shadowBlur = 16;
        }

        // Background
        ctx.fillStyle = colors.bg;
        ctx.strokeStyle = colors.border;
        ctx.lineWidth = isSelected ? 2 : 1;

        ctx.beginPath();
        ctx.roundRect(x, y, n.w, n.h, r);
        ctx.fill();
        ctx.stroke();

        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;

        // Glyph (left side)
        ctx.font = '16px serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = colors.text;
        ctx.fillText(n.glyph, x + 12, n.y);

        // Module name
        ctx.font = '500 12px "Inter", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.text;
        ctx.fillText(n.name, x + 32, n.y);

        // Generating animation
        if (n._generating) {
            const pulseAlpha = 0.3 + Math.sin(time * 0.08) * 0.2;
            ctx.strokeStyle = `rgba(245, 158, 11, ${pulseAlpha})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.roundRect(x - 2, y - 2, n.w + 4, n.h + 4, r + 2);
            ctx.stroke();
        }
    }

    // ── Interaction ──
    function setupInteraction() {
        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('dblclick', onDblClick);
    }

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function hitTest(pos) {
        for (let i = nodes.length - 1; i >= 0; i--) {
            const n = nodes[i];
            if (pos.x >= n.x - n.w / 2 && pos.x <= n.x + n.w / 2 &&
                pos.y >= n.y - n.h / 2 && pos.y <= n.y + n.h / 2) {
                return n;
            }
        }
        return null;
    }

    function onMouseDown(e) {
        const pos = getMousePos(e);
        const node = hitTest(pos);
        if (node) {
            dragging = node;
            dragOffset.x = pos.x - node.x;
            dragOffset.y = pos.y - node.y;
            canvas.style.cursor = 'grabbing';
        }
    }

    function onMouseMove(e) {
        const pos = getMousePos(e);

        if (dragging) {
            dragging.x = pos.x - dragOffset.x;
            dragging.y = pos.y - dragOffset.y;
            return;
        }

        const hit = hitTest(pos);
        hoveredNode = hit;
        canvas.style.cursor = hit ? 'pointer' : 'grab';
    }

    function onMouseUp(e) {
        if (dragging) {
            const pos = getMousePos(e);
            const node = hitTest(pos);
            if (node && node === dragging) {
                // Click — select this module
                NexusApp.selectModule(node.name);
            }
            dragging = null;
            canvas.style.cursor = 'grab';
        }
    }

    function onDblClick(e) {
        const pos = getMousePos(e);
        const node = hitTest(pos);
        if (node) {
            // Focus the chat input on the module
            const chatInput = document.getElementById('chat-input');
            if (chatInput) {
                chatInput.value = `Implement the ${node.name} module`;
                chatInput.focus();
            }
        }
    }

    // ── Public ──
    function selectModule(name) {
        selectedNode = name;
    }

    function setModuleGenerating(name, generating) {
        const node = nodes.find(n => n.name === name);
        if (node) node._generating = generating;
    }

    function updateModuleStatus(name, status) {
        const node = nodes.find(n => n.name === name);
        if (node) {
            if (status === 'verified') node.mod.is_frozen = true;
            if (status === 'hasCode') node.mod.has_code = true;
        }
    }

    return {
        init,
        selectModule,
        setModuleGenerating,
        updateModuleStatus,
    };

})();
