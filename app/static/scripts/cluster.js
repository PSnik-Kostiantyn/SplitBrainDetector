document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('graph-canvas');
    const ctx    = canvas.getContext('2d');
    let nodes = [], edges = [];
    let currentNodeName = '';
    let isErasing       = false;
    let currentEdgeType = 'one-way';
    let selectedNode    = null;
    let isDragging      = false;

    canvas.width  = 600;
    canvas.height = 400;
    ctx.font = '14px Inter, sans-serif';

    let currentMode = 'base';
    const toggle    = document.getElementById('mode-toggle');
    const modeLabel = document.getElementById('mode-label');

    if (toggle) {
        toggle.addEventListener('change', function () {
            currentMode = this.checked ? 'ensemble' : 'base';
            modeLabel.textContent = this.checked
                ? 'Режим: Ансамблі (E1 · E2 · E3)'
                : 'Режим: Базові (RF · GB · CB)';
        });
    }

    document.getElementById('create-node').addEventListener('click', function () {
        const nodeName = document.getElementById('node-name').value;
        if (validateNodeName(nodeName)) {
            if (nodes.find(n => n.name === nodeName)) {
                alert('Ця назва ноди вже існує. Виберіть іншу назву.');
            } else {
                nodes.push({
                    name: nodeName,
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                });
                drawGraph();
                document.getElementById('node-name').value = '';
            }
        } else {
            alert('Невірний формат. Назва: велика літера A-Z + цифра 1-9.');
        }
    });

    document.getElementById('line-one-way').addEventListener('click', function () {
        currentEdgeType = 'one-way';
    });
    document.getElementById('line-two-way').addEventListener('click', function () {
        currentEdgeType = 'two-way';
    });
    document.getElementById('erase').addEventListener('click', function () {
        isErasing = !isErasing;
        this.style.backgroundColor = isErasing ? '#ff4444' : '';
    });

    canvas.addEventListener('click', function (event) {
        if (isErasing) eraseNode(event);
        else           addEdge(event);
    });

    function drawGraph() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        edges.forEach(edge => {
            const a = nodes.find(n => n.name === edge.from);
            const b = nodes.find(n => n.name === edge.to);
            if (!a || !b) return;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = edge.type === 'two-way' ? '#00f' : '#f00';
            ctx.lineWidth = 2;
            ctx.stroke();
            drawArrowhead(a.x, a.y, b.x, b.y, ctx.strokeStyle);
            if (edge.type === 'two-way')
                drawArrowhead(b.x, b.y, a.x, a.y, ctx.strokeStyle);
        });
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 20, 0, Math.PI * 2);
            ctx.fillStyle = getNodeColor(node.name);
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = '#fff';
            ctx.font = '14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.name, node.x, node.y);
        });
    }

    function drawArrowhead(x1, y1, x2, y2, color) {
        const angle   = Math.atan2(y2 - y1, x2 - x1);
        const arrowSize = 10, offset = 20;
        const ax = x2 - offset * Math.cos(angle);
        const ay = y2 - offset * Math.sin(angle);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - arrowSize * Math.cos(angle - Math.PI/6),
                   ay - arrowSize * Math.sin(angle - Math.PI/6));
        ctx.lineTo(ax - arrowSize * Math.cos(angle + Math.PI/6),
                   ay - arrowSize * Math.sin(angle + Math.PI/6));
        ctx.closePath();
        ctx.fill();
    }

    function addEdge(event) {
        const clicked = nodes.find(n =>
            Math.hypot(n.x - event.offsetX, n.y - event.offsetY) < 20);
        if (!clicked) return;
        if (!currentNodeName) { currentNodeName = clicked.name; return; }
        if (currentNodeName === clicked.name) { currentNodeName = ''; return; }
        edges = edges.filter(e =>
            !(e.from === currentNodeName && e.to === clicked.name) &&
            !(e.from === clicked.name   && e.to === currentNodeName));
        edges.push({from: currentNodeName, to: clicked.name, type: currentEdgeType});
        currentNodeName = '';
        drawGraph(); updateMatrix();
    }

    function eraseNode(event) {
        const clicked = nodes.find(n =>
            Math.hypot(n.x - event.offsetX, n.y - event.offsetY) < 20);
        if (clicked) {
            nodes = nodes.filter(n => n !== clicked);
            edges = edges.filter(e =>
                e.from !== clicked.name && e.to !== clicked.name);
            drawGraph(); updateMatrix();
        } else {
            eraseEdge(event);
        }
    }

    function eraseEdge(event) {
        const hit = edges.find(e => {
            const a = nodes.find(n => n.name === e.from);
            const b = nodes.find(n => n.name === e.to);
            return a && b && isPointOnLine(
                event.offsetX, event.offsetY, a.x, a.y, b.x, b.y);
        });
        if (hit) { edges = edges.filter(e => e !== hit); drawGraph(); updateMatrix(); }
    }

    function isPointOnLine(px, py, x1, y1, x2, y2) {
        return Math.abs((y2-y1)*px-(x2-x1)*py+x2*y1-y2*x1) /
               Math.sqrt((y2-y1)**2+(x2-x1)**2) < 10;
    }

    function updateMatrix() {
        const tbl = document.getElementById('matrix');
        tbl.innerHTML = '';
        const size   = nodes.length;
        const mData  = Array.from({length:size}, () => Array(size).fill(0));
        edges.forEach(e => {
            const fi = nodes.findIndex(n => n.name === e.from);
            const ti = nodes.findIndex(n => n.name === e.to);
            if (fi>=0 && ti>=0) {
                mData[fi][ti]=1;
                if (e.type==='two-way') mData[ti][fi]=1;
            }
        });
        const hdr = document.createElement('tr');
        hdr.appendChild(document.createElement('th'));
        nodes.forEach(n => {
            const th = document.createElement('th');
            th.textContent = n.name; hdr.appendChild(th);
        });
        tbl.appendChild(hdr);
        for (let i=0; i<size; i++) {
            const row = document.createElement('tr');
            const rh  = document.createElement('th');
            rh.textContent = nodes[i].name; row.appendChild(rh);
            for (let j=0; j<size; j++) {
                const td = document.createElement('td');
                td.textContent = (i===j) ? '0' : mData[i][j];
                row.appendChild(td);
            }
            tbl.appendChild(row);
        }
    }

    canvas.addEventListener('mousedown', e => {
        const hit = nodes.find(n =>
            Math.hypot(n.x-e.offsetX, n.y-e.offsetY) < 20);
        if (hit) { selectedNode = hit; isDragging = true; }
    });
    canvas.addEventListener('mousemove', e => {
        if (!isDragging || !selectedNode) return;
        selectedNode.x = e.offsetX; selectedNode.y = e.offsetY;
        drawGraph(); updateMatrix();
    });
    canvas.addEventListener('mouseup', () => {
        isDragging = false; selectedNode = null;
    });

    document.getElementById('submit-matrix').addEventListener('click', function () {
        const nodesH = nodes.map(n => n.name);
        if (!nodesH.length) { alert('Матриця порожня, додайте вузли та ребра.'); return; }

        const size   = nodes.length;
        const matrix = Array.from({length:size}, () => Array(size).fill(0));
        edges.forEach(e => {
            const fi = nodes.findIndex(n => n.name === e.from);
            const ti = nodes.findIndex(n => n.name === e.to);
            if (fi>=0 && ti>=0) {
                matrix[fi][ti]=1;
                if (e.type==='two-way') matrix[ti][fi]=1;
            }
        });
        for (let i=0; i<size; i++) matrix[i][i]=0;

        const resultEl = document.getElementById('result');
        resultEl.innerHTML = ''; resultEl.classList.remove('hidden');

        fetch(window.location.href, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name="csrfmiddlewaretoken"]').value,
            },
            body: JSON.stringify({nodes: nodesH, matrix, mode: currentMode}),
        })
        .then(r => r.json())
        .then(data => {
            if (data.dead) {
                resultEl.innerHTML = "<p class='grey'>Кластер мертвий</p>";
                return;
            }
            if (data.split_brain) {
                resultEl.innerHTML = "<p class='red'>Ситуація Split Brain</p>";
                return;
            }

            if (data.mode === 'base') {
                const rf = data.probability_rf;
                const gb = data.probability_gb;
                const cb = data.probability_cb;
                const colorClass = p => p < 30 ? 'green' : p < 60 ? 'orange' : 'red';
                resultEl.innerHTML = `
                    <p><strong>Random Forest:</strong>
                       <span class="${colorClass(rf)}">${rf}%</span></p>
                    <p><strong>Gradient Boosting:</strong>
                       <span class="${colorClass(gb)}">${gb}%</span></p>
                    <p><strong>CatBoost:</strong>
                       <span class="${colorClass(cb)}">${cb}%</span></p>`;
                return;
            }

            if (data.mode === 'ensemble') {
                const labels = {
                    e1: 'E1 &nbsp;(RF + GB + CB)',
                    e2: 'E2 &nbsp;(CB 10% · 50% · 90%)',
                    e3: 'E3 &nbsp;(CB shallow · deep · highReg)',
                };
                const colorClass = p => p < 30 ? 'green' : p < 60 ? 'orange' : 'red';
                resultEl.innerHTML = Object.entries(labels).map(([k, label]) => {
                    const p = data[k];
                    return `<p><strong>${label}:</strong>
                        <span class="${colorClass(p)}">${p}%</span></p>`;
                }).join('');
            }
        })
        .catch(err => {
            resultEl.innerHTML = `<p class='error'>Помилка: ${err.message}</p>`;
        });
    });

    function getNodeColor(name) {
        const map = {
            A:'#4CAF50',B:'#FF5733',C:'#3375FF',D:'#FF9800',E:'#9C27B0',
            F:'#00BCD4',G:'#8BC34A',H:'#FFEB3B',I:'#607D8B',J:'#795548',
            K:'#D32F2F',L:'#3F51B5',M:'#CDDC39',N:'#009688',O:'#2196F3',
            P:'#FF5722',Q:'#9E9E9E',R:'#E91E63',S:'#8E24AA',T:'#F44336',
            U:'#00BCD4',V:'#607D8B',W:'#795548',X:'#FFC107',Y:'#00C853',Z:'#FF4081',
        };
        return map[name.charAt(0).toUpperCase()] || '#FFD700';
    }

    function validateNodeName(name) {
        return /^[A-Z][1-9]$/.test(name);
    }

    drawGraph();
});