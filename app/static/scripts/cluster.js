document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('graph-canvas');
    const ctx = canvas.getContext('2d');
    let nodes = [];
    let edges = [];
    let currentNodeName = '';
    let isErasing = false;
    let currentEdgeType = 'one-way';
    let selectedNode = null;
    let isDragging = false;

    // Створення нової ноди
    document.getElementById('create-node').addEventListener('click', function() {
        const nodeName = document.getElementById('node-name').value;
        if (nodeName) {
            nodes.push({ name: nodeName, x: Math.random() * canvas.width, y: Math.random() * canvas.height });
            drawGraph();
            document.getElementById('node-name').value = '';
        }
    });

    // Вибір типу лінії (в один бік чи в два)
    document.getElementById('line-one-way').addEventListener('click', function() {
        currentEdgeType = 'one-way';
    });

    document.getElementById('line-two-way').addEventListener('click', function() {
        currentEdgeType = 'two-way';
    });

    // Вибір інструменту для стирання
    document.getElementById('erase').addEventListener('click', function() {
        isErasing = !isErasing;
        this.style.backgroundColor = isErasing ? '#ff4444' : '';
    });

    // Слухач на натискання на канвас для малювання ліній
    canvas.addEventListener('click', function(event) {
        if (isErasing) {
            eraseNode(event);
        } else {
            addEdge(event);
        }
    });

    // Малювання графа на канвасі
    function drawGraph() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 20, 0, Math.PI * 2);
            ctx.fillStyle = '#4CAF50';
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#fff';
            ctx.fillText(node.name, node.x - 10, node.y + 5);
        });

        edges.forEach(edge => {
            const fromNode = nodes.find(n => n.name === edge.from);
            const toNode = nodes.find(n => n.name === edge.to);
            if (fromNode && toNode) {
                ctx.beginPath();
                ctx.moveTo(fromNode.x, fromNode.y);
                ctx.lineTo(toNode.x, toNode.y);
                ctx.strokeStyle = edge.type === 'two-way' ? '#00f' : '#f00';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        });
    }

    // Додавання лінії між двома нодами
    function addEdge(event) {
        const clickedNode = nodes.find(node => Math.hypot(node.x - event.offsetX, node.y - event.offsetY) < 20);
        if (clickedNode) {
            if (!currentNodeName) {
                currentNodeName = clickedNode.name;
            } else {
                const newEdge = { from: currentNodeName, to: clickedNode.name, type: currentEdgeType };
                edges.push(newEdge);
                currentNodeName = '';
                drawGraph();
                updateMatrix();
            }
        }
    }

    // Стирання ноди
    function eraseNode(event) {
        const clickedNode = nodes.find(node => Math.hypot(node.x - event.offsetX, node.y - event.offsetY) < 20);
        if (clickedNode) {
            nodes = nodes.filter(node => node !== clickedNode);
            edges = edges.filter(edge => edge.from !== clickedNode.name && edge.to !== clickedNode.name);
            drawGraph();
            updateMatrix();
        } else {
            eraseEdge(event);
        }
    }

    // Видалення зв'язку між нодами
    function eraseEdge(event) {
        const clickedEdge = edges.find(edge => {
            const fromNode = nodes.find(n => n.name === edge.from);
            const toNode = nodes.find(n => n.name === edge.to);
            return fromNode && toNode && isPointOnLine(event.offsetX, event.offsetY, fromNode.x, fromNode.y, toNode.x, toNode.y);
        });
        if (clickedEdge) {
            edges = edges.filter(edge => edge !== clickedEdge);
            drawGraph();
            updateMatrix();
        }
    }

    // Перевірка чи точка знаходиться на лінії
    function isPointOnLine(px, py, x1, y1, x2, y2) {
        const distance = Math.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) /
                         Math.sqrt(Math.pow(y2 - y1, 2) + Math.pow(x2 - x1, 2));
        return distance < 10; // Точність вибору лінії
    }

    // Оновлення матриці
    function updateMatrix() {
        const matrix = document.getElementById('matrix');
        matrix.innerHTML = '';
        const size = nodes.length;
        const matrixData = Array.from({ length: size }, () => Array(size).fill(0));

        edges.forEach(edge => {
            const fromIndex = nodes.findIndex(node => node.name === edge.from);
            const toIndex = nodes.findIndex(node => node.name === edge.to);
            if (fromIndex >= 0 && toIndex >= 0) {
                matrixData[fromIndex][toIndex] = 1;
                if (edge.type === 'two-way') {
                    matrixData[toIndex][fromIndex] = 1;
                }
            }
        });

        for (let i = 0; i < size; i++) {
            const row = document.createElement('tr');
            for (let j = 0; j < size; j++) {
                const cell = document.createElement('td');
                cell.textContent = matrixData[i][j];
                cell.setAttribute('data-node', nodes[i].name);
                cell.addEventListener('click', function() {
                    console.log('Nodes:', nodes.map(node => node.name));
                    console.log('Matrix:', matrixData);
                });
                row.appendChild(cell);
            }
            matrix.appendChild(row);
        }
    }

    // Переміщення ноди
    canvas.addEventListener('mousedown', function(event) {
        const clickedNode = nodes.find(node => Math.hypot(node.x - event.offsetX, node.y - event.offsetY) < 20);
        if (clickedNode) {
            selectedNode = clickedNode;
            isDragging = true;
        }
    });

    canvas.addEventListener('mousemove', function(event) {
        if (isDragging && selectedNode) {
            selectedNode.x = event.offsetX;
            selectedNode.y = event.offsetY;
            drawGraph();
            updateMatrix();
        }
    });

    canvas.addEventListener('mouseup', function() {
        isDragging = false;
        selectedNode = null;
    });

    // Відправлення матриці
    document.getElementById('submit-matrix').addEventListener('click', function() {
        // Код для відправлення даних (наприклад, через AJAX)
        alert('Матриця відправлена!');
    });
});
