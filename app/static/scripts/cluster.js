document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('graph-canvas');
    const ctx = canvas.getContext('2d');
    let nodes = [];
    let edges = [];
    let currentNodeName = '';
    let isErasing = false;
    let currentEdgeType = 'one-way';
    let selectedNode = null;
    let isDragging = false;

    canvas.width = 600;
    canvas.height = 400;

    ctx.font = '14px Inter, sans-serif';

    document.getElementById('create-node').addEventListener('click', function () {
        const nodeName = document.getElementById('node-name').value;
        if (validateNodeName(nodeName)) {
            const existingNode = nodes.find(node => node.name === nodeName);
            if (existingNode) {
                alert('Ця назва ноди вже існує. Виберіть іншу назву.');
            } else {
                nodes.push({name: nodeName, x: Math.random() * canvas.width, y: Math.random() * canvas.height});
                drawGraph();
                document.getElementById('node-name').value = '';
            }
        } else {
            alert('Невірний формат назви ноди. Назва має починатися з букви A-Z, за якою йде цифра 1-9.');
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
        if (isErasing) {
            eraseNode(event);
        } else {
            addEdge(event);
        }
    });

    function drawGraph() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

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

                if (edge.type === 'one-way') {
                    drawArrowhead(fromNode.x, fromNode.y, toNode.x, toNode.y, ctx.strokeStyle);
                } else if (edge.type === 'two-way') {
                    drawArrowhead(fromNode.x, fromNode.y, toNode.x, toNode.y, ctx.strokeStyle);
                    drawArrowhead(toNode.x, toNode.y, fromNode.x, fromNode.y, ctx.strokeStyle);
                }
            }
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
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const arrowSize = 10;
        const offset = 20;
        const arrowX = x2 - offset * Math.cos(angle);
        const arrowY = y2 - offset * Math.sin(angle);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX - arrowSize * Math.cos(angle - Math.PI / 6), arrowY - arrowSize * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(arrowX - arrowSize * Math.cos(angle + Math.PI / 6), arrowY - arrowSize * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }

    function addEdge(event) {
        const clickedNode = nodes.find(node => Math.hypot(node.x - event.offsetX, node.y - event.offsetY) < 20);
        if (clickedNode) {
            if (!currentNodeName) {
                currentNodeName = clickedNode.name;
            } else {
                if (currentNodeName === clickedNode.name) {
                    currentNodeName = '';
                    return;
                }
                edges = edges.filter(edge =>
                    !(edge.from === currentNodeName && edge.to === clickedNode.name) &&
                    !(edge.from === clickedNode.name && edge.to === currentNodeName)
                );
                const newEdge = {from: currentNodeName, to: clickedNode.name, type: currentEdgeType};
                edges.push(newEdge);
                currentNodeName = '';
                drawGraph();
                updateMatrix();
            }
        }
    }

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

    function isPointOnLine(px, py, x1, y1, x2, y2) {
        const distance = Math.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) /
            Math.sqrt(Math.pow(y2 - y1, 2) + Math.pow(x2 - x1, 2));
        return distance < 10;
    }

    function updateMatrix() {
        const matrix = document.getElementById('matrix');
        matrix.innerHTML = '';

        const size = nodes.length;
        const matrixData = Array.from({length: size}, () => Array(size).fill(0));

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

        const headerRow = document.createElement('tr');
        const emptyHeaderCell = document.createElement('th');
        emptyHeaderCell.textContent = '';
        headerRow.appendChild(emptyHeaderCell);

        nodes.forEach(node => {
            const headerCell = document.createElement('th');
            headerCell.textContent = node.name;
            headerRow.appendChild(headerCell);
        });

        matrix.appendChild(headerRow);

        for (let i = 0; i < size; i++) {
            const row = document.createElement('tr');
            const rowHeaderCell = document.createElement('th');
            rowHeaderCell.textContent = nodes[i].name;
            row.appendChild(rowHeaderCell);

            for (let j = 0; j < size; j++) {
                const cell = document.createElement('td');
                cell.textContent = (i === j) ? '0' : matrixData[i][j];
                row.appendChild(cell);
            }

            matrix.appendChild(row);
        }
    }

    canvas.addEventListener('mousedown', function (event) {
        const clickedNode = nodes.find(node => Math.hypot(node.x - event.offsetX, node.y - event.offsetY) < 20);
        if (clickedNode) {
            selectedNode = clickedNode;
            isDragging = true;
        }
    });

    canvas.addEventListener('mousemove', function (event) {
        if (isDragging && selectedNode) {
            selectedNode.x = event.offsetX;
            selectedNode.y = event.offsetY;
            drawGraph();
            updateMatrix();
        }
    });

    canvas.addEventListener('mouseup', function () {
        isDragging = false;
        selectedNode = null;
    });

    document.getElementById('submit-matrix').addEventListener('click', function () {
    const nodesHorizontal = nodes.map(node => node.name);

    if (nodesHorizontal.length === 0) {
        alert("Матриця порожня, додайте вузли та ребра.");
        return;
    }

    const size = nodes.length;
    const matrix = Array.from({length: size}, () => Array(size).fill(0));

    edges.forEach(edge => {
        const fromIndex = nodes.findIndex(node => node.name === edge.from);
        const toIndex = nodes.findIndex(node => node.name === edge.to);
        if (fromIndex >= 0 && toIndex >= 0) {
            matrix[fromIndex][toIndex] = 1;
            if (edge.type === 'two-way') {
                matrix[toIndex][fromIndex] = 1;
            }
        }
    });

    for (let i = 0; i < size; i++) {
        matrix[i][i] = 0;
    }

    console.log("Nodes:", nodesHorizontal);
    console.log("Matrix:", matrix);

    fetch(window.location.href, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": document.querySelector('[name="csrfmiddlewaretoken"]').value
        },
        body: JSON.stringify({nodes: nodesHorizontal, matrix: matrix})
    })
        .then((response) => response.json())
        .then((data) => {
            console.log("Отримані дані від сервера:", data);

            const resultElement = document.getElementById("result");
            resultElement.innerHTML = "";
            resultElement.classList.remove("hidden");
            if (data.probability_neural === -1) {
                resultElement.innerHTML = "<p class='grey'>Кластер мертвий</p>";
            } else if (data.probability_neural === 100) {
                resultElement.innerHTML = "<p class='red'>Ситуація Split brain</p>";
            } else {
                resultElement.innerHTML = `
                    <p><strong>Ймовірність Neural Network:</strong> ${data.probability_neural}%</p>
                    <p><strong>Ймовірність Random Forest:</strong> ${data.probability_rf}%</p>
                    <p><strong>Ймовірність Gradient Boosting:</strong> ${data.probability_gb}%</p>
                    <p><strong>Ймовірність CatBoost:</strong> ${data.probability_cb}%</p>
                `;
            }
        })
        .catch((error) => console.error("Помилка при відправці:", error));
});

    function getNodeColor(name) {
        const letter = name.charAt(0).toUpperCase();

        const colorMap = {
            'A': '#4CAF50',
            'B': '#FF5733',
            'C': '#3375FF',
            'D': '#FF9800',
            'E': '#9C27B0',
            'F': '#00BCD4',
            'G': '#8BC34A',
            'H': '#FFEB3B',
            'I': '#607D8B',
            'J': '#795548',
            'K': '#D32F2F',
            'L': '#3F51B5',
            'M': '#CDDC39',
            'N': '#009688',
            'O': '#2196F3',
            'P': '#FF5722',
            'Q': '#9E9E9E',
            'R': '#E91E63',
            'S': '#8E24AA',
            'T': '#F44336',
            'U': '#00BCD4',
            'V': '#607D8B',
            'W': '#795548',
            'X': '#FFC107',
            'Y': '#00C853',
            'Z': '#FF4081',
        };

        return colorMap[letter] || '#FFD700';
    }

    function validateNodeName(name) {
        return /^[A-Z][1-9]$/.test(name);
    }

    drawGraph();
});