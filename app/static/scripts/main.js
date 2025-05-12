document.getElementById("generate-matrix").addEventListener("click", () => {
    const size = parseInt(document.getElementById("matrix-size-input").value);
    if (size < 4 || size > 9 || isNaN(size)) {
        alert("Введіть розмір від 4 до 9");
        return;
    }

    const container = document.getElementById("matrix-container");
    container.innerHTML = "";
    const table = document.createElement("table");

    for (let i = 0; i <= size; i++) {
        const row = table.insertRow();
        for (let j = 0; j <= size; j++) {
            const cell = row.insertCell();
            if (i === 0 && j > 0) {
                const input = createNodeInput(`A${j}`);
                cell.appendChild(input);
            } else if (j === 0 && i > 0) {

                const input = createNodeInput(`A${i}`);
                cell.appendChild(input);
            } else if (i === j) {
                cell.textContent = 0;
                cell.style.backgroundColor = "#ddd";
            } else if (i > 0 && j > 0) {
                const input = document.createElement("input");
                input.type = "number";
                input.min = 0;
                input.max = 1;
                input.value = 0;
                cell.appendChild(input);
            }
        }
    }
    container.appendChild(table);
    document.getElementById("submit-matrix").classList.remove("hidden");
});

function createNodeInput(defaultValue) {
    const input = document.createElement("input");
    input.type = "text";
    input.value = defaultValue;
    input.maxLength = 2;
    input.style.width = "2em";
    return input;
}

document.getElementById("submit-matrix").addEventListener("click", () => {
    const selectedModel = document.getElementById("model-choice").value;
    const table = document.querySelector("table");
    const rows = table.rows;

    const nodesHorizontal = [];
    const nodesVertical = [];
    const matrix = [];

    for (let j = 1; j < rows[0].cells.length; j++) {
        const input = rows[0].cells[j].querySelector("input");
        if (input) {
            nodesHorizontal.push(input.value);
        }
    }

    for (let i = 1; i < rows.length; i++) {
        const input = rows[i].cells[0].querySelector("input");
        if (input) {
            nodesVertical.push(input.value);
        }
    }

    const allNodes = [...nodesHorizontal, ...nodesVertical];
    if (!allNodes.every((name) => /^[A-Z][1-9]$/.test(name))) {
        alert("Імена нод мають відповідати формату: перша літера (A-Z), друга цифра (1-9).");
        return;
    }

    if (JSON.stringify(nodesHorizontal) !== JSON.stringify(nodesVertical)) {
        alert("Горизонтальні та вертикальні імена нод мають співпадати.");
        return;
    }

    for (let i = 1; i < rows.length; i++) {
        const row = [];
        for (let j = 1; j < rows[i].cells.length; j++) {
            const input = rows[i].cells[j].querySelector("input");
            const value = input ? parseInt(input.value, 10) : 0;

            if (value !== 0 && value !== 1) {
                alert("Матриця може містити лише 0 або 1.");
                return;
            }
            row.push(value);
        }
        matrix.push(row);
    }

    console.log("Nodes:", nodesHorizontal);
    console.log("Model:", selectedModel);
    console.log("Matrix:", matrix);

    fetch(window.location.href, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": document.querySelector('[name="csrfmiddlewaretoken"]').value
        },
        body: JSON.stringify({
            nodes: nodesHorizontal,
            matrix: matrix,
            model: selectedModel
        })
    })
        .then((response) => response.json())
        .then((data) => {

            console.log("Отримані дані від сервера:", data);

            const resultElement = document.getElementById("result");
            if (data.probability === undefined) {
                alert("Помилка: некоректна відповідь сервера.");
                return;
            }

            const probability = data.probability;

            if (probability === -1) {
                resultElement.textContent = "Кластер мертвий";
                resultElement.className = "grey";
            } else {
                resultElement.textContent = `Ймовірність: ${probability}%`;
                resultElement.className = probability < 50 ? "green" : "red";
            }

            resultElement.classList.remove("hidden");
        })
        .catch((error) => console.error("Помилка при відправці:", error));
});
