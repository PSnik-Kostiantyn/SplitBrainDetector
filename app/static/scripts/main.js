document.getElementById("generate-matrix").addEventListener("click", () => {
    const size = parseInt(document.getElementById("matrix-size-input").value);
    if (size < 2 || size > 10 || isNaN(size)) {
        alert("Введіть розмір від 2 до 10");
        return;
    }

    // Створюємо таблицю
    const container = document.getElementById("matrix-container");
    container.innerHTML = "";
    const table = document.createElement("table");

    for (let i = 0; i <= size; i++) {
        const row = table.insertRow();
        for (let j = 0; j <= size; j++) {
            const cell = row.insertCell();
            if (i === 0 && j > 0) cell.textContent = `A${j}`;
            else if (j === 0 && i > 0) cell.textContent = `A${i}`;
            else if (i === j) {
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

document.getElementById("submit-matrix").addEventListener("click", () => {
    const inputs = document.querySelectorAll("table input");
    if (inputs.length === 0) {
        alert("Спочатку згенеруйте матрицю.");
        return;
    }

    const table = document.querySelector("table");
    const rows = table.rows;

    const nodes = []; // Вузли A1, A2, ...
    const matrix = []; // Матриця доступності

    // Зчитуємо вузли з першого рядка таблиці
    for (let j = 1; j < rows[0].cells.length; j++) {
        nodes.push(rows[0].cells[j].textContent);
    }

    // Зчитуємо матрицю доступності з таблиці
    for (let i = 1; i < rows.length; i++) {
        const row = [];
        for (let j = 1; j < rows[i].cells.length; j++) {
            const input = rows[i].cells[j].querySelector("input");
            const value = input ? parseInt(input.value) : 0;

            if (value !== 0 && value !== 1) {
                alert("Матриця може містити лише 0 або 1.");
                return;
            }
            row.push(value);
        }
        matrix.push(row);
    }

    console.log("Nodes:", nodes);
    console.log("Matrix:", matrix);

    // Відправляємо дані на сервер
    fetch(window.location.href, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": document.querySelector('[name="csrfmiddlewaretoken"]').value
        },
        body: JSON.stringify({ nodes: nodes, matrix: matrix })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Отримані дані від сервера:", data);

        const resultElement = document.getElementById("result");
        if (!data.data || data.data.length === 0) {
            alert("Помилка: некоректна відповідь сервера.");
            return;
        }

        const probability = data.data[0].probability;

        if (probability === -1) {
            resultElement.textContent = "Кластер мертвий";
            resultElement.className = "gray";
            document.getElementById("submit-matrix").classList.add("gray");
        } else {
            resultElement.textContent = `Ймовірність: ${probability}%`;
            resultElement.className = probability > 50 ? "green" : "red";
        }
        resultElement.classList.remove("hidden");
    })
    .catch(error => console.error("Помилка при відправці:", error));
});
