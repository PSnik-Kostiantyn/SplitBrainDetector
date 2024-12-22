document.addEventListener("DOMContentLoaded", function () {
    const confirmationCheckbox = document.getElementById("confirmation-checkbox");
    const generateButton = document.getElementById("generate-button");
    const matrixSection = document.getElementById("matrix-section");
    const matrixContainer = document.getElementById("matrix-container");
    const splitBrainYes = document.getElementById("split-brain-yes");
    const splitBrainNo = document.getElementById("split-brain-no");

    let currentNodes = [];
    let currentMatrix = [];

    confirmationCheckbox.addEventListener("change", function () {
        generateButton.disabled = !this.checked;
        generateButton.classList.toggle("gray", !this.checked);
    });

    generateButton.addEventListener("click", function () {
        const size = Math.floor(Math.random() * 4) + 2;
        currentNodes = generateNodes(size);
        currentMatrix = generateMatrix(size);
        renderMatrix(currentNodes, currentMatrix);
        matrixSection.classList.remove("hidden");
        splitBrainYes.classList.remove("hidden");
        splitBrainNo.classList.remove("hidden");
    });

    splitBrainYes.addEventListener("click", function () {
        sendToBackend(true);
    });

    splitBrainNo.addEventListener("click", function () {
        sendToBackend(false);
    });

    function sendToBackend(isSplitBrain) {
        const csrfToken = document.querySelector('[name="csrfmiddlewaretoken"]').value;

        const payload = {
            nodes: currentNodes,
            matrix: currentMatrix,
            split_brain: isSplitBrain ? 1 : 0,
        };

        fetch(window.location.href, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken,
            },
            body: JSON.stringify(payload),
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP помилка: ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                console.log("Отримані дані від сервера:", data);

                const resultElement = document.getElementById("result");
                if (!resultElement) {
                    console.error("Елемент з id='result' не знайдено.");
                    return;
                }

                const probability = data.probability;
                if (probability === undefined) {
                    alert("Помилка: некоректна відповідь сервера.");
                    return;
                }

                if (probability === -1) {
                    resultElement.textContent = "Кластер мертвий";
                    resultElement.className = "grey";
                } else if (probability === -2) {
                    resultElement.textContent = "Неправильна класифікація";
                    resultElement.className = "red";
                } else if (probability === -3) {
                    resultElement.textContent = "Проблема моделі.";
                    resultElement.className = "orange";
                } else {
                    resultElement.textContent = `Успіх! ${probability}`;
                    resultElement.className = probability < 50 ? "green" : "red";
                }

                resultElement.classList.remove("hidden");
            })
            .catch((error) => {
                console.error("Помилка при відправці:", error);
                const resultElement = document.getElementById("result");
                if (resultElement) {
                    resultElement.textContent = "Сталася помилка під час обробки запиту.";
                    resultElement.className = "error";
                    resultElement.classList.remove("hidden");
                }
            });
    }

    function generateNodes(size) {
        const types = ["A", "B", "C"];
        const nodes = [];
        for (let i = 0; i < size; i++) {
            const randomType = types[Math.floor(Math.random() * types.length)];
            nodes.push(`${randomType}${i + 1}`);
        }
        return nodes;
    }

    function generateMatrix(size) {
        const matrix = Array(size)
            .fill(null)
            .map(() => Array(size).fill(0));
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (i !== j) {
                    matrix[i][j] = Math.random() > 0.5 ? 1 : 0;
                }
            }
        }
        return matrix;
    }

    function renderMatrix(nodes, matrix) {
        let tableHTML = `<table><thead><tr><th></th>`;
        nodes.forEach((node) => {
            tableHTML += `<th>${node}</th>`;
        });
        tableHTML += `</tr></thead><tbody>`;
        matrix.forEach((row, i) => {
            tableHTML += `<tr><th>${nodes[i]}</th>`;
            row.forEach((cell, j) => {
                if (i === j) {
                    tableHTML += `<td class="diagonal">0</td>`;
                } else {
                    tableHTML += `
                    <td>
                        <input 
                            type="number" 
                            class="matrix-input" 
                            value="${cell}" 
                            data-row="${i}" 
                            data-col="${j}" 
                            min="0" 
                            max="1"
                        />
                    </td>`;
                }
            });
            tableHTML += `</tr>`;
        });
        tableHTML += `</tbody></table>`;
        matrixContainer.innerHTML = tableHTML;

        const inputs = document.querySelectorAll(".matrix-input");
        inputs.forEach((input) => {
            input.addEventListener("input", function () {
                const row = parseInt(this.getAttribute("data-row"));
                const col = parseInt(this.getAttribute("data-col"));
                const value = parseInt(this.value);

                if (value === 0 || value === 1) {
                    currentMatrix[row][col] = value; // Оновлення матриці
                    console.log(`Змінено значення: row=${row}, col=${col}, value=${value}`);
                } else {
                    alert("Значення має бути 0 або 1!");
                    this.value = currentMatrix[row][col];
                }
            });
        });
    }


    function resetForm() {
        matrixSection.classList.add("hidden");
        matrixContainer.innerHTML = "";
        splitBrainYes.classList.add("hidden");
        splitBrainNo.classList.add("hidden");
    }
});
