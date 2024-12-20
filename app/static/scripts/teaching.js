document.addEventListener("DOMContentLoaded", function () {
    const confirmationCheckbox = document.getElementById("confirmation-checkbox");
    const generateButton = document.getElementById("generate-button");
    const matrixSection = document.getElementById("matrix-section");
    const matrixContainer = document.getElementById("matrix-container");
    const splitBrainYes = document.getElementById("split-brain-yes");
    const splitBrainNo = document.getElementById("split-brain-no");

    confirmationCheckbox.addEventListener("change", function () {
        generateButton.disabled = !this.checked;
        generateButton.classList.toggle("gray", !this.checked);
    });

    generateButton.addEventListener("click", function () {
        const size = Math.floor(Math.random() * 4) + 2; // Random size between 2 and 5
        const nodes = generateNodes(size);
        const matrix = generateMatrix(size);
        renderMatrix(nodes, matrix);
        matrixSection.classList.remove("hidden");
        splitBrainYes.classList.remove("hidden");
        splitBrainNo.classList.remove("hidden");
    });

    splitBrainYes.addEventListener("click", function () {
        alert("Ви вказали, що є split brain.");
        resetForm();
    });

    splitBrainNo.addEventListener("click", function () {
        alert("Ви вказали, що нема split brain.");
        resetForm();
    });

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
            row.forEach((cell) => {
                tableHTML += `<td>${cell}</td>`;
            });
            tableHTML += `</tr>`;
        });
        tableHTML += `</tbody></table>`;
        matrixContainer.innerHTML = tableHTML;
    }

    function resetForm() {
        matrixSection.classList.add("hidden");
        matrixContainer.innerHTML = "";
        splitBrainYes.classList.add("hidden");
        splitBrainNo.classList.add("hidden");
    }
});
