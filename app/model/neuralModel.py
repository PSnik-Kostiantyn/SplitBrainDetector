import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import torch
import numpy as np

def dfs(node, graph, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor, connected in enumerate(graph[node]):
        if connected and not visited[neighbor]:
            dfs(neighbor, graph, visited, component)

def find_islands(matrix):
    n = len(matrix)
    visited = [False] * n
    islands = []

    for node in range(n):
        if not visited[node] and not all(x == -1 for x in matrix[node]):
            component = []
            dfs(node, matrix, visited, component)
            islands.append(component)
    return islands

def isClusterDead(nodes, matrix):
    if all(all(cell == 0 or cell == -1 for cell in row) for row in matrix):
        return True

    required_types = set(node[0].lower() for node in nodes if node != -1)

    islands = find_islands(matrix)

    for island in islands:
        types_in_island = set()
        for node_index in island:
            if nodes[node_index] != -1:
                node_type = nodes[node_index][0].lower()
                types_in_island.add(node_type)

        if required_types.issubset(types_in_island):
            return False

    return True

def isSplitBrain(nodes, matrix):
    node_types = set(node[0] for node in nodes if node != -1)

    islands = find_islands(matrix)

    functional_islands = 0
    for island in islands:
        types_present = set(nodes[i][0] for i in island if nodes[i] != -1)

        if node_types.issubset(types_present):
            functional_islands += 1
            if functional_islands >= 2:
                return True

    return False

def generate_cluster():
    num_nodes = random.randint(2, 9)
    nodes = [random.choice(["A", "B", "C"]) for _ in range(num_nodes)]
    matrix = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[0.7, 0.3])
    np.fill_diagonal(matrix, 0)
    return nodes, matrix

def pad_cluster(nodes, matrix, max_nodes=9):
    padded_nodes = nodes + [-1] * (max_nodes - len(nodes))
    padded_matrix = np.full((max_nodes, max_nodes), -1)
    padded_matrix[:len(matrix), :len(matrix)] = matrix
    return padded_nodes, padded_matrix

class SplitBrainModel(nn.Module):
    def __init__(self):
        super(SplitBrainModel, self).__init__()
        self.fc1 = nn.Linear(9 * 9 + 9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train_model():
    model = SplitBrainModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for _ in range(100000):
            nodes, matrix = generate_cluster()
            while isClusterDead(nodes, matrix):
                nodes, matrix = generate_cluster()

            padded_nodes, padded_matrix = pad_cluster(nodes, matrix)
            x_nodes = [1 if n == "A" else 2 if n == "B" else 3 for n in padded_nodes]
            x_matrix = padded_matrix.flatten()
            x_input = torch.tensor(x_nodes + x_matrix.tolist(), dtype=torch.float32)
            x_input = x_input / 3.0

            label = isSplitBrain(nodes, matrix)
            y_target = torch.tensor([label], dtype=torch.float32)

            optimizer.zero_grad()
            output = model(x_input)
            loss = criterion(output, y_target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model

def save_model(model, filename="split_brain_model_new_1.pth"):
    torch.save(model.state_dict(), filename)

def predict_neural_model(nodes, matrix):
    model_path = 'split_brain_model_good_2.pth'

    def preprocess(nodes, matrix):
        max_nodes = 9
        padded_nodes, padded_matrix = pad_cluster(nodes, matrix, max_nodes)
        x_nodes = [1 if n == "A" else 2 if n == "B" else 3 for n in padded_nodes]
        x_matrix = padded_matrix.flatten()
        x_input = torch.tensor(x_nodes + x_matrix.tolist(), dtype=torch.float32)
        return x_input / 3.0

    def load_model():
        model = SplitBrainModel()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            print("Модель не знайдена. Починається тренування...")
            model = train_model()
            save_model(model, model_path)
        return model

    x_input = preprocess(nodes, matrix)

    model = load_model()

    with torch.no_grad():
        prediction = model(x_input).item()
        return prediction


def teach_neural_model(nodes, matrix):
    model_path = "split_brain_model_good_2.pth"

    def preprocess(nodes, matrix):
        max_nodes = 9
        padded_nodes, padded_matrix = pad_cluster(nodes, matrix, max_nodes)
        x_nodes = [1 if n == "A" else 2 if n == "B" else 3 for n in padded_nodes]
        x_matrix = padded_matrix.flatten()
        x_input = torch.tensor(x_nodes + x_matrix.tolist(), dtype=torch.float32)
        return x_input / 3.0

    def load_model():
        model = SplitBrainModel()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print("Модель успішно завантажена.")
        else:
            print("Модель не знайдена. Створення нової моделі...")
        return model

    model = load_model()
    model.train()

    x_input = preprocess(nodes, matrix)
    x_input = x_input.unsqueeze(0)

    label = isSplitBrain(nodes, matrix)
    y_target = torch.tensor([[label]], dtype=torch.float32)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    output = model(x_input)
    loss = criterion(output, y_target)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), model_path)
    print(f"Модель успішно оновлена та збережена. Втрата: {loss.item():.6f}")
    return loss.item()

