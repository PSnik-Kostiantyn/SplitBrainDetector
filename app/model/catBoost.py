import random
import os
import numpy as np
import pickle
from catboost import CatBoostClassifier


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


def isClusterDead_2(nodes, matrix):
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


def isSplitBrain_2(nodes, matrix):
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


def preprocess(nodes, matrix):
    max_nodes = 9
    padded_nodes, padded_matrix = pad_cluster(nodes, matrix, max_nodes)
    x_nodes = [1 if n == "A" else 2 if n == "B" else 3 for n in padded_nodes]
    x_matrix = padded_matrix.flatten()
    return np.array(x_nodes + x_matrix.tolist()) / 3.0


def train_model():
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=0)
    X_train, y_train = [], []
    for _ in range(100000):
        nodes, matrix = generate_cluster()
        while isClusterDead_2(nodes, matrix):
            nodes, matrix = generate_cluster()
        X_train.append(preprocess(nodes, matrix))
        y_train.append(isSplitBrain_2(nodes, matrix))
    model.fit(X_train, y_train)
    with open("split_brain_model_cb.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def load_model():
    model_path = "split_brain_model_cb.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        return train_model()


def predict_cb(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]


def teach_cb(nodes, matrix):
    model = load_model()
    x_input = preprocess(nodes, matrix).reshape(1, -1)
    label = isSplitBrain_2(nodes, matrix)
    model.fit(x_input, [label])
    with open("split_brain_model_cb.pkl", "wb") as f:
        pickle.dump(model, f)
    return label
