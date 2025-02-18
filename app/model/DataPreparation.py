import random

import numpy as np

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
    x_nodes = [ord(n[0]) - ord("A") + 2 if isinstance(n, str) and len(n) == 1 else -1 for n in padded_nodes]

    x_matrix = padded_matrix.flatten()
    return np.array(x_nodes + x_matrix.tolist()) / 3.0