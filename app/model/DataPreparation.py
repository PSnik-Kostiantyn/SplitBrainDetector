import random
import numpy as np

def generate_cluster():
    num_nodes = random.randint(2, 9)
    nodes = [random.choice(["A", "B", "C"]) for _ in range(num_nodes)]
    matrix = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[0.7, 0.3])
    np.fill_diagonal(matrix, 0)
    matrix = np.minimum(matrix, matrix.T)
    return nodes, matrix

def pad_cluster(nodes, matrix, max_nodes=20):
    padded_nodes = nodes + [-1] * (max_nodes - len(nodes))
    padded_matrix = np.full((max_nodes, max_nodes), -1)
    padded_matrix[:len(matrix), :len(matrix)] = matrix
    return padded_nodes, padded_matrix

def preprocess(nodes, matrix):
    max_nodes = 9
    padded_nodes, padded_matrix = pad_cluster(nodes, matrix, max_nodes)
    x_nodes = [ord(n[0]) - ord("A") + 2 if isinstance(n, str) else -1 for n in padded_nodes]
    x_matrix = padded_matrix.flatten()
    # print(nodes)
    # print(matrix)
    # print(np.array(x_nodes + x_matrix.tolist()))
    return np.array(x_nodes + x_matrix.tolist())

def dfs(node, graph, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor, connected in enumerate(graph[node]):
        if connected and graph[neighbor][node] and not visited[neighbor]:
            dfs(neighbor, graph, visited, component)

def find_islands(matrix):
    n = len(matrix)
    visited = [False] * n
    islands = []
    for node in range(n):
        if not visited[node]:
            component = []
            dfs(node, matrix, visited, component)
            islands.append(component)
    return islands

def isClusterDead(nodes, matrix):
    if all(all(cell == 0 for cell in row) for row in matrix):
        return True
    required_types = set(node[0].lower() for node in nodes)
    islands = find_islands(matrix)
    for island in islands:
        types_in_island = set(nodes[node_index][0].lower() for node_index in island)
        if required_types.issubset(types_in_island):
            return False
    return True

def isSplitBrain(nodes, matrix):
    if isSingleType(nodes):
        return False
    node_types = set(node[0] for node in nodes)
    islands = find_islands(matrix)
    functional_islands = 0
    for island in islands:
        types_present = set(nodes[i][0] for i in island)
        if node_types.issubset(types_present):
            functional_islands += 1
            if functional_islands >= 2:
                return True
    return False

def isSingleType(nodes):
    type_counts = {}
    for node in nodes:
        type_counts[node[0]] = type_counts.get(node[0], 0) + 1
    return any(count < 2 for count in type_counts.values())

# TODO single type implement, split brain correct, sensitivity boost, autotests for accuracy
