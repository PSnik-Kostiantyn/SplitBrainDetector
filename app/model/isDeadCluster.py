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
        types_in_island = set()
        for node_index in island:
            node_type = nodes[node_index][0].lower()
            types_in_island.add(node_type)

        if required_types.issubset(types_in_island):
            return False

    return True
