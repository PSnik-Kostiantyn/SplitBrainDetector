def isSplitBrain(nodes, matrix):

    node_types = set(node[0] for node in nodes)
    def find_islands(matrix):
        size = len(matrix)
        visited = [False] * size
        islands = []

        def dfs(node, island):
            visited[node] = True
            island.append(node)
            for neighbor, connected in enumerate(matrix[node]):
                if connected and not visited[neighbor]:
                    dfs(neighbor, island)

        for i in range(size):
            if not visited[i]:
                island = []
                dfs(i, island)
                islands.append(island)
        return islands

    islands = find_islands(matrix)

    functional_islands = 0
    for island in islands:
        types_present = set(nodes[i][0] for i in island)

        if node_types.issubset(types_present):
            functional_islands += 1
            if functional_islands >= 2:
                return True

    return False
