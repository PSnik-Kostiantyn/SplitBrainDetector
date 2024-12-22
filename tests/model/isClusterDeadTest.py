import unittest

from app.model.isDeadCluster import find_islands, dfs, isClusterDead


class TestClusterFunctions(unittest.TestCase):

    def test_dfs_multiple_nodes(self):
        graph = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        visited = [False] * 3
        component = []
        dfs(0, graph, visited, component)
        self.assertEqual(component, [0, 1, 2])

    def test_find_islands_one_island(self):
        matrix = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        islands = find_islands(matrix)
        self.assertEqual(islands, [[0, 1, 2]])

    def test_find_islands_multiple_islands(self):
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        islands = find_islands(matrix)
        self.assertEqual(islands, [[0, 1], [2, 3]])

    def test_isClusterDead_empty_matrix(self):
        nodes = ['A1', 'B1', 'C1']
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.assertTrue(isClusterDead(nodes, matrix))

    def test_isClusterDead_no_dead_cluster(self):
        nodes = ['A1', 'B1', 'C1']
        matrix = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        self.assertFalse(isClusterDead(nodes, matrix))

    def test_isClusterDead_split_brain(self):
        nodes = ['A1', 'B1', 'C1', 'A2']
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        self.assertTrue(isClusterDead(nodes, matrix))

    def test_isClusterDead_partial_types_missing(self):
        nodes = ['A1', 'B1', 'C1', 'A2']
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        nodes[3] = 'B2'
        self.assertTrue(isClusterDead(nodes, matrix))

