import unittest

from app.model.DataPreparation import isSplitBrain


class TestSplitBrainDetection(unittest.TestCase):
    def test_find_islands_single_node(self):
        matrix = [[0]]
        expected = [[0]]
        result = self.find_islands(matrix)
        self.assertEqual(result, expected)

    def test_find_islands_no_connections(self):
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        expected = [[0], [1], [2]]
        result = self.find_islands(matrix)
        self.assertEqual(result, expected)

    def test_find_islands_one_connected_component(self):
        matrix = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        expected = [[0, 1, 2]]
        result = self.find_islands(matrix)
        self.assertEqual(result, expected)

    def test_find_islands_multiple_components(self):
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        expected = [[0, 1], [2, 3]]
        result = self.find_islands(matrix)
        self.assertEqual(result, expected)

    def test_isSplitBrain_no_split_brain(self):
        nodes = ['A1', 'B1', 'C1']
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self.assertFalse(isSplitBrain(nodes, matrix))

    def test_isSplitBrain_split_brain_detected(self):
        nodes = ['A1', 'B1', 'C1', 'A2', 'B2']
        matrix = [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ]
        self.assertFalse(isSplitBrain(nodes, matrix))

    def test_isSplitBrain_partial_types(self):
        nodes = ['A1', 'B1', 'A2', 'B2']
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        self.assertTrue(isSplitBrain(nodes, matrix))

    def test_isSplitBrain_empty_matrix(self):
        nodes = ['A1', 'B1', 'C1']
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.assertFalse(isSplitBrain(nodes, matrix))

    def test_isSplitBrain_multiple_functional_islands(self):
        nodes = ['A1', 'B1', 'C1', 'A2', 'B2', 'C2']
        matrix = [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0]
        ]
        self.assertFalse(isSplitBrain(nodes, matrix))

    def find_islands(self, matrix):
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

if __name__ == '__main__':
    unittest.main()