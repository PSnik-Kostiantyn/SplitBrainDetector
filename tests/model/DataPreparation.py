import unittest

from app.model.DataPreparation import *

class TestClusterFunctions(unittest.TestCase):
    def test_generate_cluster(self):
        nodes, matrix = generate_cluster()
        self.assertTrue(2 <= len(nodes) <= 20)
        self.assertEqual(len(matrix), len(nodes))
        self.assertEqual(len(matrix[0]), len(nodes))

    def test_pad_cluster(self):
        nodes, matrix = generate_cluster()
        padded_nodes, padded_matrix = pad_cluster(nodes, matrix)
        self.assertEqual(len(padded_nodes), 20)
        self.assertEqual(padded_matrix.shape, (20, 20))

    def test_preprocess(self):
        nodes, matrix = generate_cluster()
        processed = preprocess(nodes, matrix)
        self.assertEqual(len(processed), 20 + 400)

    def test_find_islands(self):
        matrix = np.array([[0, 1], [1, 0]])
        islands = find_islands(matrix)
        self.assertEqual(len(islands), 1)

    def test_isClusterDead(self):
        nodes = ["A", "B", "C"]
        matrix = np.zeros((3, 3), dtype=int)
        self.assertTrue(isClusterDead(nodes, matrix))

    def test_isSplitBrain(self):
        nodes = ["A1", "A2", "B1", "B2", "C1", "C2"]
        matrix = np.array([[0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 1, 0],
                           [0, 1, 0, 0, 0, 0],
                           [1, 0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])
        self.assertTrue(isSplitBrain(nodes, matrix))

        matrix_split_brain = np.array([[0, 0, 0, 1, 1, 0],
                                       [0, 0, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 0, 0],
                                       [0, 1, 1, 0, 0, 0]])

        self.assertTrue(isSplitBrain(nodes, matrix_split_brain))

    def test_isSingleType(self):
        nodes = ["A", "A", "B"]
        self.assertTrue(isSingleType(nodes))
        nodes = ["A", "A", "C", "C"]
        self.assertFalse(isSingleType(nodes))


if __name__ == "__main__":
    unittest.main()
