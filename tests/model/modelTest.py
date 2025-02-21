import unittest
from unittest.mock import patch, MagicMock
import torch

from app.model.isDeadCluster import find_islands, dfs
from app.model.neuralModel import isClusterDead, generate_cluster, pad_cluster, \
    SplitBrainModel, save_model, predict_neural_model, teach_neural_model


class TestNeuralNetwork(unittest.TestCase):
    def test_dfs(self):
        graph = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        visited = [False, False, False]
        component = []
        dfs(0, graph, visited, component)
        self.assertEqual(component, [0, 1, 2])

    def test_find_islands(self):
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        expected = [[0, 1], [2, 3]]
        self.assertEqual(find_islands(matrix), expected)

    def test_isClusterDead_all_dead(self):
        nodes = ['A', 'B', 'C']
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.assertTrue(isClusterDead(nodes, matrix))

    def test_isClusterDead_alive_cluster(self):
        nodes = ['A', 'B', 'C']
        matrix = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        self.assertFalse(isClusterDead(nodes, matrix))


    def test_generate_cluster(self):
        nodes, matrix = generate_cluster()
        self.assertTrue(2 <= len(nodes) <= 9)
        self.assertEqual(len(matrix), len(nodes))
        self.assertEqual(len(matrix[0]), len(nodes))

    def test_pad_cluster(self):
        nodes = ['A', 'B']
        matrix = [
            [0, 1],
            [1, 0]
        ]
        padded_nodes, padded_matrix = pad_cluster(nodes, matrix)
        self.assertEqual(len(padded_nodes), 9)
        self.assertEqual(padded_matrix.shape, (9, 9))

    @patch('torch.save')
    def test_save_model(self, mock_save):
        model = SplitBrainModel()
        save_model(model, 'test_model.pth')
        mock_save.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('torch.load')
    def test_predict_neural_model_load_model(self, mock_load, mock_exists):
        nodes = ['A', 'B', 'C']
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]

        fake_state_dict = {
            'fc1.weight': torch.randn(128, 90),
            'fc1.bias': torch.randn(128),
            'fc2.weight': torch.randn(64, 128),
            'fc2.bias': torch.randn(64),
            'fc3.weight': torch.randn(1, 64),
            'fc3.bias': torch.randn(1),
        }

        mock_load.return_value = fake_state_dict

        result = predict_neural_model(nodes, matrix)

        mock_load.assert_called_once()
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    @patch('os.path.exists', return_value=False)
    @patch('torch.save')
    def test_teach_neural_model_train_and_save(self, mock_save, mock_exists):
        nodes = ['A', 'B', 'C']
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        loss = teach_neural_model(nodes, matrix)
        self.assertGreater(loss, 0)
        mock_save.assert_called_once()

    # @patch('torch.optim.Adam')
    # def test_train_model_optimizer(self, mock_adam):
    #     mock_optimizer = MagicMock()
    #     mock_adam.return_value = mock_optimizer
    #     model = train_model()
    #     self.assertIsInstance(model, SplitBrainModel)
    #     mock_adam.assert_called_once()

    def test_split_brain_model_structure(self):
        model = SplitBrainModel()
        sample_input = torch.randn(9 * 9 + 9)
        output = model(sample_input)
        self.assertEqual(output.shape, torch.Size([1]))
        self.assertTrue((0 <= output.item() <= 1))

if __name__ == '__main__':
    unittest.main()