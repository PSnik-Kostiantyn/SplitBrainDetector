from django.test import TestCase, Client
from unittest.mock import patch
import json

class ClusterViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = '/cluster/'

    @patch('app.views.isClusterDead', return_value=False)
    @patch('app.views.isSplitBrain', return_value=False)
    @patch('app.views.predict_neural_model', return_value=0.65)
    def test_valid_post_request(self, mock_predict, mock_split_brain, mock_cluster_dead):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]
            ]
        }
        response = self.client.post(self.url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"probability": 65.0})
        mock_cluster_dead.assert_called_once()
        mock_split_brain.assert_called_once()
        mock_predict.assert_called_once()

    def test_post_request_with_empty_data(self):
        response = self.client.post(self.url, data=json.dumps({}), content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Матриця або вузли не можуть бути порожніми."})

    def test_post_request_with_invalid_json(self):
        response = self.client.post(self.url, data="invalid-json", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Невірний формат JSON."})

    @patch('app.views.isClusterDead', return_value=True)
    def test_post_request_with_dead_cluster(self, mock_cluster_dead):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        }
        response = self.client.post(self.url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"probability": -1})
        mock_cluster_dead.assert_called_once()

    @patch('app.views.isSplitBrain', return_value=True)
    def test_post_request_with_split_brain(self, mock_split_brain):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ]
        }
        response = self.client.post(self.url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"probability": 100})
        mock_split_brain.assert_called_once()

    def test_get_request(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'cluster.html')
