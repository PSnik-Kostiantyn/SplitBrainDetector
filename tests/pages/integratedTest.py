from django.test import TestCase, Client
import json


class IntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.teach_url = '/teach/'
        self.cluster_url = '/cluster/'

    def test_teach_and_cluster_interaction(self):
        nodes = ["A", "B", "C"]
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]

        teach_data = {
            "nodes": nodes,
            "matrix": matrix,
            "split_brain": 1
        }
        teach_response = self.client.post(
            self.teach_url, data=json.dumps(teach_data), content_type="application/json"
        )
        self.assertEqual(teach_response.status_code, 200)
        self.assertIn("probability", teach_response.json())
        self.assertIsInstance(teach_response.json()["probability"], (int, float))

        cluster_data = {
            "nodes": nodes,
            "matrix": matrix
        }
        cluster_response = self.client.post(
            self.cluster_url, data=json.dumps(cluster_data), content_type="application/json"
        )
        self.assertEqual(cluster_response.status_code, 200)
        self.assertIn("probability", cluster_response.json())
        self.assertIsInstance(cluster_response.json()["probability"], (int, float))

    def test_full_workflow(self):
        nodes = ["A", "B", "C"]
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]

        split_brain_data = {
            "nodes": nodes,
            "matrix": [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0]
            ]
        }
        split_brain_response = self.client.post(
            self.cluster_url, data=json.dumps(split_brain_data), content_type="application/json"
        )
        self.assertEqual(split_brain_response.status_code, 200)
        self.assertEqual(split_brain_response.json()["probability"], -1)

        teach_data = {
            "nodes": nodes,
            "matrix": matrix,
            "split_brain": 0
        }
        teach_response = self.client.post(
            self.teach_url, data=json.dumps(teach_data), content_type="application/json"
        )
        self.assertEqual(teach_response.status_code, 200)
        self.assertIn("probability", teach_response.json())
        self.assertGreaterEqual(teach_response.json()["probability"], 0)

        predict_data = {
            "nodes": nodes,
            "matrix": matrix
        }
        predict_response = self.client.post(
            self.cluster_url, data=json.dumps(predict_data), content_type="application/json"
        )
        self.assertEqual(predict_response.status_code, 200)
        self.assertIn("probability", predict_response.json())
        self.assertIsInstance(predict_response.json()["probability"], (int, float))

    def test_teach_with_split_brain_and_prediction(self):
        nodes = ["A", "B", "C"]
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]

        teach_data = {
            "nodes": nodes,
            "matrix": matrix,
            "split_brain": 1
        }
        teach_response = self.client.post(
            self.teach_url, data=json.dumps(teach_data), content_type="application/json"
        )
        self.assertEqual(teach_response.status_code, 200)
        self.assertIn("probability", teach_response.json())
        self.assertGreaterEqual(teach_response.json()["probability"], -2)

        predict_data = {
            "nodes": nodes,
            "matrix": matrix
        }
        predict_response = self.client.post(
            self.cluster_url, data=json.dumps(predict_data), content_type="application/json"
        )
        self.assertEqual(predict_response.status_code, 200)
        self.assertIn("probability", predict_response.json())
        self.assertEqual(predict_response.json()["probability"], 0)

    def test_multiple_requests_interaction(self):
        nodes = ["A", "B", "C"]
        matrix_1 = [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ]
        matrix_2 = [
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ]

        teach_data_1 = {
            "nodes": nodes,
            "matrix": matrix_1,
            "split_brain": 0
        }
        teach_response_1 = self.client.post(
            self.teach_url, data=json.dumps(teach_data_1), content_type="application/json"
        )
        self.assertEqual(teach_response_1.status_code, 200)
        self.assertIn("probability", teach_response_1.json())
        self.assertGreaterEqual(teach_response_1.json()["probability"], 0)

        predict_data_2 = {
            "nodes": nodes,
            "matrix": matrix_2
        }
        predict_response_2 = self.client.post(
            self.cluster_url, data=json.dumps(predict_data_2), content_type="application/json"
        )
        self.assertEqual(predict_response_2.status_code, 200)
        self.assertIn("probability", predict_response_2.json())
        self.assertEqual(predict_response_2.json()["probability"], -1)
