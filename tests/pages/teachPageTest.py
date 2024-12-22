from django.test import TestCase, Client
import json

class TeachViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = '/teach/'

    def test_get_request(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'teaching.html')

    def test_post_request_with_valid_data(self):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ],
            "split_brain": 1
        }
        response = self.client.post(self.url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertIsInstance(response.json()["probability"], (int, float))

    def test_post_request_with_split_brain_mismatch(self):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]
            ],
            "split_brain": 1
        }
        response = self.client.post(self.url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertEqual(response.json()["probability"], -2)

    def test_post_request_with_invalid_json(self):
        response = self.client.post(self.url, data="INVALID_JSON", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Невірний формат JSON.")

    def test_post_request_with_large_matrix(self):
        nodes = [f"Node{i}" for i in range(20)]
        matrix = [[1 if i != j else 0 for j in range(20)] for i in range(20)]
        data = {
            "nodes": nodes,
            "matrix": matrix,
            "split_brain": 1
        }
        response = self.client.post(self.url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertIsInstance(response.json()["probability"], (int, float))

    def test_post_request_with_sparse_matrix(self):
        nodes = ["A", "B", "C", "D"]
        matrix = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        data = {
            "nodes": nodes,
            "matrix": matrix,
            "split_brain": 0
        }
        response = self.client.post(self.url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertIsInstance(response.json()["probability"], (int, float))
