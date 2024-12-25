from django.test import TestCase
from django.urls import reverse
import json


class TeachPageUnitTests(TestCase):
    def setUp(self):
        self.teach_url = reverse('teach')

    def test_teach_empty_nodes_and_matrix(self):
        data = {
            "nodes": [],
            "matrix": [],
            "split_brain": 0
        }
        response = self.client.post(
            self.teach_url, data=json.dumps(data), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)

    def test_teach_invalid_json_format(self):
        invalid_json = "{invalid: json, data}"
        response = self.client.post(
            self.teach_url, data=invalid_json, content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Невірний формат JSON.")

    def test_teach_split_brain_mismatch(self):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0]
            ],
            "split_brain": 0
        }
        response = self.client.post(
            self.teach_url, data=json.dumps(data), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertEqual(response.json()["probability"], -1)

    def test_teach_valid_input(self):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]
            ],
            "split_brain": 0
        }
        response = self.client.post(
            self.teach_url, data=json.dumps(data), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertGreaterEqual(response.json()["probability"], 0)

    def test_teach_dead_cluster(self):
        data = {
            "nodes": ["A", "B", "C"],
            "matrix": [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            "split_brain": 0
        }
        response = self.client.post(
            self.teach_url, data=json.dumps(data), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertEqual(response.json()["probability"], -1)