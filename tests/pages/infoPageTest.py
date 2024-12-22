from django.test import TestCase, Client

class InfoViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = '/info/'

    def test_get_request(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'info.html')

    def test_post_request_not_allowed(self):
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, 200)
