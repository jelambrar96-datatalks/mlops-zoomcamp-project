"""
Test case class for testing Flask application routes.
"""
import unittest
from app import app


class AppTestCase(unittest.TestCase):
    """    
    This class contains unit tests for the '/reload' and '/predict' routes
    of the Flask application defined in app.py. The tests ensure that the
    routes return the expected HTTP status codes and response content.
    """

    def setUp(self):
        """
        Set up the test client before each test.

        This method initializes the Flask test client, allowing us to send 
        HTTP requests to the application in a controlled test environment.
        It also sets the testing mode to True, which enables certain features 
        like better error messages during tests.
        """
        self.app = app.test_client()
        self.app.testing = True


    def test_01_reload_route(self):
        """
        Test the /reload route.

        This test sends a POST request to the /reload route and verifies 
        that the response status code is 200 (OK) and that the response 
        JSON contains a 'reloaded' key.
        """
        response = self.app.post('/reload')
        self.assertEqual(response.status_code, 200)
        self.assertIn('reloaded', response.get_json())


    def test_02_predict_route(self):
        """
        Test the /predict route.

        This test sends a POST request to the /predict route with a JSON 
        payload containing test data. It verifies that the response status 
        code is 200 (OK) and that the response JSON contains a 'prediction' key.
        """
        test_data = {'feature1': 1, 'feature2': 2}
        response = self.app.post('/predict', json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.get_json())


if __name__ == '__main__':
    unittest.main()
